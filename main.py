"""
nba-video-analytics CLI entry point.

Usage:
    python main.py --config config/default.yaml --input <video_path>
"""
import argparse
from pathlib import Path

import yaml

from basketball.annotators.ball_annotator import BallAnnotator
from basketball.annotators.event_annotator import EventAnnotator
from basketball.annotators.keypoint_annotator import KeypointAnnotator
from basketball.annotators.metrics_annotator import MetricsAnnotator
from basketball.annotators.minimap_annotator import MinimapAnnotator
from basketball.annotators.player_annotator import PlayerAnnotator
from basketball.annotators.possession_annotator import PossessionAnnotator
from basketball.detection.ball_detector import BallDetector
from basketball.detection.player_detector import PlayerDetector
from basketball.events.event_detector import EventDetector
from basketball.homography.court_mapper import CourtMapper
from basketball.keypoints.court_keypoints import CourtKeypoints
from basketball.metrics.player_metrics import PlayerMetrics
from basketball.possession.possession_io import possession_ball_input
from basketball.possession.possession_tracker import PossessionTracker
from basketball.team_classifier.classifier import (
    EmbeddingClusteringClassifier,
    FashionCLIPClassifier,
    KMeansClassifier,
)
from basketball.utils.io_utils import get_video_metadata, load_video, write_video


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single-camera NBA video analytics pipeline.'
    )
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/outputs')
    return parser.parse_args()


ALLOWED_CLIP_KEYS = {'team_1_description', 'team_2_description', 'team_1_colour', 'team_2_colour'}


def resolve_clip_settings(config: dict, clip_name: str) -> dict:
    """Resolve one clip's team prompts and annotator colours, falling back to team_classifier's description/colour defaults where no per-clip override exists."""
    tc_cfg = config['team_classifier']
    defaults = {
        # .get(key) rather than plain subscription: this function runs
        # for every method, including kmeans, where the descriptions are
        # unused: a kmeans-only config that never defines them must not
        # raise KeyError before any classification runs. A missing
        # description resolves to None here; require_fashionclip_descriptions()
        # raises a clear, method-naming error at the point fashionclip
        # actually needs one.
        'team_1_description': tc_cfg.get('team_1_description'),
        'team_2_description': tc_cfg.get('team_2_description'),
        'team_1_colour': tuple(tc_cfg.get('team_1_colour', PlayerAnnotator.DEFAULT_TEAM_1_COLOUR)),
        'team_2_colour': tuple(tc_cfg.get('team_2_colour', PlayerAnnotator.DEFAULT_TEAM_2_COLOUR)),
    }

    # `config.get('clips', {})` would return None, not {}, for a
    # present-but-empty `clips:` block in the YAML (the default only kicks
    # in when the key is absent entirely, not when its value is falsy);
    # `.get(clip_name)` on that None would then crash with AttributeError.
    clip_entry = (config.get('clips') or {}).get(clip_name)
    if clip_entry is None:
        print(
            f'[main] No clips[{clip_name!r}] override — using team_classifier defaults: '
            f'{defaults["team_1_description"]!r} / {defaults["team_2_description"]!r}'
        )
        return defaults

    # An unknown key inside a clip entry raises rather than being silently
    # ignored: it is very likely a typo, and silent acceptance would never
    # catch it.
    unknown_keys = set(clip_entry) - ALLOWED_CLIP_KEYS
    if unknown_keys:
        raise ValueError(
            f'clips[{clip_name!r}] has unknown key(s) {sorted(unknown_keys)}; '
            f'expected a subset of {sorted(ALLOWED_CLIP_KEYS)}.'
        )

    resolved = {**defaults, **clip_entry}
    resolved['team_1_colour'] = tuple(resolved['team_1_colour'])
    resolved['team_2_colour'] = tuple(resolved['team_2_colour'])
    print(
        f'[main] clips[{clip_name!r}] override in use: '
        f'{resolved["team_1_description"]!r} / {resolved["team_2_description"]!r}'
    )
    return resolved


def require_fashionclip_descriptions(clip_settings: dict, clip_name: str) -> None:
    """Raise a clear, method-naming error if a resolved clip's team descriptions are missing (required by fashionclip, unlike kmeans)."""
    missing = [key for key in ('team_1_description', 'team_2_description') if clip_settings.get(key) is None]
    if missing:
        raise ValueError(
            f'team_classifier method \'fashionclip\' requires {sorted(missing)} to be set, but '
            f'got None for clip {clip_name!r} — set team_classifier.<key> or clips[{clip_name!r}].<key> in config.'
        )


def load_config(path: str) -> dict:
    """Load and return the YAML pipeline configuration from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f'Config file does not exist: {path}')

    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as error:
        raise ValueError(f'Config at {path} is not valid YAML: {error}') from error

    if not isinstance(config, dict):
        raise ValueError(f'Config at {path} did not parse to a mapping, got {type(config).__name__}.')

    if 'team_classifier' not in config:
        raise KeyError(f'Config at {path} has no \'team_classifier\' section.')

    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    print(f'[main] Config loaded from {args.config}')
    print(f'[main] Input video: {args.input}')

    video_path = args.input
    clip_name = Path(video_path).stem
    cache_dir = f'data/processed/{clip_name}'
    output_path = str(Path(args.output) / f'{clip_name}_annotated.avi')

    if not Path(video_path).exists():
        raise FileNotFoundError(f'Input video does not exist: {video_path}')

    print('[main] Loading frames...')
    frames = [frame for _, frame in load_video(video_path)]
    metadata = get_video_metadata(video_path)
    print(f'[main] {len(frames)} frames loaded at {metadata["fps"]:.2f} fps')

    player_weights = 'models/ball.pt'
    ball_weights = 'models/ball.pt'
    # Keypoint weights are deliberately NOT checked here. This loop runs before
    # any frame is processed, so a missing keypoint checkpoint would abort
    # Stages 1-6 as well: the whole pipeline stopping for the one stage that
    # needs the file. CourtKeypoints.load_model() raises a clear error naming
    # both the path and the stage at the point it is actually required.
    for weights in (player_weights, ball_weights):
        if not Path(weights).exists():
            raise FileNotFoundError(f'Model weights do not exist: {weights}')

    kp_cfg = config['keypoints']

    player_detector = PlayerDetector(model_path=player_weights)
    ball_detector = BallDetector(model_path=ball_weights)
    # config's keypoints.confidence_threshold maps to the INSTANCE threshold
    # only, of the two CourtKeypoints takes. That is the one passed to
    # inference, so it determines the cached predictions and is part of the
    # cache fingerprint. The per-keypoint threshold is deliberately left at its
    # default: it is applied after retrieval by filter_keypoints(), which is
    # NOT called anywhere below; the A3 filtering-policy ablation was not
    # run, so that value is untuned and applying it here would
    # present an unmeasured policy as a chosen one. Fingerprinting it would
    # also force full re-inference for every value a later ablation sweeps.
    # What is drawn is governed by the annotator's own render threshold.
    court_keypoints = CourtKeypoints(
        model_path=kp_cfg['weights'],
        instance_confidence_threshold=kp_cfg['confidence_threshold'],
    )

    print('[main] Running player detection and tracking...')
    player_tracks = player_detector.run_tracking(
        frames=frames,
        video_path=video_path,
        cache_path=f'{cache_dir}/player_detections.pkl',
    )

    print('[main] Running ball detection...')
    raw_ball_detections = ball_detector.run_detection(
        frames=frames,
        video_path=video_path,
        cache_path=f'{cache_dir}/ball_detections.pkl',
    )

    print('[main] Detecting court keypoints...')
    # Grouped with the other detectors rather than placed near the annotator
    # that consumes it: this is an independent inference pass over the frames,
    # depending on nothing upstream. Stage 8's CourtMapper consumes it, and
    # the annotator also reads it.
    keypoints_per_frame = court_keypoints.run_detection(
        frames=frames,
        video_path=video_path,
        cache_path=f'{cache_dir}/keypoints.pkl',
    )

    print('[main] Interpolating ball positions...')
    # possession_ball_input() is the single source of truth for the adopted
    # gate followed by interpolation, in this order; call it here so this
    # ordering can never drift from the one that possession's own tests pin.
    ball_input = possession_ball_input(raw_ball_detections)
    ball_detections = ball_input.filled

    print('[main] Classifying teams...')
    tc_cfg = config['team_classifier']
    method = tc_cfg['method']
    clip_settings = resolve_clip_settings(config, clip_name)

    if method not in ('fashionclip', 'kmeans', 'embedding'):
        # Anything unrecognised must raise: falling through to the K-means
        # baseline would silently report results from a method the config did
        # not ask for.
        raise ValueError(
            f'Unknown team_classifier method {method!r}; '
            f'expected \'fashionclip\', \'kmeans\' or \'embedding\'.'
        )

    if method == 'fashionclip':
        require_fashionclip_descriptions(clip_settings, clip_name)
        team_classifier = FashionCLIPClassifier(
            team_1_description=clip_settings['team_1_description'],
            team_2_description=clip_settings['team_2_description'],
            reset_interval=tc_cfg['reset_interval'],
            # crop_fraction (unqualified) is FashionCLIP's own key, despite
            # the generic name; K-means and embedding read their own
            # kmeans_crop_fraction/embedding_crop_fraction below instead.
            crop_fraction=tc_cfg['crop_fraction'],
            confidence_threshold=tc_cfg['fashionclip_confidence_threshold'],
            # Off: the 11 August 2026 sweep measured this policy 2.7pp
            # worse than deciding every frame fresh.
            use_assignment_cache=False,
            aggregation=tc_cfg['aggregation'],
            aggregation_window=tc_cfg['aggregation_window'],
        )
    elif method == 'embedding':
        # Prompt-free by construction: the encoder below is built only to
        # reach get_image_embeddings(), and require_fashionclip_descriptions()
        # is deliberately NOT called for this method. The two description
        # arguments are placeholders the encoder's unused prompted path
        # would need; no text is ever encoded or compared here.
        encoder = FashionCLIPClassifier(
            team_1_description='',
            team_2_description='',
            reset_interval=tc_cfg['reset_interval'],
            # This arm's own measured-best crop, not FashionCLIP's
            # crop_fraction; the two were swept independently.
            crop_fraction=tc_cfg['embedding_crop_fraction'],
            confidence_threshold=tc_cfg['fashionclip_confidence_threshold'],
        )
        team_classifier = EmbeddingClusteringClassifier(
            encoder=encoder,
            fit_stride=tc_cfg['embedding_fit_stride'],
            margin_threshold=tc_cfg['embedding_margin_threshold'],
        )
    else:
        # K-means fits colour clusters directly from pixels: it has no
        # notion of a text prompt, so clip_settings' descriptions are unused
        # here (they only ever apply to FashionCLIP).
        team_classifier = KMeansClassifier(
            # K-means' own measured-best crop, not FashionCLIP's
            # crop_fraction; the two were swept independently.
            crop_fraction=tc_cfg['kmeans_crop_fraction'],
            fit_stride=tc_cfg['kmeans_fit_stride'],
            bg_distance_threshold=tc_cfg['kmeans_background_distance_threshold'],
            margin_threshold=tc_cfg['kmeans_margin_threshold'],
        )

    team_assignment = team_classifier.assign_teams(
        frames,
        player_tracks,
        cache_path=f'{cache_dir}/team_assignment_{method}.pkl',
    )

    print('[main] Assigning possession...')
    possession_tracker = PossessionTracker(config)
    # Possession consumes ball_input.filled (aliased to ball_detections above),
    # the same gated-then-interpolated track produced by
    # possession_io.possession_ball_input() and built only once, matching
    # what its measured accuracy was produced against and what every other
    # downstream consumer in this function sees.
    possession = possession_tracker.assign_possession(
        player_tracks,
        ball_detections,
        cache_path=f'{cache_dir}/possession.pkl',
    )

    print('[main] Detecting pass and interception events...')
    # EventDetector takes its default MAX_TRANSITION_GAP: the leave-one-clip-out
    # sweep found the parameter inert at or above 20 and the folds disagreed, so
    # no value was adopted. There is deliberately no config key for it: one
    # would imply a tuned parameter where none exists.
    event_detector = EventDetector()
    # Three traversals of the same data, deliberately: it keeps the public
    # interface fixed, and the cost is microseconds.
    passes = event_detector.identify_passes(possession, team_assignment)
    interceptions = event_detector.identify_interceptions(possession, team_assignment)
    unclassified = event_detector.identify_unclassified(possession, team_assignment)
    print(f'[main] {len(passes)} passes, {len(interceptions)} interceptions, '
          f'{len(unclassified)} unclassified transitions')

    print('[main] Mapping players onto the court plane...')
    # CourtMapper takes its OWN default per-keypoint threshold. Deliberately
    # NOT kp_cfg['confidence_threshold']: that key is CourtKeypoints' INSTANCE
    # threshold, gating whether a court is detected at all, and is already
    # passed above. CourtMapper's is the per-keypoint threshold selecting which
    # landmarks are trusted as correspondences, a different quantity that
    # happens to share a default of 0.5, so passing the config value here would
    # look correct and be wrong. No config key is added for it either, since no
    # value has been selected by measurement.
    court_mapper = CourtMapper()
    court_positions, mapping_report = court_mapper.map_to_court(player_tracks, keypoints_per_frame)
    # The availability figure for this stage, and this run is where it is
    # produced. Roughly 10% of frames carry too few confident keypoints to fit
    # a homography at all, and on clip_3 that shortfall is one contiguous
    # 45-frame run; an empty dict on those frames is the correct output, not
    # a fault, so the counts are printed rather than left to be inferred.
    # summary() carries its own [court] prefix, so it is printed bare rather
    # than wrapped in a second one.
    print(mapping_report.summary())

    print('[main] Measuring speed and distance...')
    # fps comes from the video metadata, never a literal. PlayerMetrics takes
    # it as a required argument with no default precisely so this cannot be
    # got wrong silently: a 25.0 default on this project's 30 fps footage
    # would report every speed 20% low, a plausible-looking number rather
    # than an error.
    #
    # speed_window is the one Stage 9 parameter in config. court_length_m and
    # court_width_m are deliberately NOT read: PlayerMetrics performs no unit
    # conversion, because CourtMapper already returns metres.
    player_metrics = PlayerMetrics(
        fps=metadata['fps'],
        speed_window=config['metrics']['speed_window'],
    )
    distances, speeds, metrics_report = player_metrics.compute(court_positions)
    # This stage's availability figure, in the same spirit as the mapping
    # report above: roughly 0.5% of displacements span a gap longer than
    # MAX_SPEED_GAP and so carry distance but no speed, which is the designed
    # outcome rather than a fault. summary() carries its own [metrics] prefix.
    print(metrics_report.summary())

    print('[main] Annotating frames...')
    player_annotator = PlayerAnnotator(
        team_1_colour=clip_settings['team_1_colour'],
        team_2_colour=clip_settings['team_2_colour'],
    )
    ball_annotator = BallAnnotator()
    possession_annotator = PossessionAnnotator()
    event_annotator = EventAnnotator()
    keypoint_annotator = KeypointAnnotator()
    metrics_annotator = MetricsAnnotator()
    # The template path IS a config key, unlike CourtMapper's per-keypoint
    # threshold: a different court render or a different install location is a
    # genuine deployment setting, changed without editing code. The threshold
    # is absent from config because no value has been selected by measurement,
    # which is a different kind of reason.
    #
    # Read by direct subscription, matching how team_classifier and keypoints
    # are read: a missing key raises rather than silently falling back to the
    # constructor default, which would leave the config saying one thing and
    # the run doing another. `clips` uses .get() only because it is genuinely
    # optional, and this is not.
    minimap_annotator = MinimapAnnotator(
        template_path=config['homography']['court_template_path'],
        team_1_colour=clip_settings['team_1_colour'],
        team_2_colour=clip_settings['team_2_colour'],
    )

    # Corner ownership: PossessionAnnotator has the top-left, EventAnnotator
    # the top-right, MinimapAnnotator the bottom-left panel.
    #
    # Keypoints are drawn FIRST, under everything else. They are court
    # landmarks rather than moving subjects, so they sit at the frame edges and
    # on painted lines where the players are not, and drawing them last would
    # let a keypoint dot and its index label land on a caption. Being
    # underneath costs nothing here: a 4-pixel dot is not obscured by a player
    # ellipse it does not overlap.
    annotated_frames = keypoint_annotator.draw(frames, keypoints_per_frame)
    annotated_frames = player_annotator.draw(annotated_frames, player_tracks, team_assignment)
    annotated_frames = ball_annotator.draw(annotated_frames, ball_detections)
    annotated_frames = possession_annotator.draw(annotated_frames, player_tracks, possession, ball_detections)
    annotated_frames = event_annotator.draw(annotated_frames, passes, interceptions, unclassified)
    # Metrics text attaches to each player's own box, which keeps it clear of
    # the two top captions. That is NOT sufficient against the minimap: it is
    # composited after this call and simply overwrites the bottom-left, so a
    # player near that sideline had their speed drawn and then covered. The
    # panel's rectangle is therefore passed in, and the annotator flips the
    # text above the box there exactly as it already does at the frame edge.
    #
    # Passing the region rather than reordering the two draws: the minimap must
    # stay last, because it is Stage 8's verification mechanism and per-player
    # text drawn over the tactical panel would obscure the thing the panel
    # exists to prove. The region is read from the annotator that owns it, so
    # the reserved rectangle and the composited one cannot drift apart.
    annotated_frames = metrics_annotator.draw(
        annotated_frames, player_tracks, distances, speeds,
        reserved_region=minimap_annotator.occupied_region_for(frames),
    )
    # The minimap is drawn LAST, over everything else, in the bottom-left no
    # other annotator uses, so it obscures neither caption. Being on top keeps
    # the tactical view legible rather than showing player ellipses through it,
    # and it is the only annotator whose output is read as a panel rather than
    # as marks on the broadcast image.
    #
    # frame_has_homography comes from the report, never from
    # bool(court_positions[i]): an empty dict has three causes and only one is
    # a failure, so deriving it would caption correctly-mapped frames as
    # failures on the very output this stage is verified by.
    annotated_frames = minimap_annotator.draw(
        annotated_frames,
        court_positions,
        mapping_report.frame_has_homography,
        team_assignment,
    )

    print(f'[main] Writing output to {output_path}...')
    write_video(annotated_frames, output_path, fps=metadata['fps'])
    print('[main] Done.')


if __name__ == '__main__':
    main()