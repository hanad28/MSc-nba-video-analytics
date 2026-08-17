"""Unit tests for main.py's per-clip config resolution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import main
from basketball.annotators.minimap_annotator import MinimapAnnotator
from basketball.annotators.player_annotator import PlayerAnnotator
from basketball.keypoints.court_keypoints import CourtKeypoints
from basketball.metrics.player_metrics import DEFAULT_SPEED_WINDOW, PlayerMetrics
from basketball.team_classifier.classifier import (
    EmbeddingClusteringClassifier,
    FashionCLIPClassifier,
    KMeansClassifier,
)
from main import require_fashionclip_descriptions, resolve_clip_settings


def base_config() -> dict:
    return {
        'team_classifier': {
            'team_1_description': 'white basketball jersey',
            'team_2_description': 'dark blue basketball jersey',
        },
        'clips': {
            'clip_3': {
                'team_1_description': 'white basketball jersey',
                'team_2_description': 'red basketball jersey',
                'team_2_colour': [0, 0, 200],
            },
        },
    }


def test_a_clip_with_an_override_uses_it():
    resolved = resolve_clip_settings(base_config(), 'clip_3')

    assert resolved['team_1_description'] == 'white basketball jersey'
    assert resolved['team_2_description'] == 'red basketball jersey'
    assert resolved['team_2_colour'] == (0, 0, 200)


def test_an_override_omitting_a_colour_falls_back_to_the_team_classifier_default():
    # clip_3's entry overrides team_2_colour but not team_1_colour.
    resolved = resolve_clip_settings(base_config(), 'clip_3')

    assert resolved['team_1_colour'] == PlayerAnnotator.DEFAULT_TEAM_1_COLOUR


def test_a_clip_absent_from_the_config_falls_back_to_team_classifier_defaults():
    resolved = resolve_clip_settings(base_config(), 'clip_1')

    assert resolved['team_1_description'] == 'white basketball jersey'
    assert resolved['team_2_description'] == 'dark blue basketball jersey'
    assert resolved['team_1_colour'] == PlayerAnnotator.DEFAULT_TEAM_1_COLOUR
    assert resolved['team_2_colour'] == PlayerAnnotator.DEFAULT_TEAM_2_COLOUR


def test_missing_clips_section_entirely_falls_back_to_defaults():
    config = {'team_classifier': base_config()['team_classifier']}

    resolved = resolve_clip_settings(config, 'clip_1')

    assert resolved['team_1_description'] == 'white basketball jersey'


def test_a_present_but_empty_clips_section_falls_back_to_defaults():
    # A YAML `clips:` block with nothing under it parses to None, not {};
    # config.get('clips', {}) only substitutes the default when the key is
    # absent, so this must be handled explicitly rather than crashing with
    # AttributeError on None.get(clip_name).
    config = {'team_classifier': base_config()['team_classifier'], 'clips': None}

    resolved = resolve_clip_settings(config, 'clip_1')

    assert resolved['team_1_description'] == 'white basketball jersey'
    assert resolved['team_2_description'] == 'dark blue basketball jersey'


def test_an_unknown_key_in_a_clip_entry_raises():
    config = base_config()
    config['clips']['clip_3']['unexpected_key'] = 'oops'

    with pytest.raises(ValueError, match='unknown key'):
        resolve_clip_settings(config, 'clip_3')


def test_team_classifier_section_colour_overrides_flow_through_as_the_fallback():
    config = base_config()
    config['team_classifier']['team_1_colour'] = [9, 9, 9]

    resolved = resolve_clip_settings(config, 'clip_1')

    assert resolved['team_1_colour'] == (9, 9, 9)


def test_a_kmeans_only_config_without_any_descriptions_resolves_without_raising():
    # kmeans has no notion of a text prompt, so a config that never defines
    # team_1_description/team_2_description anywhere must not raise KeyError
    # before any classification runs; resolve_clip_settings() now runs for
    # every method, not just fashionclip.
    config = {'team_classifier': {'method': 'kmeans'}}

    resolved = resolve_clip_settings(config, 'clip_1')

    assert resolved['team_1_description'] is None
    assert resolved['team_2_description'] is None


def test_require_fashionclip_descriptions_passes_when_both_are_set():
    require_fashionclip_descriptions(
        {'team_1_description': 'white basketball jersey', 'team_2_description': 'red basketball jersey'},
        'clip_1',
    )  # must not raise


def test_require_fashionclip_descriptions_raises_naming_the_missing_key():
    clip_settings = {'team_1_description': 'white basketball jersey', 'team_2_description': None}

    with pytest.raises(ValueError, match="fashionclip.*team_2_description"):
        require_fashionclip_descriptions(clip_settings, 'clip_1')


def test_require_fashionclip_descriptions_raises_naming_both_missing_keys():
    clip_settings = {'team_1_description': None, 'team_2_description': None}

    with pytest.raises(ValueError, match='team_1_description.*team_2_description'):
        require_fashionclip_descriptions(clip_settings, 'clip_1')


# --- three-way method selection ---------------------------------------------

def test_an_embedding_only_config_without_any_descriptions_resolves_without_raising():
    # The embedding arm is prompt-free by construction, so like kmeans it
    # must resolve without descriptions ever being defined.
    config = {'team_classifier': {'method': 'embedding'}}

    resolved = resolve_clip_settings(config, 'clip_1')

    assert resolved['team_1_description'] is None
    assert resolved['team_2_description'] is None


def test_the_three_accepted_methods_are_wired_and_anything_else_raises():
    # Guards the selector in main.run_pipeline(): reading its source keeps
    # this honest without needing video, weights or a GPU to run the
    # pipeline itself.
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert "if method not in ('fashionclip', 'kmeans', 'embedding'):" in source
    assert 'Unknown team_classifier method' in source
    assert 'EmbeddingClusteringClassifier(' in source
    # The 11 August 2026 sweep found per-player memoisation 2.7pp worse
    # than deciding every frame fresh -- production must construct the
    # classifier with it explicitly off, not merely rely on the default.
    assert 'use_assignment_cache=False' in source
    # The adopted temporal aggregation must come from config, not a default
    # the config file cannot see.
    assert "aggregation=tc_cfg['aggregation']" in source
    assert "aggregation_window=tc_cfg['aggregation_window']" in source


def test_the_embedding_branch_never_requires_descriptions():
    source = Path(main.__file__).read_text(encoding='utf-8')
    embedding_branch = source.split("elif method == 'embedding':")[1].split('    else:')[0]
    # Comments are stripped before the check: the branch's own comment names
    # require_fashionclip_descriptions() to explain why it is NOT called, and
    # a substring match would read that explanation as the call itself.
    code_lines = [line for line in embedding_branch.splitlines() if not line.strip().startswith('#')]
    code = '\n'.join(code_lines)

    assert 'require_fashionclip_descriptions' not in code
    assert "tc_cfg['embedding_fit_stride']" in code
    assert "tc_cfg['embedding_margin_threshold']" in code


def test_the_default_config_defines_every_key_each_method_reads():
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))
    tc_cfg = config['team_classifier']

    assert tc_cfg['method'] in ('fashionclip', 'kmeans', 'embedding')
    for key in (
        'crop_fraction',
        'reset_interval',
        'fashionclip_confidence_threshold',
        'aggregation',
        'aggregation_window',
        'kmeans_fit_stride',
        'kmeans_crop_fraction',
        'kmeans_background_distance_threshold',
        'kmeans_margin_threshold',
        'embedding_fit_stride',
        'embedding_crop_fraction',
        'embedding_margin_threshold',
    ):
        assert key in tc_cfg, f'config/default.yaml is missing team_classifier.{key}'
    # The retired parameter must be gone, not left to mislead.
    assert 'kmeans_fit_frames' not in tc_cfg


def test_the_default_config_pins_the_adopted_sweep_values():
    # A future accidental edit reverting these has no other test to catch
    # it -- main.py only ever reads whatever is in the file, it never
    # hardcodes the adopted numbers. Pins the 11 August 2026 sweep result.
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))
    tc_cfg = config['team_classifier']

    assert tc_cfg['method'] == 'fashionclip'
    assert tc_cfg['crop_fraction'] == 1.0  # FashionCLIP's own key, despite the generic name
    assert tc_cfg['fashionclip_confidence_threshold'] == 0.5
    assert tc_cfg['team_1_description'] == 'white NBA jersey'
    assert tc_cfg['team_2_description'] == 'dark blue NBA jersey'
    # Each OTHER method's own measured-best crop, not FashionCLIP's 1.0;
    # see the crop_fraction-per-method regression test below for why this
    # split exists at all.
    assert tc_cfg['kmeans_crop_fraction'] == 0.667
    assert tc_cfg['embedding_crop_fraction'] == 0.5
    # The adopted temporal aggregation: 0.9443 mean held-out effective
    # accuracy against 0.9242 for per-frame decisions.
    assert tc_cfg['aggregation'] == 'window_weighted'
    assert tc_cfg['aggregation_window'] == 9

    clips = config['clips']
    assert clips['clip_1']['team_1_description'] == 'white NBA jersey'
    assert clips['clip_1']['team_2_description'] == 'dark blue NBA jersey'
    assert clips['clip_2']['team_1_description'] == 'white NBA jersey'
    assert clips['clip_2']['team_2_description'] == 'dark blue NBA jersey'
    assert clips['clip_3']['team_1_description'] == 'white NBA jersey'
    assert clips['clip_3']['team_2_description'] == 'red NBA jersey'


def test_the_default_config_pins_the_retained_possession_values():
    # These are the swept-and-retained pre-sweep values: the 240-config
    # possession sweep found no adoptable optimum, so they were deliberately
    # kept. Changing any of them invalidates the phase's measured possession
    # results, and nothing else would catch it -- PossessionTracker.__init__
    # raises only on an ABSENT key, never on a changed value, and the
    # possession unit tests use their own deliberately different fixture
    # values so they can exercise the logic independently of production.
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    assert config['possession'] == {
        'max_ball_distance': 50,
        'hold_threshold': 5,
        'bbox_overlap_min': 0.8,
    }


def test_resolve_clip_settings_against_the_real_config_uses_the_nba_jersey_scheme():
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    resolved = resolve_clip_settings(config, 'clip_3')

    assert resolved['team_1_description'] == 'white NBA jersey'
    assert resolved['team_2_description'] == 'red NBA jersey'


def test_each_method_constructed_from_the_real_config_gets_its_own_crop_fraction():
    """
    Regression test for the shared-crop_fraction bug: main.py's three
    construction sites previously all read the one team_classifier.
    crop_fraction key, so shipping FashionCLIP's adopted 1.0 there silently
    moved K-means and EmbeddingClusteringClassifier off their own
    measured-best crops too. Mirrors main.py's three construction calls
    exactly (same tc_cfg keys) without running the full pipeline.
    """
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))
    tc_cfg = config['team_classifier']

    fashionclip = FashionCLIPClassifier(
        team_1_description=tc_cfg['team_1_description'],
        team_2_description=tc_cfg['team_2_description'],
        reset_interval=tc_cfg['reset_interval'],
        crop_fraction=tc_cfg['crop_fraction'],
        confidence_threshold=tc_cfg['fashionclip_confidence_threshold'],
    )
    kmeans = KMeansClassifier(
        crop_fraction=tc_cfg['kmeans_crop_fraction'],
        fit_stride=tc_cfg['kmeans_fit_stride'],
        bg_distance_threshold=tc_cfg['kmeans_background_distance_threshold'],
        margin_threshold=tc_cfg['kmeans_margin_threshold'],
    )
    embedding_encoder = FashionCLIPClassifier(
        team_1_description='',
        team_2_description='',
        reset_interval=tc_cfg['reset_interval'],
        crop_fraction=tc_cfg['embedding_crop_fraction'],
        confidence_threshold=tc_cfg['fashionclip_confidence_threshold'],
    )
    embedding = EmbeddingClusteringClassifier(
        encoder=embedding_encoder,
        fit_stride=tc_cfg['embedding_fit_stride'],
        margin_threshold=tc_cfg['embedding_margin_threshold'],
    )

    assert fashionclip.crop_fraction == 1.0
    assert kmeans.crop_fraction == 0.667
    assert embedding.encoder.crop_fraction == 0.5
    # Each method's own value, not all three collapsed onto one shared number.
    assert len({fashionclip.crop_fraction, kmeans.crop_fraction, embedding.encoder.crop_fraction}) == 3


# --- possession wiring --------------------------------------------------

def test_possession_tracker_is_constructed_and_assigned_after_team_classification():
    # Guards main.run_pipeline()'s ordering by reading its source, the same
    # approach the method-selector test above uses -- no video, weights or
    # GPU needed. Possession must consume the same ball_detections variable
    # already reassigned by the gate + fill_missing() calls above it, not a
    # separately rebuilt track.
    source = Path(main.__file__).read_text(encoding='utf-8')

    team_classification_idx = source.index('team_assignment = team_classifier.assign_teams(')
    possession_idx = source.index('PossessionTracker(config)')
    annotation_idx = source.index('player_annotator = PlayerAnnotator(')

    assert team_classification_idx < possession_idx < annotation_idx
    assert 'possession_tracker.assign_possession(\n        player_tracks,\n        ball_detections,' in source


def test_possession_cache_path_follows_the_per_clip_convention():
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert "cache_path=f'{cache_dir}/possession.pkl'" in source


def test_possession_annotator_is_drawn_after_ball_annotator():
    source = Path(main.__file__).read_text(encoding='utf-8')

    ball_draw_idx = source.index("ball_annotator.draw(annotated_frames, ball_detections)")
    possession_draw_idx = source.index('possession_annotator.draw(')

    assert ball_draw_idx < possession_draw_idx


# --- event wiring -------------------------------------------------------

def test_event_detection_runs_after_possession_and_before_annotation():
    source = Path(main.__file__).read_text(encoding='utf-8')

    possession_idx = source.index('possession_tracker.assign_possession(')
    detector_idx = source.index('event_detector = EventDetector()')
    annotation_idx = source.index('player_annotator = PlayerAnnotator(')

    assert possession_idx < detector_idx < annotation_idx
    for call in (
        'event_detector.identify_passes(possession, team_assignment)',
        'event_detector.identify_interceptions(possession, team_assignment)',
        'event_detector.identify_unclassified(possession, team_assignment)',
    ):
        assert call in source


def test_event_annotator_is_drawn_after_possession_annotator():
    source = Path(main.__file__).read_text(encoding='utf-8')

    possession_draw_idx = source.index('possession_annotator.draw(')
    event_draw_idx = source.index('event_annotator.draw(')

    assert possession_draw_idx < event_draw_idx


def test_no_cache_path_is_passed_to_any_identify_call():
    # Event detection is pure computation over two in-memory lists; a cache
    # would need a fingerprint and a revision constant for no benefit.
    source = Path(main.__file__).read_text(encoding='utf-8')

    for call in ('identify_passes(', 'identify_interceptions(', 'identify_unclassified('):
        # index() rather than a bare `in` check: the call must exist for the
        # cache_path assertion below to mean anything at all.
        start = source.index(call)
        assert 'cache_path' not in source[start:source.index(')', start)]


def test_event_detector_is_constructed_without_a_configured_gap():
    # MAX_TRANSITION_GAP was swept leave-one-clip-out and found inert at or
    # above 20 with the folds disagreeing, so it stands untuned -- a config key
    # would imply a validated value that does not exist.
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert 'EventDetector()' in source
    assert 'max_transition_gap' not in source


# --- Stage 7 wiring -------------------------------------------------------

class RecordingDetector:
    """A detector stand-in returning one entry per supplied frame and recording every call."""

    def __init__(self, per_frame_factory: Callable[[int], object]) -> None:
        # One entry per frame rather than a fixed-length list: a stub whose
        # output length ignores its input cannot catch a frame-alignment bug,
        # which has been the gap in several fixtures this phase.
        self.per_frame_factory = per_frame_factory
        self.model_path = None
        self.construction_kwargs: dict = {}
        self.calls: list[dict] = []

    def __call__(self, *args: object, **kwargs: object) -> 'RecordingDetector':
        """Stand in for the class itself, recording the weights path and every other construction argument."""
        self.model_path = kwargs.get('model_path', args[0] if args else None)
        self.construction_kwargs = dict(kwargs)
        return self

    def run_detection(self, **kwargs: object) -> list:
        """Record the call and return one entry per supplied frame."""
        self.calls.append(kwargs)
        return [self.per_frame_factory(index) for index in range(len(kwargs['frames']))]

    def run_tracking(self, **kwargs: object) -> list:
        """Record the call and return one entry per supplied frame."""
        self.calls.append(kwargs)
        return [self.per_frame_factory(index) for index in range(len(kwargs['frames']))]


class RecordingAnnotator:
    """An annotator stand-in returning one frame per supplied frame and recording every draw call."""

    # Carried because resolve_clip_settings() reads them off PlayerAnnotator
    # for its colour defaults: a stand-in missing an attribute its caller uses
    # cannot exercise the caller at all.
    DEFAULT_TEAM_1_COLOUR = PlayerAnnotator.DEFAULT_TEAM_1_COLOUR
    DEFAULT_TEAM_2_COLOUR = PlayerAnnotator.DEFAULT_TEAM_2_COLOUR

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.construction_kwargs: dict = {}
        # Keyword arguments are recorded separately from positional ones:
        # reserved_region is passed by keyword, and a stand-in that swallowed
        # it into *args could not tell 'passed' from 'defaulted'.
        self.draw_kwargs: list[dict] = []
        self.region_requests: list[list] = []

    def __call__(self, *args: object, **kwargs: object) -> 'RecordingAnnotator':
        """Stand in for the class itself, recording every construction argument."""
        self.construction_kwargs = dict(kwargs)
        return self

    def draw(self, frames: list, *args: object, **kwargs: object) -> list:
        """Record the arguments and return a same-length list, as every real annotator does."""
        self.calls.append((list(frames), args))
        self.draw_kwargs.append(dict(kwargs))
        return [f'{frame}+drawn' for frame in frames]

    def occupied_region_for(self, frames: list) -> tuple[int, int, int, int]:
        """Stand in for MinimapAnnotator's published rectangle, returning a distinctive one."""
        # A recognisable rectangle rather than the real geometry: the wiring
        # test asserts this exact value reached the metrics annotator, so a
        # caller substituting its own guess at the region fails rather than
        # coincidentally matching.
        self.region_requests.append(list(frames))
        return (12, 596, 221, 708)


class StubClassifier:
    """Team classifier stand-in returning one empty assignment per frame."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def assign_teams(self, frames: list, tracks: list, cache_path: str) -> list:
        """Return one empty per-frame assignment mapping."""
        return [{} for _ in frames]


class StubPossession:
    """Possession stand-in returning one holder entry per tracked frame."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def assign_possession(self, tracks: list, ball: list, cache_path: str) -> list:
        """Return a no-holder result of the same length as the tracks."""
        return [-1 for _ in tracks]


class StubEvents:
    """Event detector stand-in emitting no events."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def identify_passes(self, possession: list, teams: list) -> list:
        """Return no passes."""
        return []

    def identify_interceptions(self, possession: list, teams: list) -> list:
        """Return no interceptions."""
        return []

    def identify_unclassified(self, possession: list, teams: list) -> list:
        """Return no unclassified transitions."""
        return []


class RecordingCourtMapper:
    """A CourtMapper stand-in returning one position dict per supplied frame and recording every call."""

    def __init__(self) -> None:
        self.construction_kwargs: dict = {}
        self.calls: list[tuple] = []
        # One flag per frame, with a False in the middle so a caller deriving
        # the flags from the positions instead of from the report produces a
        # visibly different answer. A stub whose report agrees with
        # bool(positions) could not catch that conflation at all.
        self.report = None

    def __call__(self, *args: object, **kwargs: object) -> 'RecordingCourtMapper':
        self.construction_kwargs = dict(kwargs)
        return self

    def map_to_court(self, player_tracks: list, keypoints_per_frame: list) -> tuple:
        """Record the call and return one entry per supplied frame, plus a report whose flags disagree with the positions."""
        self.calls.append((list(player_tracks), list(keypoints_per_frame)))
        n_frames = len(player_tracks)
        # Frame 0 is deliberately EMPTY but flagged as having a homography, and
        # the last frame is populated but flagged as having none. Either
        # mistake in main.py (deriving flags from positions, or passing the
        # wrong list) changes what the annotator receives.
        positions = [{} if index == 0 else {7: (5.0, 5.0)} for index in range(n_frames)]
        flags = [index != n_frames - 1 for index in range(n_frames)]
        self.report = SimpleNamespace(
            frame_has_homography=flags,
            summary=lambda: '[court] stub mapping report',
        )
        return positions, self.report


class RecordingPlayerMetrics:
    """A PlayerMetrics stand-in returning one distance and one speed entry per supplied frame and recording every call."""

    def __init__(self) -> None:
        self.construction_kwargs: dict = {}
        self.calls: list[list] = []

    def __call__(self, *args: object, **kwargs: object) -> 'RecordingPlayerMetrics':
        """Stand in for the class itself, recording every construction argument."""
        self.construction_kwargs = dict(kwargs)
        return self

    def compute(self, court_positions: list) -> tuple:
        """Record the call and return one entry per supplied frame, plus a report."""
        self.calls.append(list(court_positions))
        n_frames = len(court_positions)
        # Derived from the input length rather than fixed: a stub whose output
        # length ignores its input cannot catch a frame-alignment bug, which
        # has been the gap in six PRs across the last three stages.
        #
        # The speeds are deliberately SPARSER than the distances: absent on
        # frame 0, as the real class leaves them whenever the gap rule
        # suppresses one, so passing the two lists in the wrong order, or
        # passing one twice, changes what the annotator receives.
        distances = [{7: float(index)} for index in range(n_frames)]
        speeds = [{} if index == 0 else {7: f'speed_{index}'} for index in range(n_frames)]
        report = SimpleNamespace(summary=lambda: '[metrics] stub metrics report')
        return distances, speeds, report


def stage_7_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, n_frames: int = 3) -> dict:
    """Replace every heavy stage so main.main() runs offline, returning the recording stubs."""
    frames = [f'frame_{index}' for index in range(n_frames)]
    monkeypatch.setattr(main, 'load_video', lambda path: list(enumerate(frames)))
    monkeypatch.setattr(main, 'get_video_metadata', lambda path: {'fps': 30.0})
    monkeypatch.setattr(main, 'write_video', lambda frames, path, fps: None)
    monkeypatch.setattr(Path, 'exists', lambda self: True)

    keypoint_detector = RecordingDetector(lambda index: [f'kp_{index}'] * 18)
    player_detector = RecordingDetector(lambda index: [f'track_{index}'])
    ball_detector = RecordingDetector(lambda index: f'ball_{index}')
    keypoint_annotator = RecordingAnnotator()
    court_mapper = RecordingCourtMapper()
    minimap_annotator = RecordingAnnotator()
    player_metrics = RecordingPlayerMetrics()
    metrics_annotator = RecordingAnnotator()

    monkeypatch.setattr(main, 'CourtKeypoints', keypoint_detector)
    monkeypatch.setattr(main, 'PlayerDetector', player_detector)
    monkeypatch.setattr(main, 'BallDetector', ball_detector)
    monkeypatch.setattr(main, 'KeypointAnnotator', keypoint_annotator)
    monkeypatch.setattr(main, 'CourtMapper', court_mapper)
    monkeypatch.setattr(main, 'MinimapAnnotator', minimap_annotator)
    monkeypatch.setattr(main, 'PlayerMetrics', player_metrics)
    monkeypatch.setattr(main, 'MetricsAnnotator', metrics_annotator)

    monkeypatch.setattr(
        main, 'possession_ball_input',
        lambda raw: type('Input', (), {'filled': list(raw), 'gated': list(raw)})(),
    )
    monkeypatch.setattr(main, 'FashionCLIPClassifier', StubClassifier)
    monkeypatch.setattr(main, 'PossessionTracker', StubPossession)
    monkeypatch.setattr(main, 'EventDetector', StubEvents)
    for name in ('PlayerAnnotator', 'BallAnnotator', 'PossessionAnnotator', 'EventAnnotator'):
        monkeypatch.setattr(main, name, RecordingAnnotator())

    monkeypatch.setattr(
        main, 'parse_args',
        lambda: type('Args', (), {
            'config': 'config/default.yaml',
            'input': 'data/raw/clip_1.mp4',
            'output': str(tmp_path),
        })(),
    )
    return {
        'frames': frames,
        'keypoint_detector': keypoint_detector,
        'keypoint_annotator': keypoint_annotator,
        'court_mapper': court_mapper,
        'minimap_annotator': minimap_annotator,
        'player_metrics': player_metrics,
        'metrics_annotator': metrics_annotator,
    }


def test_court_keypoints_is_constructed_with_the_keypoint_weights(monkeypatch, tmp_path):
    harness = stage_7_harness(monkeypatch, tmp_path)

    main.main()

    assert harness['keypoint_detector'].model_path == 'models/keypoints.pt'


def test_the_keypoint_weights_come_from_config_not_a_hardcoded_path(monkeypatch, tmp_path):
    # A hardcoded path is the hand-synced duplicate config/default.yaml's own
    # comments warn against: the video, detection and tracking blocks were
    # deleted for exactly that reason.
    harness = stage_7_harness(monkeypatch, tmp_path)
    real_load = main.load_config

    def override(path: str) -> dict:
        config = real_load(path)
        config['keypoints']['weights'] = 'models/elsewhere.pt'
        return config

    monkeypatch.setattr(main, 'load_config', override)
    main.main()

    assert harness['keypoint_detector'].model_path == 'models/elsewhere.pt'


def test_the_config_threshold_reaches_the_instance_threshold_only(monkeypatch, tmp_path):
    # keypoints.confidence_threshold maps to the INSTANCE threshold, which is
    # passed to inference and so determines the cached predictions. The
    # per-keypoint threshold is applied after retrieval by filter_keypoints(),
    # which main.py does not call, and is deliberately not fingerprinted.
    harness = stage_7_harness(monkeypatch, tmp_path)
    real_load = main.load_config

    def override(path: str) -> dict:
        config = real_load(path)
        config['keypoints']['confidence_threshold'] = 0.37
        return config

    monkeypatch.setattr(main, 'load_config', override)
    main.main()

    kwargs = harness['keypoint_detector'].construction_kwargs
    assert kwargs['instance_confidence_threshold'] == 0.37
    assert 'keypoint_confidence_threshold' not in kwargs


def test_the_default_config_defines_every_keypoint_key_main_reads():
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    assert 'keypoints' in config, 'config/default.yaml is missing the keypoints section'
    for key in ('weights', 'confidence_threshold'):
        assert key in config['keypoints'], f'config/default.yaml is missing keypoints.{key}'


def test_keypoint_detection_uses_the_conventional_cache_path(monkeypatch, tmp_path):
    harness = stage_7_harness(monkeypatch, tmp_path)

    main.main()

    call = harness['keypoint_detector'].calls[0]
    assert call['cache_path'] == 'data/processed/clip_1/keypoints.pkl'
    assert call['video_path'] == 'data/raw/clip_1.mp4'


def test_the_keypoint_annotator_receives_the_detection_output_one_entry_per_frame(monkeypatch, tmp_path):
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=5)

    main.main()

    frames, args = harness['keypoint_annotator'].calls[0]
    keypoints_per_frame = args[0]

    assert len(frames) == 5
    assert len(keypoints_per_frame) == len(frames)
    # The detector's own output, not a rebuilt, filtered or truncated copy.
    assert keypoints_per_frame == [[f'kp_{index}'] * 18 for index in range(5)]


def test_the_keypoint_detector_is_given_every_loaded_frame(monkeypatch, tmp_path):
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=7)

    main.main()

    assert harness['keypoint_detector'].calls[0]['frames'] == harness['frames']


def test_the_preflight_loop_does_not_gate_the_run_on_the_keypoint_checkpoint():
    # The loop runs before any frame is processed, so including the keypoint
    # weights would abort Stages 1-6 too: the whole pipeline stopping for the
    # one stage that needs the file. CourtKeypoints.load_model() raises a
    # clear error naming the path and the stage where it is actually required.
    source = Path(main.__file__).read_text(encoding='utf-8')
    loop = source[source.index('for weights in ('):]
    loop = loop[:loop.index(')')]

    assert 'player_weights' in loop
    assert 'ball_weights' in loop
    assert 'keypoint' not in loop


def test_court_keypoints_raises_its_own_clear_error_for_a_missing_checkpoint(tmp_path):
    # Pins that the error the pre-flight loop no longer raises really does
    # exist downstream, rather than trusting that it does.
    detector = CourtKeypoints(model_path=str(tmp_path / 'absent.pt'))

    with pytest.raises(FileNotFoundError, match=r'Court keypoint weights do not exist'):
        detector.load_model()


def test_stage_7_failing_does_not_prevent_the_earlier_stages_from_running(monkeypatch, tmp_path):
    # The behavioural consequence of removing it from the pre-flight loop:
    # player tracking and ball detection must both have run before the
    # keypoint stage raises.
    harness = stage_7_harness(monkeypatch, tmp_path)

    def raise_missing(**kwargs: object) -> list:
        raise FileNotFoundError('Court keypoint weights do not exist: models/keypoints.pt.')

    monkeypatch.setattr(harness['keypoint_detector'], 'run_detection', raise_missing)

    with pytest.raises(FileNotFoundError, match='Court keypoint weights do not exist'):
        main.main()

    assert main.PlayerDetector.calls, 'player tracking must have run before Stage 7 failed'
    assert main.BallDetector.calls, 'ball detection must have run before Stage 7 failed'


def test_keypoint_detection_runs_before_ball_interpolation():
    # Grouped with the other detectors, matching the pipeline's stage ordering,
    # rather than appended next to the annotator that reads it.
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert source.index('CourtKeypoints(') < source.index('possession_ball_input(')
    assert source.index('keypoints.pkl') < source.index('possession_ball_input(')


def test_main_never_calls_filter_keypoints():
    # The A3 filtering-policy ablation was closed without running, so
    # keypoint_confidence_threshold stands at its untuned 0.5 default and
    # applying it here would present an unmeasured policy as a chosen one.
    # Comments are stripped first: the construction site's own comment names
    # filter_keypoints() to explain why it is not called, and a substring
    # match would read that explanation as the call.
    source = Path(main.__file__).read_text(encoding='utf-8')
    code = '\n'.join(
        line for line in source.splitlines() if not line.strip().startswith('#')
    )

    assert 'filter_keypoints' not in code


def test_keypoints_are_drawn_beneath_the_possession_and_event_captions():
    # Keypoint dots carry index labels and would otherwise land on top of the
    # possession caption (top-left) or the event captions (top-right), which
    # are the two readings a viewer checks a frame for.
    source = Path(main.__file__).read_text(encoding='utf-8')

    keypoint_draw = source.index('keypoint_annotator.draw(')

    assert keypoint_draw < source.index('possession_annotator.draw(')
    assert keypoint_draw < source.index('event_annotator.draw(')
    # Drawn on the raw frames, so no earlier annotation layer is discarded.
    assert 'keypoint_annotator.draw(frames, keypoints_per_frame)' in source


def test_every_annotator_still_runs_and_the_chain_is_not_broken(monkeypatch, tmp_path):
    # Guards against the keypoint layer being inserted in a way that drops a
    # later annotator's output: each draw must receive the previous one's.
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=4)

    main.main()

    keypoint_output = harness['keypoint_annotator'].calls[0]
    assert len(keypoint_output[0]) == 4
    assert main.PlayerAnnotator.calls, 'the player annotator must still run'
    player_input = main.PlayerAnnotator.calls[0][0]
    # The player annotator receives the keypoint annotator's output, not the
    # raw frames, so the keypoint layer is not silently discarded.
    assert player_input == [f'frame_{index}+drawn' for index in range(4)]


# --- Stage 8 wiring -------------------------------------------------------

def test_court_mapper_is_constructed_and_called_with_tracks_and_keypoints(monkeypatch, tmp_path):
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=4)

    main.main()

    assert len(harness['court_mapper'].calls) == 1
    tracks, keypoints = harness['court_mapper'].calls[0]
    assert len(tracks) == 4
    assert len(keypoints) == 4
    # The detector's own output, not a rebuilt or re-filtered copy.
    assert keypoints == [[f'kp_{index}'] * 18 for index in range(4)]


def test_court_mapping_runs_after_keypoint_detection():
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert source.index('keypoints.pkl') < source.index('CourtMapper(')
    assert source.index('CourtMapper(') < source.index('minimap_annotator.draw(')


def test_the_keypoint_config_threshold_is_not_passed_into_court_mapper(monkeypatch, tmp_path):
    # config's keypoints.confidence_threshold is CourtKeypoints' INSTANCE
    # threshold, gating whether a court is detected at all. CourtMapper's
    # parameter is the per-keypoint threshold selecting which landmarks are
    # trusted. Both default to 0.5, so a conflation would be invisible.
    harness = stage_7_harness(monkeypatch, tmp_path)

    main.main()

    assert harness['court_mapper'].construction_kwargs == {}


def test_no_config_key_is_added_for_the_per_keypoint_threshold():
    # No value has been selected by measurement, so a config key would imply a
    # tuned parameter where none exists.
    source = Path(main.__file__).read_text(encoding='utf-8')
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    assert 'keypoint_confidence_threshold' not in source
    assert 'keypoint_confidence_threshold' not in config.get('keypoints', {})


def test_the_minimap_receives_the_reports_flags_not_a_positions_derived_fallback(monkeypatch, tmp_path):
    # The stub's report deliberately disagrees with its own positions: frame 0
    # is empty but flagged True, and the last frame is populated but flagged
    # False. Deriving the flags from bool(positions[i]) therefore produces a
    # different list, and this test fails.
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=4)

    main.main()

    _, args = harness['minimap_annotator'].calls[0]
    positions, flags = args[0], args[1]

    assert flags == harness['court_mapper'].report.frame_has_homography
    assert flags == [True, True, True, False]
    # And that is NOT what a positions-derived fallback would have produced.
    assert flags != [bool(entry) for entry in positions]


def test_the_minimap_receives_one_entry_per_frame(monkeypatch, tmp_path):
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=5)

    main.main()

    frames, args = harness['minimap_annotator'].calls[0]
    positions, flags = args[0], args[1]

    assert len(frames) == 5
    assert len(positions) == 5
    assert len(flags) == 5


def test_the_minimap_is_drawn_last_over_every_other_annotator():
    # It occupies the bottom-left, which no other annotator uses, so it
    # obscures neither the possession caption (top-left) nor the event
    # captions (top-right).
    source = Path(main.__file__).read_text(encoding='utf-8')
    minimap = source.index('minimap_annotator.draw(')

    for earlier in (
        'keypoint_annotator.draw(',
        'player_annotator.draw(',
        'ball_annotator.draw(',
        'possession_annotator.draw(',
        'event_annotator.draw(',
    ):
        assert source.index(earlier) < minimap, f'{earlier} must run before the minimap'


def test_the_minimap_receives_the_previous_annotators_output(monkeypatch, tmp_path):
    # The chain must not be broken: the minimap draws on what came before it,
    # not on the raw frames.
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=3)

    main.main()

    frames, _ = harness['minimap_annotator'].calls[0]

    assert frames != harness['frames'], 'the minimap must not receive the raw frames'


def test_the_minimap_gets_the_team_assignment(monkeypatch, tmp_path):
    # A player must be the same colour in the broadcast view and the tactical
    # view, so the team assignment reaches both annotators.
    harness = stage_7_harness(monkeypatch, tmp_path)

    main.main()

    _, args = harness['minimap_annotator'].calls[0]

    assert len(args) == 3, 'team_assignment must be passed, not left to default'


def test_the_minimap_is_constructed_with_the_per_clip_colours():
    source = Path(main.__file__).read_text(encoding='utf-8')
    construction = source[source.index('MinimapAnnotator('):]
    construction = construction[:construction.index(')')]

    assert 'team_1_colour=clip_settings' in construction
    assert 'team_2_colour=clip_settings' in construction


def test_the_mapping_report_is_printed(monkeypatch, tmp_path, capsys):
    # These counts are this stage's availability figure and the run is where
    # they are produced, so they must not be left to be inferred.
    stage_7_harness(monkeypatch, tmp_path)

    main.main()

    assert '[court] stub mapping report' in capsys.readouterr().out


def test_the_court_template_path_comes_from_config(monkeypatch, tmp_path):
    # The key is a genuine deployment setting (a different court render or
    # install location is changed here rather than in code), unlike
    # CourtMapper's per-keypoint threshold, which is absent from config
    # because no value has been selected by measurement.
    harness = stage_7_harness(monkeypatch, tmp_path)
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    main.main()

    assert harness['minimap_annotator'].construction_kwargs['template_path'] == (
        config['homography']['court_template_path']
    )


def test_a_non_default_template_path_is_honoured(monkeypatch, tmp_path):
    # Reading the config value is not enough on its own: it must actually
    # reach the annotator rather than being read and discarded, and the
    # committed value happens to equal MinimapAnnotator's own default, so a
    # test against the default alone could not tell the two apart.
    harness = stage_7_harness(monkeypatch, tmp_path)
    elsewhere = 'assets/some_other_court.png'
    assert elsewhere != MinimapAnnotator.__init__.__defaults__[0], 'fixture must differ from the default'
    real_load = main.load_config

    def override(path: str) -> dict:
        config = real_load(path)
        config['homography']['court_template_path'] = elsewhere
        return config

    monkeypatch.setattr(main, 'load_config', override)
    main.main()

    assert harness['minimap_annotator'].construction_kwargs['template_path'] == elsewhere


def test_the_committed_config_value_matches_the_annotator_default():
    # They agree today, which is exactly why the override test above exists:
    # a wiring bug that ignored config would still produce the right path on
    # the committed file and pass any test that only checked the default.
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    assert config['homography']['court_template_path'] == (
        MinimapAnnotator.__init__.__defaults__[0]
    )


def test_a_missing_homography_section_raises_rather_than_falling_back():
    # Direct subscription, matching how team_classifier and keypoints are read.
    # A silent fallback to the constructor default would leave the config
    # saying one thing and the run doing another.
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert 'court_template_path' in source
    assert 'config.get(' not in source.split('minimap_annotator = MinimapAnnotator(')[1]


def test_the_default_config_defines_the_homography_key_main_reads():
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    assert 'homography' in config, 'config/default.yaml is missing the homography section'
    assert 'court_template_path' in config['homography']


# --- Stage 9 wiring -------------------------------------------------------

def test_player_metrics_is_constructed_with_the_video_metadata_fps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = stage_7_harness(monkeypatch, tmp_path)

    main.main()

    assert harness['player_metrics'].construction_kwargs['fps'] == 30.0


def test_the_fps_comes_from_the_metadata_not_a_literal_or_a_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The harness' own metadata says 30.0, which is also this project's real
    # frame rate, so a hardcoded 30.0 would pass the test above. Changing what
    # the metadata reports is the only way to tell a read from a literal apart.
    # A 25.0 fps default on PlayerMetrics would, on 30 fps footage, report
    # every speed 20% low, a plausible-looking wrong number rather than an
    # error, which is why there is no default at all.
    harness = stage_7_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(main, 'get_video_metadata', lambda path: {'fps': 59.94})

    main.main()

    assert harness['player_metrics'].construction_kwargs['fps'] == 59.94


def test_player_metrics_has_no_fps_default_to_fall_back_on() -> None:
    # Pins the property the test above depends on: were a default reinstated,
    # a wiring bug that dropped the argument would silently produce speeds
    # scaled by the ratio between the default and the real frame rate.
    with pytest.raises(TypeError):
        PlayerMetrics()


def test_the_speed_window_comes_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = stage_7_harness(monkeypatch, tmp_path)
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    main.main()

    assert harness['player_metrics'].construction_kwargs['speed_window'] == (
        config['metrics']['speed_window']
    )


def test_a_non_default_speed_window_is_honoured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The committed value happens to equal PlayerMetrics' own default, so a
    # wiring bug that ignored config entirely would still produce the right
    # number on the committed file and pass the test above.
    harness = stage_7_harness(monkeypatch, tmp_path)
    elsewhere = 11
    assert elsewhere != DEFAULT_SPEED_WINDOW, 'fixture must differ from the default'
    real_load = main.load_config

    def override(path: str) -> dict:
        config = real_load(path)
        config['metrics']['speed_window'] = elsewhere
        return config

    monkeypatch.setattr(main, 'load_config', override)
    main.main()

    assert harness['player_metrics'].construction_kwargs['speed_window'] == elsewhere


def test_the_court_dimensions_are_not_passed_into_player_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # PlayerMetrics performs no unit conversion (CourtMapper already returns
    # metres), so there is nothing for a court dimension to calibrate. The two
    # keys remain in config, pinned by tests/test_court_template.py, but
    # production must not read them.
    harness = stage_7_harness(monkeypatch, tmp_path)

    main.main()

    kwargs = harness['player_metrics'].construction_kwargs
    assert set(kwargs) == {'fps', 'speed_window'}


def test_main_never_reads_the_court_dimension_keys() -> None:
    source = Path(main.__file__).read_text(encoding='utf-8')
    # Comments are stripped first: the construction site's own comment names
    # both keys to explain why they are NOT read, and a substring match would
    # read that explanation as the read.
    code = '\n'.join(
        line for line in source.splitlines() if not line.strip().startswith('#')
    )

    assert 'court_length_m' not in code
    assert 'court_width_m' not in code


def test_the_default_config_defines_the_metrics_key_main_reads() -> None:
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    assert 'metrics' in config, 'config/default.yaml is missing the metrics section'
    assert 'speed_window' in config['metrics']


def test_metrics_are_computed_from_the_mapped_court_positions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The mapper's own output, not the raw tracks: a metric speed is only
    # meaningful in court coordinates, and the tracks are still in pixels.
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=4)

    main.main()

    assert len(harness['player_metrics'].calls) == 1
    supplied = harness['player_metrics'].calls[0]
    # Exactly what RecordingCourtMapper returned, including its empty frame 0,
    # not the player tracks, which are a different shape entirely.
    assert supplied == [{} if index == 0 else {7: (5.0, 5.0)} for index in range(4)]


def test_metrics_run_after_court_mapping_and_before_annotation() -> None:
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert source.index('CourtMapper(') < source.index('PlayerMetrics(')
    assert source.index('PlayerMetrics(') < source.index('[main] Annotating frames')


def test_the_metrics_annotator_receives_one_entry_per_frame(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=5)

    main.main()

    frames, args = harness['metrics_annotator'].calls[0]
    tracks, distances, speeds = args[0], args[1], args[2]

    assert len(frames) == 5
    assert len(tracks) == 5
    assert len(distances) == 5
    assert len(speeds) == 5


def test_the_metrics_annotator_receives_the_two_lists_in_the_right_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The stub's speeds are deliberately sparser than its distances: absent on
    # frame 0, as the real class leaves them wherever the gap rule suppresses
    # one, so swapping the two arguments, or passing one twice, fails here.
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=4)

    main.main()

    _, args = harness['metrics_annotator'].calls[0]
    distances, speeds = args[1], args[2]

    assert distances == [{7: float(index)} for index in range(4)]
    assert speeds == [{}, {7: 'speed_1'}, {7: 'speed_2'}, {7: 'speed_3'}]


def test_the_metrics_annotator_gets_the_player_tracks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The text is drawn beneath each player's own box, so it needs the boxes.
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=3)

    main.main()

    _, args = harness['metrics_annotator'].calls[0]

    assert args[0] == [[f'track_{index}'] for index in range(3)]


def test_the_metrics_annotator_is_drawn_after_the_player_annotator() -> None:
    # Its text sits over the player ellipse rather than under it.
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert source.index('player_annotator.draw(') < source.index('metrics_annotator.draw(')


def test_the_metrics_annotator_is_drawn_before_the_minimap() -> None:
    # The minimap stays last, over everything: it is a panel rather than marks
    # on the broadcast image, and being on top keeps it legible.
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert source.index('metrics_annotator.draw(') < source.index('minimap_annotator.draw(')


def test_the_metrics_annotator_receives_the_previous_annotators_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The chain must not be broken: it draws on what came before it, not on
    # the raw frames.
    harness = stage_7_harness(monkeypatch, tmp_path, n_frames=3)

    main.main()

    frames, _ = harness['metrics_annotator'].calls[0]

    assert frames != harness['frames'], 'the metrics annotator must not receive the raw frames'
    assert frames == [f'frame_{index}+drawn+drawn+drawn+drawn+drawn' for index in range(3)]


def test_the_metrics_report_is_printed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # The counts of displacements measured, speeds computed and speeds
    # suppressed by the gap rule are this stage's availability figure, and this
    # run is where they are produced.
    stage_7_harness(monkeypatch, tmp_path)

    main.main()

    assert '[metrics] stub metrics report' in capsys.readouterr().out


def test_the_metrics_annotator_is_given_the_minimaps_occupied_region(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The minimap composites AFTER the metrics text and overwrites the
    # bottom-left, so attaching the text to a player is not sufficient on its
    # own: a player near that sideline had their speed drawn and then covered.
    # The rectangle must reach the annotator for it to flip clear.
    harness = stage_7_harness(monkeypatch, tmp_path)

    main.main()

    kwargs = harness['metrics_annotator'].draw_kwargs[0]

    assert 'reserved_region' in kwargs, 'the minimap region must be passed, not left to default'
    # The minimap stub's own distinctive rectangle, so a caller computing its
    # own guess at the geometry fails rather than coincidentally agreeing.
    assert kwargs['reserved_region'] == (12, 596, 221, 708)
    assert harness['minimap_annotator'].region_requests, 'the region must be read from the minimap'


def test_the_reserved_region_comes_from_the_minimap_annotator_itself() -> None:
    # Read from the annotator that owns the geometry rather than recomputed, so
    # the reserved rectangle and the composited one cannot drift apart.
    source = Path(main.__file__).read_text(encoding='utf-8')
    code = '\n'.join(
        line for line in source.splitlines() if not line.strip().startswith('#')
    )

    assert 'minimap_annotator.occupied_region_for(' in code


def test_the_minimap_still_draws_after_the_metrics_text() -> None:
    # The fix is to pass the region, NOT to reorder: the minimap is Stage 8's
    # verification mechanism, and per-player text over the tactical panel would
    # obscure the thing the panel exists to prove.
    source = Path(main.__file__).read_text(encoding='utf-8')

    assert source.index('metrics_annotator.draw(') < source.index('minimap_annotator.draw(')
