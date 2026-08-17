"""
run_evaluation.py

Runs the CLEAR MOT evaluation across five
tracker configurations on all three clips, against the hand-labelled ground
truth in data/annotations/{clip}_gt.txt. Runs on JupyterHub; it needs the
model weights, the clips and the GPU stack.

Each configuration caches its tracks under
data/processed/{clip}/mot_eval/{label}_tracks.pkl, so the production caches
at data/processed/{clip}/player_detections.pkl are never touched. Cache
invalidation is the cache fingerprint mechanism: the swept settings
(conf_threshold, minimum_consecutive_frames) are fingerprint keys, so
configurations can never serve each other's tracks.

A clip whose ground truth extends beyond the tracker output (mismatched
inputs) is reported and skipped; the sweep continues with the remaining
clips. Pooled rows accumulate association events across clips in a single
accumulator (MOTEvaluator.evaluate_pooled); averaging MOTA across clips of
different sizes is wrong. Every ID-switch figure is a lower bound: the
ground truth is sampled every 10th frame, so switches that occur and
resolve between labelled frames are invisible to it.

Outputs: per-clip and pooled tables printed and written to
data/outputs/mot_evaluation/results.csv, plus
data/outputs/mot_evaluation/switches.csv listing the frames where ID
switches are scored, for visual inspection.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# Running this file directly (`python scripts/run_evaluation.py`) puts scripts/
# on sys.path[0], not the repo root, so `basketball` and `evaluation` would not
# be importable. Insert the repo root explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basketball.detection.player_detector import PlayerDetector, PlayerTrack
from basketball.utils.io_utils import get_video_metadata, load_video
from evaluation.ground_truth import GTAnnotation, load_mot_annotations
from evaluation.mot_metrics import MOTEvaluator, MOTResult

CLIPS = ('clip_1', 'clip_2', 'clip_3')

CLIP_PATH_TEMPLATE = 'data/raw/{clip}.mp4'
GT_PATH_TEMPLATE = 'data/annotations/{clip}_gt.txt'
TRACKS_CACHE_TEMPLATE = 'data/processed/{clip}/mot_eval/{label}_tracks.pkl'

OUTPUT_DIR = Path('data/outputs/mot_evaluation')
RESULTS_CSV = OUTPUT_DIR / 'results.csv'
SWITCHES_CSV = OUTPUT_DIR / 'switches.csv'

# Sampled ground truth (every 10th frame) cannot see a switch that occurs and
# resolves between labelled frames, so every switch figure is a lower bound.
SWITCHES_HEADER = 'id_switches (lower bound — sampled GT)'
RESULT_HEADERS = ['mota', 'motp', 'idf1', SWITCHES_HEADER, 'false_positives', 'misses', 'precision', 'recall']


@dataclass
class TrackerConfig:
    label: str
    model_path: str
    conf_threshold: float
    minimum_consecutive_frames: int


CONFIGURATIONS = [
    TrackerConfig('production', 'models/ball.pt', 0.5, 2),
    TrackerConfig('players_pt', 'models/players.pt', 0.5, 2),
    TrackerConfig('lowscore_025', 'models/ball.pt', 0.25, 2),
    TrackerConfig('lowscore_010', 'models/ball.pt', 0.10, 2),
    TrackerConfig('mcf1', 'models/ball.pt', 0.5, 1),
]


def confirm_model_paths() -> None:
    """Raises FileNotFoundError before any tracking when a configuration's checkpoint is missing."""
    for path in sorted({config.model_path for config in CONFIGURATIONS}):
        if not Path(path).exists():
            raise FileNotFoundError(
                f'{path} not found. Model weights are distributed via the GitHub Release '
                f'for this repository; download them into models/ before running the sweep.'
            )


def load_scoreable_ground_truth(clip_name: str) -> list[GTAnnotation] | None:
    """Load a clip's ground truth, or None (with a message) when there is nothing to score; never raises for an unlabelled clip."""
    gt_path = GT_PATH_TEMPLATE.format(clip=clip_name)
    if not Path(gt_path).exists():
        # The labelling tool never writes empty gt files; a never-labelled clip
        # is simply absent. Letting load_mot_annotations raise here would kill
        # the whole sweep and discard every per-clip row already computed.
        print(f'[evaluate] {clip_name}: no ground-truth file at {gt_path} — clip skipped, not scored.')
        return None

    ground_truth = load_mot_annotations(gt_path)
    if not ground_truth:
        # An all-zero result for an unlabelled clip is indistinguishable from a
        # real measurement, the same hazard the labelling tool guards against
        # by never writing empty gt files in the first place.
        print(f'[evaluate] {clip_name}: ground truth holds no annotations — clip skipped, not scored.')
        return None
    return ground_truth


def pooled_row(
    evaluator: MOTEvaluator,
    config_label: str,
    sequences: list[tuple[list[GTAnnotation], list[dict[int, PlayerTrack]]]],
) -> list[str] | None:
    """The pooled table row for one configuration, or None (with a message) when no clip could be scored."""
    if not sequences:
        # evaluate_pooled([]) deliberately returns the zero result, but written
        # into the results table it would be indistinguishable from a real
        # measurement of a terrible configuration: omit the row and say so.
        print(f'[evaluate] {config_label}: no clip could be scored — pooled row omitted.')
        return None
    return [config_label, 'all'] + result_cells(evaluator.evaluate_pooled(sequences))


def track_clip(detector: PlayerDetector, config_label: str, clip_name: str) -> list[dict[int, PlayerTrack]] | None:
    """Run tracking on one clip; only an unreadable clip is skipped; storage faults must surface as themselves."""
    clip_path = CLIP_PATH_TEMPLATE.format(clip=clip_name)
    try:
        # Only the clip-reading step is guarded. The cache layer inside
        # run_tracking() raises IOError too (write failures, corrupt pickles
        # deliberately surfaced, unreadable fingerprint sidecars), and those
        # must not be reported as a missing video, nor bin finished tracking.
        frames = [frame for _, frame in load_video(clip_path)]
        get_video_metadata(clip_path)
    except IOError as error:
        # A missing or unreadable clip is this clip's problem, not the run's:
        # every already-computed row must survive it.
        print(f'[evaluate] {config_label} / {clip_name}: clip could not be read — skipped: {error}')
        return None

    return detector.run_tracking(
        frames=frames,
        video_path=clip_path,
        cache_path=TRACKS_CACHE_TEMPLATE.format(clip=clip_name, label=config_label),
    )


def evaluate_clip(
    evaluator: MOTEvaluator,
    config_label: str,
    clip_name: str,
    ground_truth: list[GTAnnotation],
    tracks: list[dict[int, PlayerTrack]],
) -> MOTResult | None:
    """Evaluate one clip, reporting and skipping mismatched inputs so one bad clip cannot abort the whole sweep."""
    try:
        return evaluator.evaluate(ground_truth, tracks)
    except ValueError as error:
        # MOTEvaluator deliberately leaves this decision to the caller: a sweep
        # must keep going and report exactly what it could not score.
        print(f'[evaluate] {config_label} / {clip_name}: evaluation skipped — {error}')
        return None


def result_cells(result: MOTResult) -> list[str]:
    """One table row of formatted metric values, in RESULT_HEADERS order."""
    return [
        f'{result.mota:.4f}',
        f'{result.motp:.4f}',
        f'{result.idf1:.4f}',
        str(result.num_switches),
        str(result.num_false_positives),
        str(result.num_misses),
        f'{result.precision:.4f}',
        f'{result.recall:.4f}',
    ]


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print one aligned results table."""
    print(f'\n{title}')
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)] if rows else [len(h) for h in headers]
    print('  '.join(header.ljust(width) for header, width in zip(headers, widths)))
    for row in rows:
        print('  '.join(str(cell).ljust(width) for cell, width in zip(row, widths)))


def write_outputs(
    per_clip_rows: list[list[str]],
    pooled_rows: list[list[str]],
    switch_rows: list[list[str]],
) -> None:
    """Print both tables and persist results.csv + switches.csv; called in a finally so partial results survive."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_headers = ['config', 'clip'] + RESULT_HEADERS

    print_table('Per-clip results', table_headers, per_clip_rows)
    print_table('Pooled across clips (single event-level accumulation, never a per-clip average)', table_headers, pooled_rows)

    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scope'] + table_headers)
        writer.writerows(['per_clip'] + row for row in per_clip_rows)
        writer.writerows(['pooled'] + row for row in pooled_rows)

    with open(SWITCHES_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'clip', 'switch_frame'])
        writer.writerows(switch_rows)

    print(f'\nWritten: {RESULTS_CSV}')
    print(f'Written: {SWITCHES_CSV} ({len(switch_rows)} switch events — lower bound, sampled GT)')


def main() -> None:
    confirm_model_paths()
    evaluator = MOTEvaluator()

    per_clip_rows: list[list[str]] = []
    pooled_rows: list[list[str]] = []
    switch_rows: list[list[str]] = []

    try:
        for config in CONFIGURATIONS:
            print(
                f'\n=== {config.label}: {config.model_path}, conf {config.conf_threshold}, '
                f'minimum_consecutive_frames {config.minimum_consecutive_frames} ==='
            )
            detector = PlayerDetector(
                model_path=config.model_path,
                conf_threshold=config.conf_threshold,
                minimum_consecutive_frames=config.minimum_consecutive_frames,
            )

            sequences: list[tuple[list[GTAnnotation], list[dict[int, PlayerTrack]]]] = []
            for clip in CLIPS:
                ground_truth = load_scoreable_ground_truth(clip)
                if ground_truth is None:
                    continue

                tracks = track_clip(detector, config.label, clip)
                if tracks is None:
                    continue

                result = evaluate_clip(evaluator, config.label, clip, ground_truth, tracks)
                if result is None:
                    continue

                per_clip_rows.append([config.label, clip] + result_cells(result))
                sequences.append((ground_truth, tracks))
                switch_rows.extend(
                    [config.label, clip, str(frame)]
                    for frame in evaluator.switch_frames(ground_truth, tracks)
                )

            row = pooled_row(evaluator, config.label, sequences)
            if row is not None:
                pooled_rows.append(row)

            # One YOLO model in memory at a time; the next configuration loads its own.
            del detector
    finally:
        # Hours of tracking must not vanish with a traceback: whatever was
        # computed before any unexpected exception is printed and persisted.
        write_outputs(per_clip_rows, pooled_rows, switch_rows)


if __name__ == '__main__':
    main()
