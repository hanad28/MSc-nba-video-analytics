"""
measure_mapping_and_metrics.py regenerates Stage 8's mapping counts and Stage 9's
distance and speed counts from the cached detections and keypoints, so the figures
quoted in the dissertation are traceable to evidence rather than to a main.py run
whose stdout has scrolled away. The CSVs are written to data/outputs/, which is
gitignored, and are copied into results/ for the submitted repository. Runs on
JupyterHub; it needs the caches and the raw clips, but no model weights and no GPU.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

# Running this file directly (`python scripts/measure_mapping_and_metrics.py`) puts
# scripts/ on sys.path[0], not the repo root, so `basketball` would not be
# importable. Insert the repo root explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basketball.cache.cache_utils import load_cache
from basketball.homography.court_mapper import CourtMapper, MappingReport
from basketball.metrics.player_metrics import MetricsReport, PlayerMetrics
from basketball.utils.io_utils import get_video_metadata

CLIPS = ('clip_1', 'clip_2', 'clip_3')
CONFIG_PATH = 'config/default.yaml'
CACHE_TEMPLATE = 'data/processed/{clip}'
RAW_VIDEO_TEMPLATE = 'data/raw/{clip}.mp4'

OUTPUT_DIR = Path('data/outputs')
MAPPING_CSV = OUTPUT_DIR / 'stage8_mapping_counts.csv'
METRICS_CSV = OUTPUT_DIR / 'stage9_metrics_counts.csv'

TOTAL_LABEL = 'TOTAL'

MAPPING_HEADERS = [
    'clip', 'n_frames', 'mapped_frames', 'unmapped_frames',
    'insufficient_keypoints', 'degenerate_keypoints', 'malformed_input',
    'positions_mapped', 'positions_dropped_out_of_bounds',
    'positions_dropped_at_horizon', 'reconciles',
]
METRICS_HEADERS = [
    'clip', 'n_frames', 'tracks', 'displacements_measured',
    'speeds_computed', 'speeds_suppressed_by_gap', 'speed_rate',
]

# Pre-registered before running, per this project's practice. A mismatch is a
# finding to investigate, not a number to quietly overwrite: main() exits
# non-zero on any disagreement so a drifted figure cannot ship unnoticed.
EXPECTED_METRICS = {
    'clip_1': {'tracks': 13, 'displacements_measured': 651, 'speeds_computed': 643, 'speeds_suppressed_by_gap': 8},
    'clip_2': {'tracks': 14, 'displacements_measured': 1609, 'speeds_computed': 1606, 'speeds_suppressed_by_gap': 3},
    'clip_3': {'tracks': 17, 'displacements_measured': 1581, 'speeds_computed': 1572, 'speeds_suppressed_by_gap': 9},
}
# Deliberately no 'tracks' entry: the pre-registration left the total unstated,
# and summing per-clip track counts would not be meaningful anyway, since a
# track id identifies a player within one clip rather than across clips.
EXPECTED_METRICS_TOTALS = {
    'displacements_measured': 3841,
    'speeds_computed': 3821,
    'speeds_suppressed_by_gap': 20,
}
# Stage 8 maps exactly the frames Stage 7 found sufficient, so these are Stage
# 7's own per-clip sufficiency counts, restated as a Stage 8 prediction.
EXPECTED_MAPPED_FRAMES = {'clip_1': 107, 'clip_2': 174, 'clip_3': 198}


def load_config(path: str) -> dict:
    """Load and return the YAML pipeline configuration from disk."""
    with open(path, 'r') as handle:
        return yaml.safe_load(handle)


def require_cache(path: str, clip: str) -> str:
    """Return path, raising a clear error when the cache it names has not been produced yet."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f'{path} not found. Caches are produced by running main.py on the corresponding '
            f'clip and normally live under data/processed/{clip}/.'
        )
    return path


def measure_clip(clip: str, speed_window: int) -> tuple[MappingReport, MetricsReport]:
    """Run Stages 8 and 9 over one clip's cached tracks and keypoints, returning both reports."""
    cache_dir = CACHE_TEMPLATE.format(clip=clip)
    player_tracks = load_cache(require_cache(f'{cache_dir}/player_detections.pkl', clip))
    keypoints_per_frame = load_cache(require_cache(f'{cache_dir}/keypoints.pkl', clip))
    metadata = get_video_metadata(RAW_VIDEO_TEMPLATE.format(clip=clip))

    # Constructed exactly as main.py does. CourtMapper takes no arguments:
    # its per-keypoint threshold is deliberately not wired to config, so
    # passing anything here would measure a configuration production never runs.
    court_mapper = CourtMapper()
    court_positions, mapping_report = court_mapper.map_to_court(player_tracks, keypoints_per_frame)

    # fps comes from the video metadata, never a literal: PlayerMetrics takes it
    # with no default precisely because a wrong frame rate scales every speed by
    # the ratio between them and produces plausible numbers rather than an error.
    player_metrics = PlayerMetrics(fps=metadata['fps'], speed_window=speed_window)
    _, _, metrics_report = player_metrics.compute(court_positions)

    return mapping_report, metrics_report


def require_reconciliation(clip: str, report: MappingReport) -> None:
    """Raise if a clip's mapped and unmapped frame counts do not account for every frame."""
    if not report.reconciles():
        raise ValueError(
            f'{clip}: mapping report does not reconcile — {report.mapped_frames} mapped + '
            f'{report.unmapped_frames} unmapped != {report.n_frames} frames.'
        )


def mapping_row(clip: str, report: MappingReport) -> list[object]:
    """Return one clip's Stage 8 counts as a CSV row."""
    return [
        clip, report.n_frames, report.mapped_frames, report.unmapped_frames,
        report.insufficient_keypoints, report.degenerate_keypoints, report.malformed_input,
        report.positions_mapped, report.positions_dropped_out_of_bounds,
        report.positions_dropped_at_horizon, report.reconciles(),
    ]


def metrics_row(clip: str, report: MetricsReport) -> list[object]:
    """Return one clip's Stage 9 counts as a CSV row."""
    attempted = report.speeds_computed + report.speeds_suppressed_by_gap
    rate = report.speeds_computed / attempted if attempted else float('nan')
    return [
        clip, report.n_frames, len(report.total_distance_m), report.displacements_measured,
        report.speeds_computed, report.speeds_suppressed_by_gap, round(rate, 4),
    ]


def mapping_total_row(rows: list[list[object]]) -> list[object]:
    """Return the TOTAL row for the Stage 8 table, summing every count across clips."""
    totals = [sum(int(row[index]) for row in rows) for index in range(1, len(MAPPING_HEADERS) - 1)]
    # Recomputed from the summed counts rather than AND-ed across the per-clip
    # flags. The two agree, but deriving it the same way a clip row does keeps
    # one definition of what reconciling means.
    n_frames, mapped, unmapped = totals[0], totals[1], totals[2]
    return [TOTAL_LABEL] + totals + [mapped + unmapped == n_frames]


def metrics_total_row(rows: list[list[object]]) -> list[object]:
    """Return the TOTAL row for the Stage 9 table, leaving tracks empty and recomputing the speed rate from the totals."""
    # Summed by header name rather than by position, so the tracks column can be
    # left out of the summation without the remaining indices shifting.
    totals = {
        name: sum(int(row[METRICS_HEADERS.index(name)]) for row in rows)
        for name in ('n_frames', 'displacements_measured', 'speeds_computed', 'speeds_suppressed_by_gap')
    }
    attempted = totals['speeds_computed'] + totals['speeds_suppressed_by_gap']
    # Recomputed from the summed counts, never averaged across the per-clip
    # rates: the clips carry very different displacement counts, so a mean of
    # rates would weight a 651-displacement clip equally with a 1609 one.
    rate = totals['speeds_computed'] / attempted if attempted else float('nan')
    # tracks is deliberately EMPTY rather than summed. A track id identifies a
    # player within one clip and carries no meaning across clips, so a sum
    # counts the same person once per clip they appear in and reads as a squad
    # size no clip actually has. The pre-registration leaves this cell as a dash
    # for the same reason, which is also why EXPECTED_METRICS_TOTALS carries no
    # tracks entry: there is no correct number to register.
    return [
        TOTAL_LABEL, totals['n_frames'], '', totals['displacements_measured'],
        totals['speeds_computed'], totals['speeds_suppressed_by_gap'], round(rate, 4),
    ]


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    """Write a header row and every supplied row to a CSV, creating the output directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def compare(scope: str, quantity: str, expected: int, actual: int) -> tuple[str, str, int, int, bool]:
    """Return one pre-registered expectation paired with the measured value and whether they agree."""
    return (scope, quantity, expected, actual, expected == actual)


def collect_comparisons(
    mapping_reports: dict[str, MappingReport],
    metrics_reports: dict[str, MetricsReport],
) -> list[tuple[str, str, int, int, bool]]:
    """Return every pre-registered expectation compared against what this run measured."""
    comparisons: list[tuple[str, str, int, int, bool]] = []
    for clip in sorted(mapping_reports):
        comparisons.append(compare(
            clip, 'mapped_frames', EXPECTED_MAPPED_FRAMES[clip], mapping_reports[clip].mapped_frames,
        ))
    for clip in sorted(metrics_reports):
        report = metrics_reports[clip]
        measured = {
            'tracks': len(report.total_distance_m),
            'displacements_measured': report.displacements_measured,
            'speeds_computed': report.speeds_computed,
            'speeds_suppressed_by_gap': report.speeds_suppressed_by_gap,
        }
        for quantity, expected in EXPECTED_METRICS[clip].items():
            comparisons.append(compare(clip, quantity, expected, measured[quantity]))
    for quantity, expected in EXPECTED_METRICS_TOTALS.items():
        actual = sum(getattr(report, quantity) for report in metrics_reports.values())
        comparisons.append(compare(TOTAL_LABEL, quantity, expected, actual))
    return comparisons


def print_comparisons(comparisons: list[tuple[str, str, int, int, bool]]) -> bool:
    """Print each pre-registered expectation as PASS or MISMATCH, returning whether all agreed."""
    print()
    print('Pre-registered expectations')
    # Built by concatenation rather than an f-string: the column labels are
    # literals, and nesting single quotes inside a single-quoted f-string is a
    # syntax error before Python 3.12 (this project runs 3.11).
    print('scope'.ljust(8) + ' ' + 'quantity'.ljust(28) + ' '
          + 'expected'.rjust(9) + ' ' + 'measured'.rjust(9) + '  verdict')
    for scope, quantity, expected, actual, ok in comparisons:
        verdict = 'PASS' if ok else 'MISMATCH'
        print(f'{scope:8s} {quantity:28s} {expected:>9d} {actual:>9d}  {verdict}')

    failures = [row for row in comparisons if not row[4]]
    if failures:
        print()
        print(f'{len(failures)} of {len(comparisons)} expectations did not match.')
        # Stated rather than left implicit: a drifted figure means an input or a
        # stage behaviour changed since the numbers were registered, and that
        # needs establishing before anything quoting them ships.
        print('A mismatch is a finding. Investigate what changed before citing these figures.')
    return not failures


def main() -> int:
    """Measure every clip, write both CSVs, and return a non-zero exit code on any mismatch."""
    config = load_config(CONFIG_PATH)
    speed_window = config['metrics']['speed_window']

    mapping_reports: dict[str, MappingReport] = {}
    metrics_reports: dict[str, MetricsReport] = {}
    mapping_rows: list[list[object]] = []
    metrics_rows: list[list[object]] = []

    for clip in CLIPS:
        print(f'[{clip}] measuring...')
        mapping_report, metrics_report = measure_clip(clip, speed_window)
        require_reconciliation(clip, mapping_report)

        print(mapping_report.summary())
        print(metrics_report.summary())

        mapping_reports[clip] = mapping_report
        metrics_reports[clip] = metrics_report
        mapping_rows.append(mapping_row(clip, mapping_report))
        metrics_rows.append(metrics_row(clip, metrics_report))

    mapping_rows.append(mapping_total_row(mapping_rows))
    metrics_rows.append(metrics_total_row(metrics_rows))

    write_csv(MAPPING_CSV, MAPPING_HEADERS, mapping_rows)
    write_csv(METRICS_CSV, METRICS_HEADERS, metrics_rows)
    print()
    print(f'Wrote {MAPPING_CSV}')
    print(f'Wrote {METRICS_CSV}')

    # The CSVs are written before the comparison is scored, deliberately: a
    # mismatched run is exactly the one whose numbers someone needs to look at.
    return 0 if print_comparisons(collect_comparisons(mapping_reports, metrics_reports)) else 1


if __name__ == '__main__':
    sys.exit(main())
