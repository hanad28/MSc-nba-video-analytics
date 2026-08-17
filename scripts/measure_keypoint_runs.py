"""
measure_keypoint_runs.py scores the court keypoint spec's prediction K1 by
validating both training runs against the grouped test split, so the repository
ships the outcome of its own pre-registration rather than a prediction a reader
cannot check. Ultralytics' val mode writes no results.csv, so without this the
four figures behind K1's refutation would exist only as prose. The CSV is
written to data/outputs/, which is gitignored, and is copied into results/ for
the submitted repository.
Runs on JupyterHub; it needs both checkpoints, the grouped split and a GPU. It
writes into its own project directory and never into runs/pose_test/, so the
earlier validation artefacts stay exactly as they were produced.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO

# Running this file directly (`python scripts/measure_keypoint_runs.py`) puts
# scripts/ on sys.path[0], not the repo root, so `basketball` would not be
# importable. Insert the repo root explicitly, matching the other measurement
# scripts, even though this one currently imports nothing from the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATA_YAML = Path('training/court-keypoints-grouped/data.yaml')
TEST_IMAGE_DIR = Path('training/court-keypoints-grouped/test/images')
TEST_LABEL_DIR = Path('training/court-keypoints-grouped/test/labels')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}

# A distinct project directory, never runs/pose_test/: that holds the earlier
# validation plots and sample images this measurement exists to supplement, and
# Ultralytics overwrites or increments a run directory rather than refusing to
# reuse it.
OUTPUT_PROJECT = 'runs/pose_k1_scoring'

IMAGE_SIZE = 640
BATCH_SIZE = 16

OUTPUT_CSV = Path('data/outputs/stage7_run_comparison.csv')
HEADERS = [
    'run', 'schedule', 'epochs', 'patience',
    'best_pose_map50_95_epoch', 'checkpoint_epoch',
    'test_pose_map50', 'test_pose_map50_95', 'test_box_map50_95',
    'n_images',
]

# Twice the observed drift. The 18 June re-validation diagnostic measured GPU
# float non-determinism on this exact operation at 0.0001 to 0.0006, so exact
# equality would fail for reasons that are not findings.
TOLERANCE = 0.001

# The column the pose mAP50-95 argmax is taken over, matching the comparison
# cell in training/train_court_keypoints.ipynb that produced the pre-registered
# 98 and 439. Reproducing Ultralytics' own fitness here instead would answer a
# different question; see the note at the resolution site in main().
POSE_MAP50_95_COLUMN = 'metrics/mAP50-95(P)'

# Metric columns compared when locating the epoch best.pt was saved at. The
# checkpoint stores its train_metrics as full floats while results.csv rounds
# them, so the match is to a tolerance rather than exact.
CHECKPOINT_MATCH_COLUMNS = (
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
    'metrics/mAP50(P)', 'metrics/mAP50-95(P)',
)
CHECKPOINT_MATCH_TOLERANCE = 1e-4


@dataclass
class RunSpec:
    """One training run's identity, schedule and pre-registered test figures."""

    run: str
    directory: Path
    schedule: str
    epochs: int
    patience: int
    best_pose_map50_95_epoch: int
    checkpoint_epoch: int
    pose_map50: float
    pose_map50_95: float
    box_map50_95: float


@dataclass
class RunResult:
    """One training run's measured test-split figures."""

    run: str
    best_pose_map50_95_epoch: int
    checkpoint_epoch: int
    pose_map50: float
    pose_map50_95: float
    box_map50_95: float
    n_images: int


RUN_A = RunSpec(
    run='A',
    directory=Path('runs/pose/court_kp_e100'),
    schedule='epochs=100, patience=30',
    epochs=100,
    patience=30,
    best_pose_map50_95_epoch=98,
    checkpoint_epoch=100,
    pose_map50=0.9849,
    pose_map50_95=0.9352,
    box_map50_95=0.9252,
)
RUN_B = RunSpec(
    run='B',
    directory=Path('runs/pose/court_kp_e500'),
    schedule='epochs=500, patience=100',
    epochs=500,
    patience=100,
    best_pose_map50_95_epoch=439,
    checkpoint_epoch=494,
    pose_map50=0.9950,
    pose_map50_95=0.9669,
    box_map50_95=0.9353,
)
RUNS = (RUN_A, RUN_B)


def require_path(path: Path, what: str, hint: str) -> Path:
    """Return the path, or raise a message naming what is missing and how to produce it."""
    # Both the checkpoints and the dataset are gitignored, so neither ships with
    # the repository. An examiner running this without them should learn what
    # the measurement needs, rather than reading an Ultralytics stack trace.
    if not path.exists():
        raise FileNotFoundError(f'{what} not found at {path}. {hint}')
    return path


def weights_path(spec: RunSpec) -> Path:
    """Return one run's best-checkpoint path, raising a clear error if the run directory was never produced."""
    return require_path(
        spec.directory / 'weights' / 'best.pt',
        f'Run {spec.run} checkpoint ({spec.schedule})',
        'It is written by training/train_court_keypoints.ipynb and is gitignored, '
        'so it must be present on the machine running this script.',
    )


def results_csv_path(spec: RunSpec) -> Path:
    """Return one run's training results.csv path, raising a clear error if it is absent."""
    return require_path(
        spec.directory / 'results.csv',
        f'Run {spec.run} training results.csv',
        'It is written per epoch by the training run and is gitignored.',
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a results.csv into dictionaries with whitespace-stripped column names."""
    # Ultralytics has shipped results.csv with padded column names in several
    # versions, so the headers are stripped rather than matched verbatim.
    with open(path, 'r', newline='') as handle:
        reader = csv.DictReader(handle)
        return [
            {(key or '').strip(): (value or '').strip() for key, value in row.items()}
            for row in reader
        ]


def best_pose_epoch_from_results(path: Path) -> int:
    """Return the epoch with the highest pose mAP50-95, matching the training notebook's comparison cell."""
    rows = read_rows(path)
    if not rows:
        raise ValueError(f'{path} holds no epoch rows.')

    best = max(rows, key=lambda row: float(row[POSE_MAP50_95_COLUMN]))
    # The epoch column's own value, never the row index: the two differ under
    # any 0-indexed or resumed-run convention, and reporting the row index would
    # silently shift every figure by one.
    return int(float(best['epoch']))


def checkpoint_train_metrics(path: Path) -> dict[str, float]:
    """Return the validation metrics stored inside a checkpoint by the epoch that saved it."""
    # weights_only=False because an Ultralytics checkpoint carries the model
    # object alongside its metrics, which is also how Ultralytics loads it.
    checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
    metrics = checkpoint.get('train_metrics') if isinstance(checkpoint, dict) else None
    if not metrics:
        raise ValueError(
            f'{path} stores no train_metrics, so the epoch it was saved at cannot be '
            f'recovered by matching against results.csv.'
        )
    return metrics


def checkpoint_epoch_from_results(metrics: dict[str, float], rows: list[dict[str, str]]) -> int:
    """Return the epoch whose results row matches a checkpoint's stored metrics."""
    # A saved best.pt records epoch -1 once training completes, so the epoch it
    # was actually written at is recoverable only by matching what it measured.
    matches = [
        row for row in rows
        if all(
            abs(float(row[column]) - float(metrics[column])) <= CHECKPOINT_MATCH_TOLERANCE
            for column in CHECKPOINT_MATCH_COLUMNS
        )
    ]
    if not matches:
        raise ValueError(
            'No epoch in results.csv matches the metrics stored in the checkpoint, so the epoch it '
            'was saved at cannot be determined. The checkpoint and the results.csv are most '
            'likely from different runs.'
        )
    if len(matches) > 1:
        epochs = [int(float(row['epoch'])) for row in matches]
        raise ValueError(
            f'{len(matches)} epochs match the metrics stored in the checkpoint ({epochs}), so the '
            f'epoch it was saved at is ambiguous rather than merely unknown.'
        )
    return int(float(matches[0]['epoch']))


def resolved_epochs(spec: RunSpec) -> tuple[int, int]:
    """Return a run's best pose mAP50-95 epoch and the epoch its checkpoint was saved at, raising if either disagrees with the pre-registration."""
    rows = read_rows(results_csv_path(spec))
    if not rows:
        raise ValueError(f'{results_csv_path(spec)} holds no epoch rows.')

    best_pose = best_pose_epoch_from_results(results_csv_path(spec))
    if best_pose != spec.best_pose_map50_95_epoch:
        raise ValueError(
            f'Run {spec.run}: results.csv gives best pose mAP50-95 at epoch {best_pose}, but '
            f'{spec.best_pose_map50_95_epoch} was pre-registered. Either the run is not the one '
            f'that produced the registered figures, or the column being maximised has changed.'
        )

    saved_at = checkpoint_epoch_from_results(checkpoint_train_metrics(weights_path(spec)), rows)
    if saved_at != spec.checkpoint_epoch:
        raise ValueError(
            f'Run {spec.run}: best.pt matches epoch {saved_at}, but {spec.checkpoint_epoch} was '
            f'pre-registered. Either the checkpoint is not the one this run produced, or '
            f'the Ultralytics selection criterion has changed.'
        )
    return best_pose, saved_at


def require_split() -> int:
    """Check every grouped-split path the validation needs and return the test image count."""
    rebuild = 'Run training/split_court_keypoints.py to rebuild the source-video-grouped split.'
    require_path(DATA_YAML, 'Grouped dataset config', rebuild)
    require_path(TEST_IMAGE_DIR, 'Grouped test split images', rebuild)
    # Guarded exactly as the image directory is. Ultralytics treats an absent
    # label directory as a set of unlabelled images rather than an error, so
    # without this the validation would run to completion and report metrics
    # against nothing, a plausible-looking zero rather than a failure.
    require_path(TEST_LABEL_DIR, 'Grouped test split labels', rebuild)

    images = sorted(path for path in TEST_IMAGE_DIR.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(
            f'Grouped test split images directory {TEST_IMAGE_DIR} exists but holds no images. {rebuild}'
        )

    # Existence of the directory is not enough: an empty one, or one missing
    # some of the images' labels, produces the same silently-degraded metrics as
    # no directory at all. Matched by stem, the same correspondence
    # scripts/measure_test_split.py enforces per image.
    if not any(TEST_LABEL_DIR.glob('*.txt')):
        raise FileNotFoundError(
            f'Grouped test split labels directory {TEST_LABEL_DIR} exists but holds no label files. {rebuild}'
        )

    missing = [path.name for path in images if not (TEST_LABEL_DIR / f'{path.stem}.txt').exists()]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} of {len(images)} test images have no label in {TEST_LABEL_DIR}, '
            f'starting with {missing[0]}. Ultralytics scores an unlabelled image as having no '
            f'objects rather than skipping it, so this would silently depress both runs. {rebuild}'
        )

    # A property of the split rather than of either run, counted once and
    # written to both rows: the two runs validate the same 220 images, so a
    # difference between the rows would mean they had not.
    #
    # There is deliberately no instance count. This dataset labels one court per
    # image, so an instance count would duplicate the image count exactly and
    # read as an independent figure while being the same number.
    return len(images)


def validate_run(spec: RunSpec) -> tuple[float, float, float]:
    """Validate one run's checkpoint against the grouped test split, returning pose mAP50, pose mAP50-95 and box mAP50-95."""
    weights = weights_path(spec)

    # Identical settings for both runs: the checkpoint is the only variable,
    # which is what makes the comparison a controlled one.
    metrics = YOLO(str(weights)).val(
        data=str(DATA_YAML),
        split='test',
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=OUTPUT_PROJECT,
        name=f'run_{spec.run}',
    )
    return float(metrics.pose.map50), float(metrics.pose.map), float(metrics.box.map)


def measure_run(spec: RunSpec, epochs: tuple[int, int], n_images: int) -> RunResult:
    """Validate one run and pair its figures with the two epochs already resolved from its results.csv and checkpoint."""
    best_pose_epoch, checkpoint_epoch = epochs
    pose_map50, pose_map50_95, box_map50_95 = validate_run(spec)
    return RunResult(
        run=spec.run,
        best_pose_map50_95_epoch=best_pose_epoch,
        checkpoint_epoch=checkpoint_epoch,
        pose_map50=pose_map50,
        pose_map50_95=pose_map50_95,
        box_map50_95=box_map50_95,
        n_images=n_images,
    )


def run_row(spec: RunSpec, result: RunResult) -> list[object]:
    """Return one run's measured figures as a CSV row."""
    return [
        result.run, spec.schedule, spec.epochs, spec.patience,
        result.best_pose_map50_95_epoch, result.checkpoint_epoch,
        round(result.pose_map50, 4), round(result.pose_map50_95, 4),
        round(result.box_map50_95, 4), result.n_images,
    ]


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    """Write a header row and every supplied row to a CSV, creating the output directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def compare(run: str, quantity: str, expected: float, actual: float) -> tuple[str, str, float, float, bool]:
    """Return one pre-registered figure paired with the measured value and whether they agree within tolerance."""
    return (run, quantity, expected, actual, abs(actual - expected) <= TOLERANCE)


def collect_comparisons(results: dict[str, RunResult]) -> list[tuple[str, str, float, float, bool]]:
    """Return every pre-registered figure compared against what this run measured."""
    comparisons: list[tuple[str, str, float, float, bool]] = []
    for spec in RUNS:
        result = results[spec.run]
        comparisons.append(compare(spec.run, 'test_pose_map50', spec.pose_map50, result.pose_map50))
        comparisons.append(compare(spec.run, 'test_pose_map50_95', spec.pose_map50_95, result.pose_map50_95))
        comparisons.append(compare(spec.run, 'test_box_map50_95', spec.box_map50_95, result.box_map50_95))
    return comparisons


def print_comparisons(comparisons: list[tuple[str, str, float, float, bool]]) -> bool:
    """Print each pre-registered figure as PASS or MISMATCH, returning whether all agreed within tolerance."""
    print()
    print(f'Pre-registered figures (tolerance {TOLERANCE})')
    print('run'.ljust(5) + ' ' + 'quantity'.ljust(20) + ' '
          + 'expected'.rjust(9) + ' ' + 'measured'.rjust(9) + ' ' + 'delta'.rjust(9) + '  verdict')
    for run, quantity, expected, actual, ok in comparisons:
        verdict = 'PASS' if ok else 'MISMATCH'
        print(f'{run:5s} {quantity:20s} {expected:9.4f} {actual:9.4f} {actual - expected:9.4f}  {verdict}')

    failures = [row for row in comparisons if not row[4]]
    if failures:
        print()
        print(f'{len(failures)} of {len(comparisons)} figures did not match within {TOLERANCE}.')
        print('A mismatch is a finding. Investigate what changed before citing these figures.')
    return not failures


def k1_holds(results: dict[str, RunResult]) -> bool:
    """Return whether Run B's test pose mAP50-95 exceeds Run A's, which is the comparison K1 was refuted on."""
    return results['B'].pose_map50_95 > results['A'].pose_map50_95


def print_k1_verdict(results: dict[str, RunResult]) -> bool:
    """Print which run won on test pose mAP50-95 and by how much, returning whether K1 stays refuted."""
    a, b = results['A'].pose_map50_95, results['B'].pose_map50_95
    winner, margin = ('B', b - a) if b > a else ('A', a - b)

    print()
    print(f'Run {winner} wins on test pose mAP50-95 by {abs(margin):.4f} '
          f'({margin * 100:+.2f} points): A {a:.4f}, B {b:.4f}.')

    # Pinned independently of whether the individual figures reproduce. K1
    # predicted Run A would win; the direction is the claim the pre-registration
    # is scored on, so a reversal is a different finding from a figure drifting
    # by a thousandth, and must not be reported as the same thing.
    if k1_holds(results):
        print('K1 (Run A beats Run B on test pose mAP50-95) is REFUTED, as recorded in the spec.')
        return True

    print('K1 REVERSED: Run A now leads, contradicting the recorded refutation.')
    print('The spec records refuted predictions as refuted and never rewrites them, so this '
          'is a finding about the measurement, not a licence to amend the spec.')
    return False


def main() -> int:
    """Validate both runs, write the comparison CSV, and return a non-zero exit code on any mismatch."""
    # Every precondition is checked before the first YOLO() is constructed. A
    # missing Run B checkpoint, or a best epoch disagreeing with the
    # pre-registration, is cheap to detect and would otherwise surface only
    # after Run A's full GPU validation pass had already run.
    n_images = require_split()
    print(f'Grouped test split: {n_images} images.')

    for spec in RUNS:
        weights_path(spec)

    # TWO DIFFERENT SELECTION CRITERIA, reported side by side because they do
    # not agree and the difference is the point:
    #
    #   best_pose_map50_95_epoch: argmax of metrics/mAP50-95(P) over the
    #     training run, which is what training/train_court_keypoints.ipynb's
    #     comparison cell computed and what the pre-registered 98 and 439 are.
    #   checkpoint_epoch: the epoch Ultralytics actually wrote best.pt at,
    #     selected by ITS fitness: 0.1 x mAP50 + 0.9 x mAP50-95, summed over the
    #     pose and box heads. That is 100 for Run A and 494 for Run B.
    #
    # So the checkpoint carried into production is not the pose mAP50-95
    # optimum: models/keypoints.pt is byte-identical to Run B's best.pt and is
    # therefore epoch 494, not 439. Reporting only the argmax would describe a
    # checkpoint that was never saved, and reporting only Ultralytics' choice
    # would not reproduce the registered figures.
    epochs = {spec.run: resolved_epochs(spec) for spec in RUNS}
    for spec in RUNS:
        best_pose, saved_at = epochs[spec.run]
        print(f'Run {spec.run}: best pose mAP50-95 at epoch {best_pose}; '
              f'best.pt saved at epoch {saved_at} by Ultralytics fitness.')

    results: dict[str, RunResult] = {}
    rows: list[list[object]] = []
    for spec in RUNS:
        print()
        print(f'[Run {spec.run}] validating {spec.schedule}...')
        result = measure_run(spec, epochs[spec.run], n_images)
        results[spec.run] = result
        rows.append(run_row(spec, result))
        print(f'[Run {spec.run}] pose mAP50 {result.pose_map50:.4f}, '
              f'pose mAP50-95 {result.pose_map50_95:.4f}, box mAP50-95 {result.box_map50_95:.4f} '
              f'(pose argmax epoch {result.best_pose_map50_95_epoch}, '
              f'checkpoint epoch {result.checkpoint_epoch})')

    write_csv(OUTPUT_CSV, HEADERS, rows)
    print()
    print(f'Wrote {OUTPUT_CSV}')

    # Both scored before returning, so one failure does not hide the other:
    # the figures reproducing and the verdict holding are separate claims.
    figures_ok = print_comparisons(collect_comparisons(results))
    verdict_ok = print_k1_verdict(results)
    return 0 if figures_ok and verdict_ok else 1


if __name__ == '__main__':
    sys.exit(main())
