"""Unit tests for the Stage 7 K1 run comparison (scripts/measure_keypoint_runs.py)."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

import scripts.measure_keypoint_runs as measure
from scripts.measure_keypoint_runs import (
    HEADERS,
    RUN_A,
    RUN_B,
    TOLERANCE,
    RunResult,
    RunSpec,
    best_pose_epoch_from_results,
    checkpoint_epoch_from_results,
    collect_comparisons,
    k1_holds,
    print_comparisons,
    print_k1_verdict,
    require_path,
    resolved_epochs,
    run_row,
    write_csv,
)

# The pre-registered figures themselves, so a fixture meant to reproduce them
# reproduces them exactly rather than approximately.
MEASURED_A = {'pose_map50': 0.9849, 'pose_map50_95': 0.9352, 'box_map50_95': 0.9252}
MEASURED_B = {'pose_map50': 0.9950, 'pose_map50_95': 0.9669, 'box_map50_95': 0.9353}


def result(run: str, figures: dict[str, float]) -> RunResult:
    """A RunResult carrying one run's figures and its pre-registered pair of epochs."""
    spec = RUN_A if run == 'A' else RUN_B
    return RunResult(
        run=run,
        best_pose_map50_95_epoch=spec.best_pose_map50_95_epoch,
        checkpoint_epoch=spec.checkpoint_epoch,
        pose_map50=figures['pose_map50'],
        pose_map50_95=figures['pose_map50_95'],
        box_map50_95=figures['box_map50_95'],
        n_images=220,
    )


def matching_results() -> dict[str, RunResult]:
    """Both runs reproducing their pre-registered figures exactly."""
    return {'A': result('A', MEASURED_A), 'B': result('B', MEASURED_B)}


def results_csv(path: Path, best_epoch: int, n_epochs: int, padded: bool = False) -> Path:
    """Write a minimal Ultralytics-shaped results.csv whose fittest row is the given epoch."""
    columns = ['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/mAP50(P)', 'metrics/mAP50-95(P)']
    if padded:
        # Ultralytics has shipped results.csv with padded column names in
        # several versions, so the reader must strip them.
        columns = [f'   {name}  ' for name in columns]

    rows = []
    for epoch in range(1, n_epochs + 1):
        # The fittest row is the target epoch; every other row is strictly worse
        # on all four metrics, so the argmax is unambiguous.
        value = 0.9 if epoch == best_epoch else 0.5
        rows.append([epoch, value, value, value, value])

    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


# --- CSV shape and headers -------------------------------------------------

def test_a_run_row_has_one_field_per_header() -> None:
    row = run_row(RUN_A, result('A', MEASURED_A))

    assert len(row) == len(HEADERS)
    assert row[0] == 'A'


def test_the_headers_are_the_columns_the_brief_specifies() -> None:
    assert HEADERS == [
        'run', 'schedule', 'epochs', 'patience',
        'best_pose_map50_95_epoch', 'checkpoint_epoch',
        'test_pose_map50', 'test_pose_map50_95', 'test_box_map50_95',
        'n_images',
    ]


def test_a_row_carries_its_runs_schedule_and_patience() -> None:
    # The schedule is the whole point of the comparison, so it belongs in the
    # artefact rather than only in the run directory's name.
    row_a = run_row(RUN_A, result('A', MEASURED_A))
    row_b = run_row(RUN_B, result('B', MEASURED_B))

    assert row_a[HEADERS.index('epochs')] == 100
    assert row_a[HEADERS.index('patience')] == 30
    assert row_b[HEADERS.index('epochs')] == 500
    assert row_b[HEADERS.index('patience')] == 100


def test_written_csv_round_trips_with_a_header_and_one_row_per_run(tmp_path: Path) -> None:
    rows = [run_row(RUN_A, result('A', MEASURED_A)), run_row(RUN_B, result('B', MEASURED_B))]
    path = tmp_path / 'nested' / 'stage7_run_comparison.csv'

    write_csv(path, HEADERS, rows)
    read_back = list(csv.reader(path.read_text(encoding='utf-8').splitlines()))

    assert read_back[0] == HEADERS
    assert len(read_back) == 3, 'a header and one row per run'
    assert [row[0] for row in read_back[1:]] == ['A', 'B']
    assert all(len(row) == len(HEADERS) for row in read_back)


def test_both_rows_report_the_same_image_count() -> None:
    # The two runs validate the same 220 images, so a difference between the
    # rows would mean they had not been given the same split.
    rows = [run_row(RUN_A, result('A', MEASURED_A)), run_row(RUN_B, result('B', MEASURED_B))]

    assert rows[0][HEADERS.index('n_images')] == rows[1][HEADERS.index('n_images')]


def test_no_instance_count_is_reported() -> None:
    # This dataset labels one court per image, so an instance count would
    # duplicate the image count exactly while reading as an independent figure.
    assert 'n_instances' not in HEADERS


# --- the tolerance comparison ----------------------------------------------

def test_figures_reproducing_exactly_all_pass() -> None:
    comparisons = collect_comparisons(matching_results())

    assert len(comparisons) == 6, 'three figures per run'
    assert all(row[4] for row in comparisons)


def test_the_tolerance_is_roughly_twice_the_observed_gpu_drift() -> None:
    # The 18 June re-validation diagnostic measured GPU float non-determinism on
    # this exact operation at 0.0001 to 0.0006. Tightening this constant towards
    # the observed maximum would start failing runs for reasons that are not
    # findings, which is the whole reason exact equality is not used.
    assert TOLERANCE == 0.001
    assert TOLERANCE > 0.0006


def test_a_deviation_inside_the_tolerance_passes() -> None:
    # 0.0009 is within the 0.001 tolerance, which exists because the 18 June
    # diagnostic measured GPU float non-determinism at 0.0001 to 0.0006 on this
    # exact operation. A drift that size is not a finding.
    drifted = dict(MEASURED_B)
    drifted['pose_map50_95'] = RUN_B.pose_map50_95 + 0.0009
    comparisons = collect_comparisons({'A': result('A', MEASURED_A), 'B': result('B', drifted)})

    assert all(row[4] for row in comparisons)


def test_a_deviation_outside_the_tolerance_fails() -> None:
    drifted = dict(MEASURED_B)
    drifted['pose_map50_95'] = RUN_B.pose_map50_95 + 0.0011
    comparisons = collect_comparisons({'A': result('A', MEASURED_A), 'B': result('B', drifted)})
    failed = [row for row in comparisons if not row[4]]

    assert len(failed) == 1
    assert failed[0][0] == 'B'
    assert failed[0][1] == 'test_pose_map50_95'


def test_the_tolerance_is_symmetric() -> None:
    # A figure that drifts DOWN by more than the tolerance must fail too, which
    # a comparison written as actual - expected <= TOLERANCE would not catch.
    drifted = dict(MEASURED_A)
    drifted['pose_map50'] = RUN_A.pose_map50 - 0.0011
    comparisons = collect_comparisons({'A': result('A', drifted), 'B': result('B', MEASURED_B)})

    assert not all(row[4] for row in comparisons)


# There is deliberately no exactly-at-the-tolerance test. The comparison uses
# <=, so the bound is inclusive, but that case cannot be constructed at these
# decimal values: 0.9252 + 0.001 evaluates to a hair above 0.9262 in binary
# floating point, giving a delta of 0.0010000000000000009. A test asserting
# inclusivity here would pass or fail on representation luck rather than on the
# comparison's behaviour, and nothing depends on the boundary: the measured
# figures carry four decimals and the tolerance is twice the observed drift.


def test_every_pre_registered_figure_is_compared() -> None:
    comparisons = collect_comparisons(matching_results())
    scoped = {(run, quantity) for run, quantity, _, _, _ in comparisons}

    for run in ('A', 'B'):
        for quantity in ('test_pose_map50', 'test_pose_map50_95', 'test_box_map50_95'):
            assert (run, quantity) in scoped


def test_print_comparisons_returns_true_and_prints_pass_when_everything_matches(capsys: pytest.CaptureFixture) -> None:
    ok = print_comparisons(collect_comparisons(matching_results()))
    out = capsys.readouterr().out

    assert ok is True
    assert 'PASS' in out
    assert 'MISMATCH' not in out


def test_print_comparisons_returns_false_and_names_the_mismatch(capsys: pytest.CaptureFixture) -> None:
    drifted = dict(MEASURED_A)
    drifted['pose_map50_95'] = RUN_A.pose_map50_95 + 0.02
    ok = print_comparisons(collect_comparisons({'A': result('A', drifted), 'B': result('B', MEASURED_B)}))
    out = capsys.readouterr().out

    assert ok is False
    assert 'MISMATCH' in out
    assert 'test_pose_map50_95' in out
    assert 'A mismatch is a finding' in out


# --- the K1 verdict --------------------------------------------------------

def test_k1_holds_when_run_b_leads() -> None:
    # K1 predicted Run A would win and was refuted; Run B leading is the
    # recorded outcome.
    assert k1_holds(matching_results()) is True


def test_k1_does_not_hold_when_the_result_is_reversed() -> None:
    reversed_results = {'A': result('A', MEASURED_B), 'B': result('B', MEASURED_A)}

    assert k1_holds(reversed_results) is False


def test_the_verdict_is_pinned_independently_of_the_individual_figures() -> None:
    # Both runs' figures swapped: every pre-registered figure still appears
    # somewhere, so a comparison alone could report all six as matching, yet
    # the claim the spec is scored on has reversed.
    reversed_results = {'A': result('A', MEASURED_B), 'B': result('B', MEASURED_A)}

    assert k1_holds(reversed_results) is False


def test_print_k1_verdict_reports_the_winner_and_the_margin(capsys: pytest.CaptureFixture) -> None:
    ok = print_k1_verdict(matching_results())
    out = capsys.readouterr().out

    assert ok is True
    assert 'Run B wins' in out
    # 0.9669 - 0.9352 = 0.0317, the 3.2 points the refutation rests on.
    assert '0.0317' in out
    assert 'REFUTED' in out


def test_print_k1_verdict_reports_a_reversal_rather_than_a_pass(capsys: pytest.CaptureFixture) -> None:
    ok = print_k1_verdict({'A': result('A', MEASURED_B), 'B': result('B', MEASURED_A)})
    out = capsys.readouterr().out

    assert ok is False
    assert 'REVERSED' in out
    assert 'Run A wins' in out
    # The spec never rewrites a refuted prediction, so a reversal is a finding
    # about the measurement rather than a licence to amend the record.
    assert 'never rewrites' in out


def test_an_exact_tie_does_not_count_as_k1_remaining_refuted() -> None:
    # A strict inequality: K1 is scored on Run B exceeding Run A, and a tie is
    # not evidence for the recorded refutation.
    tied = {'A': result('A', MEASURED_A), 'B': result('B', MEASURED_A)}

    assert k1_holds(tied) is False


# --- the two epochs ---------------------------------------------------------

def test_the_best_pose_epoch_is_the_argmax_of_pose_map50_95(tmp_path: Path) -> None:
    # Matching the training notebook's comparison cell, which is what produced
    # the pre-registered 98 and 439.
    path = results_csv(tmp_path / 'results.csv', best_epoch=98, n_epochs=100)

    assert best_pose_epoch_from_results(path) == 98


def test_the_argmax_ignores_the_box_columns(tmp_path: Path) -> None:
    # Ultralytics' own fitness weights the box head too, so a row that wins on
    # box and loses on pose must NOT be selected here. That difference between
    # the two criteria is exactly why the two epoch columns disagree.
    path = tmp_path / 'results.csv'
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                         'metrics/mAP50(P)', 'metrics/mAP50-95(P)'])
        # Epoch 1 wins on box by a mile; epoch 2 wins on pose by a little.
        writer.writerow([1, 0.99, 0.99, 0.50, 0.50])
        writer.writerow([2, 0.10, 0.10, 0.60, 0.60])

    assert best_pose_epoch_from_results(path) == 2


def test_padded_column_names_are_still_read(tmp_path: Path) -> None:
    path = results_csv(tmp_path / 'results.csv', best_epoch=439, n_epochs=500, padded=True)

    assert best_pose_epoch_from_results(path) == 439


def test_the_epoch_column_value_is_used_not_the_row_index(tmp_path: Path) -> None:
    # A 1-indexed epoch column against a 0-indexed row position differ by one,
    # and reporting the row index would shift every figure silently.
    path = tmp_path / 'results.csv'
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                         'metrics/mAP50(P)', 'metrics/mAP50-95(P)'])
        # The best row sits at index 0 but records epoch 7.
        writer.writerow([7, 0.9, 0.9, 0.9, 0.9])
        writer.writerow([8, 0.5, 0.5, 0.5, 0.5])

    assert best_pose_epoch_from_results(path) == 7


def test_an_empty_results_csv_raises_rather_than_returning_a_default(tmp_path: Path) -> None:
    path = tmp_path / 'results.csv'
    with open(path, 'w', newline='') as handle:
        csv.writer(handle).writerow(['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                                     'metrics/mAP50(P)', 'metrics/mAP50-95(P)'])

    with pytest.raises(ValueError, match='no epoch rows'):
        best_pose_epoch_from_results(path)


def test_the_checkpoint_epoch_is_found_by_matching_its_stored_metrics() -> None:
    # A saved best.pt records epoch -1 once training completes, so the epoch it
    # was written at is recoverable only by matching what it measured.
    rows = [
        {'epoch': '98', 'metrics/mAP50(B)': '0.90', 'metrics/mAP50-95(B)': '0.90',
         'metrics/mAP50(P)': '0.90', 'metrics/mAP50-95(P)': '0.90'},
        {'epoch': '100', 'metrics/mAP50(B)': '0.95', 'metrics/mAP50-95(B)': '0.93',
         'metrics/mAP50(P)': '0.98', 'metrics/mAP50-95(P)': '0.88'},
    ]
    stored = {'metrics/mAP50(B)': 0.95, 'metrics/mAP50-95(B)': 0.93,
              'metrics/mAP50(P)': 0.98, 'metrics/mAP50-95(P)': 0.88}

    assert checkpoint_epoch_from_results(stored, rows) == 100


def test_the_checkpoint_match_tolerates_the_rounding_in_results_csv() -> None:
    # results.csv rounds; the checkpoint stores full floats. An exact match
    # would fail on every real run.
    rows = [{'epoch': '494', 'metrics/mAP50(B)': '0.99501', 'metrics/mAP50-95(B)': '0.93531',
             'metrics/mAP50(P)': '0.99503', 'metrics/mAP50-95(P)': '0.96688'}]
    stored = {'metrics/mAP50(B)': 0.995013, 'metrics/mAP50-95(B)': 0.935307,
              'metrics/mAP50(P)': 0.995034, 'metrics/mAP50-95(P)': 0.966881}

    assert checkpoint_epoch_from_results(stored, rows) == 494


def test_a_checkpoint_matching_no_epoch_raises() -> None:
    rows = [{'epoch': '1', 'metrics/mAP50(B)': '0.10', 'metrics/mAP50-95(B)': '0.10',
             'metrics/mAP50(P)': '0.10', 'metrics/mAP50-95(P)': '0.10'}]
    stored = {'metrics/mAP50(B)': 0.90, 'metrics/mAP50-95(B)': 0.90,
              'metrics/mAP50(P)': 0.90, 'metrics/mAP50-95(P)': 0.90}

    with pytest.raises(ValueError, match='No epoch in results.csv matches'):
        checkpoint_epoch_from_results(stored, rows)


def test_a_checkpoint_matching_several_epochs_raises_rather_than_guessing() -> None:
    # Ambiguous is a different state from unknown, and picking the first would
    # report a specific epoch the evidence does not single out.
    identical = {'metrics/mAP50(B)': '0.90', 'metrics/mAP50-95(B)': '0.90',
                 'metrics/mAP50(P)': '0.90', 'metrics/mAP50-95(P)': '0.90'}
    rows = [{'epoch': '10', **identical}, {'epoch': '11', **identical}]
    stored = {key: 0.90 for key in identical}

    with pytest.raises(ValueError, match='ambiguous'):
        checkpoint_epoch_from_results(stored, rows)


def stub_epoch_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, best_epoch: int,
                       stored: dict[str, float], n_epochs: int = 100) -> None:
    """Point the epoch resolution at a temporary results.csv and a faked checkpoint metrics dict."""
    path = results_csv(tmp_path / 'results.csv', best_epoch=best_epoch, n_epochs=n_epochs)
    monkeypatch.setattr(measure, 'results_csv_path', lambda spec: path)
    monkeypatch.setattr(measure, 'weights_path', lambda spec: tmp_path / 'best.pt')
    monkeypatch.setattr(measure, 'checkpoint_train_metrics', lambda path: stored)


PEAK_METRICS = {
    'metrics/mAP50(B)': 0.9, 'metrics/mAP50-95(B)': 0.9,
    'metrics/mAP50(P)': 0.9, 'metrics/mAP50-95(P)': 0.9,
}


def test_both_epochs_agreeing_with_the_pre_registration_are_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # results_csv() makes only the target epoch score 0.9, so matching 0.9
    # across all four columns locates that same epoch as the checkpoint's.
    stub_epoch_sources(monkeypatch, tmp_path, best_epoch=98, stored=PEAK_METRICS)
    spec = replace(RUN_A, checkpoint_epoch=98)

    assert resolved_epochs(spec) == (98, 98)


def test_a_best_pose_epoch_disagreeing_with_the_pre_registration_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Registered as 98; this run's results.csv peaks at 97.
    stub_epoch_sources(monkeypatch, tmp_path, best_epoch=97, stored=PEAK_METRICS)

    with pytest.raises(ValueError, match=r'best pose mAP50-95 at epoch 97, but 98 was pre-registered'):
        resolved_epochs(RUN_A)


def test_a_checkpoint_epoch_disagreeing_with_the_pre_registration_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The pose argmax agrees at 98, but the checkpoint matches epoch 98 too,
    # against the registered 100, so only the second check can catch it.
    stub_epoch_sources(monkeypatch, tmp_path, best_epoch=98, stored=PEAK_METRICS)

    with pytest.raises(ValueError, match=r'matches epoch 98, but 100 was pre-registered'):
        resolved_epochs(RUN_A)


def test_the_two_registered_epochs_differ_for_both_runs() -> None:
    # The whole reason both columns exist: Ultralytics' fitness weights the box
    # head as well, so the checkpoint it saved is not the pose mAP50-95 optimum.
    # models/keypoints.pt is byte-identical to Run B's best.pt, so production
    # runs epoch 494 rather than 439.
    assert (RUN_A.best_pose_map50_95_epoch, RUN_A.checkpoint_epoch) == (98, 100)
    assert (RUN_B.best_pose_map50_95_epoch, RUN_B.checkpoint_epoch) == (439, 494)
    for spec in (RUN_A, RUN_B):
        assert spec.best_pose_map50_95_epoch != spec.checkpoint_epoch


# --- missing artefacts ------------------------------------------------------

def test_require_path_returns_an_existing_path(tmp_path: Path) -> None:
    present = tmp_path / 'here.txt'
    present.write_text('x')

    assert require_path(present, 'Thing', 'hint') == present


def test_require_path_names_the_missing_path_and_how_to_produce_it(tmp_path: Path) -> None:
    # Both the checkpoints and the dataset are gitignored, so neither ships.
    # This message is what an examiner running the script actually learns from.
    missing = tmp_path / 'absent' / 'best.pt'

    with pytest.raises(FileNotFoundError) as error:
        require_path(missing, 'Run A checkpoint', 'It is written by the training notebook.')

    message = str(error.value)
    assert str(missing) in message
    assert 'Run A checkpoint' in message
    assert 'written by the training notebook' in message


def test_a_missing_label_directory_fails_rather_than_reporting_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Ultralytics treats an absent label directory as a set of unlabelled
    # images rather than an error, so without this guard the validation would
    # run to completion and report metrics against nothing.
    images = tmp_path / 'images'
    images.mkdir()
    (images / 'frame.jpg').write_text('x')
    monkeypatch.setattr(measure, 'DATA_YAML', tmp_path / 'data.yaml')
    (tmp_path / 'data.yaml').write_text('x')
    monkeypatch.setattr(measure, 'TEST_IMAGE_DIR', images)
    monkeypatch.setattr(measure, 'TEST_LABEL_DIR', tmp_path / 'labels')

    with pytest.raises(FileNotFoundError, match='Grouped test split labels'):
        measure.require_split()


def test_the_split_check_returns_the_image_count_when_every_path_is_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    images = tmp_path / 'images'
    labels = tmp_path / 'labels'
    images.mkdir()
    labels.mkdir()
    for index in range(3):
        (images / f'frame_{index}.jpg').write_text('x')
        (labels / f'frame_{index}.txt').write_text('0 0.5 0.5 0.2 0.2')
    # A non-image file in the same directory must not be counted, and must not
    # be required to have a label either.
    (images / 'notes.txt').write_text('x')
    (tmp_path / 'data.yaml').write_text('x')
    monkeypatch.setattr(measure, 'DATA_YAML', tmp_path / 'data.yaml')
    monkeypatch.setattr(measure, 'TEST_IMAGE_DIR', images)
    monkeypatch.setattr(measure, 'TEST_LABEL_DIR', tmp_path / 'labels')

    assert measure.require_split() == 3


def temp_spec(tmp_path: Path, spec: RunSpec, with_checkpoint: bool = False) -> RunSpec:
    """Return a copy of a run spec pointing at a temporary directory, optionally containing a checkpoint file."""
    # Every path guard is exercised against tmp_path rather than the real run
    # directories. runs/ is gitignored, so a test asserting a real checkpoint is
    # ABSENT passes locally and fails on JupyterHub, where both checkpoints
    # exist: the guard is a property of the code, not of which machine the
    # suite happens to run on. Both versions of these tests were confirmed
    # against dummy files at the real paths before this helper was introduced.
    directory = tmp_path / f'run_{spec.run}'
    if with_checkpoint:
        (directory / 'weights').mkdir(parents=True)
        (directory / 'weights' / 'best.pt').write_text('not a real checkpoint')
    return replace(spec, directory=directory)


def build_split(tmp_path: Path, n_images: int = 3, n_labels: int | None = None) -> None:
    """Point the module at a temporary grouped split with the given number of images and labels."""
    images = tmp_path / 'images'
    labels = tmp_path / 'labels'
    images.mkdir()
    labels.mkdir()
    (tmp_path / 'data.yaml').write_text('x')
    for index in range(n_images):
        (images / f'frame_{index}.jpg').write_text('x')
    for index in range(n_images if n_labels is None else n_labels):
        (labels / f'frame_{index}.txt').write_text('0 0.5 0.5 0.2 0.2')


def test_an_empty_label_directory_fails_rather_than_validating_against_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The directory existing is not enough: Ultralytics scores an unlabelled
    # image as having no objects rather than skipping it, so an empty label
    # directory depresses both runs identically and silently.
    build_split(tmp_path, n_images=3, n_labels=0)
    monkeypatch.setattr(measure, 'DATA_YAML', tmp_path / 'data.yaml')
    monkeypatch.setattr(measure, 'TEST_IMAGE_DIR', tmp_path / 'images')
    monkeypatch.setattr(measure, 'TEST_LABEL_DIR', tmp_path / 'labels')

    with pytest.raises(FileNotFoundError, match='holds no label files'):
        measure.require_split()


def test_a_single_missing_label_fails_and_names_the_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Two of three labelled. A partially-labelled split is the harder case,
    # because it still produces plausible metrics rather than obviously wrong
    # ones, and the count in the message is what makes the gap visible.
    build_split(tmp_path, n_images=3, n_labels=2)
    monkeypatch.setattr(measure, 'DATA_YAML', tmp_path / 'data.yaml')
    monkeypatch.setattr(measure, 'TEST_IMAGE_DIR', tmp_path / 'images')
    monkeypatch.setattr(measure, 'TEST_LABEL_DIR', tmp_path / 'labels')

    with pytest.raises(FileNotFoundError) as error:
        measure.require_split()

    message = str(error.value)
    assert '1 of 3 test images have no label' in message
    assert 'frame_2.jpg' in message


def test_an_empty_image_directory_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    build_split(tmp_path, n_images=0)
    monkeypatch.setattr(measure, 'DATA_YAML', tmp_path / 'data.yaml')
    monkeypatch.setattr(measure, 'TEST_IMAGE_DIR', tmp_path / 'images')
    monkeypatch.setattr(measure, 'TEST_LABEL_DIR', tmp_path / 'labels')

    with pytest.raises(FileNotFoundError, match='holds no images'):
        measure.require_split()


def test_a_fully_labelled_split_passes_the_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The positive case, so the three tests above cannot pass merely because
    # require_split() always raises.
    build_split(tmp_path, n_images=4)
    monkeypatch.setattr(measure, 'DATA_YAML', tmp_path / 'data.yaml')
    monkeypatch.setattr(measure, 'TEST_IMAGE_DIR', tmp_path / 'images')
    monkeypatch.setattr(measure, 'TEST_LABEL_DIR', tmp_path / 'labels')

    assert measure.require_split() == 4


def test_a_missing_checkpoint_is_reported_before_ultralytics_is_reached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The guard must run before YOLO() is constructed, or the reader gets an
    # Ultralytics stack trace instead of the message above.
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError('YOLO must not be constructed when the checkpoint is absent')

    monkeypatch.setattr(measure, 'YOLO', fail)

    with pytest.raises(FileNotFoundError, match='Run A checkpoint'):
        measure.validate_run(temp_spec(tmp_path, RUN_A))


def test_a_present_checkpoint_is_not_reported_as_missing(tmp_path: Path) -> None:
    # The other half of the guard, so the test above cannot pass merely because
    # require_path always raises.
    spec = temp_spec(tmp_path, RUN_A, with_checkpoint=True)

    assert measure.weights_path(spec) == spec.directory / 'weights' / 'best.pt'


# --- the exit code ----------------------------------------------------------

def stub_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, figures: dict[str, dict[str, float]]) -> Path:
    """Point main() at a temporary CSV and stub out validation, the split counts and the epoch lookup."""
    path = tmp_path / 'stage7_run_comparison.csv'
    monkeypatch.setattr(measure, 'OUTPUT_CSV', path)
    monkeypatch.setattr(measure, 'require_split', lambda: 220)
    monkeypatch.setattr(measure, 'weights_path', lambda spec: Path('stub/best.pt'))
    monkeypatch.setattr(
        measure, 'resolved_epochs',
        lambda spec: (spec.best_pose_map50_95_epoch, spec.checkpoint_epoch),
    )
    monkeypatch.setattr(
        measure, 'validate_run',
        lambda spec: (
            figures[spec.run]['pose_map50'],
            figures[spec.run]['pose_map50_95'],
            figures[spec.run]['box_map50_95'],
        ),
    )
    return path


def test_every_precondition_is_checked_before_any_validation_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Run B's best epoch disagrees with the pre-registration. That must surface
    # before Run A's validation pass, which on a GPU is minutes of work whose
    # result is discarded the moment the mismatch is found.
    stub_main(monkeypatch, tmp_path, {'A': MEASURED_A, 'B': MEASURED_B})

    def fail(spec: object) -> object:
        raise AssertionError('no run may be validated before every precondition is checked')

    monkeypatch.setattr(measure, 'validate_run', fail)
    monkeypatch.setattr(
        measure, 'resolved_epochs',
        lambda spec: (98, 100) if spec.run == 'A' else _raise_epoch_mismatch(),
    )

    with pytest.raises(ValueError, match='pre-registered'):
        measure.main()


def _raise_epoch_mismatch() -> tuple[int, int]:
    """Stand in for a run whose results.csv disagrees with its pre-registered best epoch."""
    raise ValueError('Run B: results.csv gives best pose mAP50-95 at epoch 300, but 439 was pre-registered.')


def test_a_missing_second_checkpoint_is_found_before_the_first_is_validated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The same fail-fast property for the checkpoints themselves: an absent
    # Run B best.pt must not be discovered only after Run A has been validated.
    #
    # RUNS is replaced with temporary specs rather than weights_path being
    # stubbed, so the REAL guard runs against a directory this test controls.
    # Stubbing weights_path would have tested the stub, and pointing at the real
    # runs/ directory would make the outcome depend on the machine.
    monkeypatch.setattr(measure, 'OUTPUT_CSV', tmp_path / 'stage7_run_comparison.csv')
    monkeypatch.setattr(measure, 'require_split', lambda: 220)
    monkeypatch.setattr(measure, 'RUNS', (
        temp_spec(tmp_path, RUN_A, with_checkpoint=True),
        temp_spec(tmp_path, RUN_B),
    ))

    def fail(spec: object) -> object:
        raise AssertionError('Run A must not be validated while Run B has no checkpoint')

    monkeypatch.setattr(measure, 'validate_run', fail)

    with pytest.raises(FileNotFoundError, match='Run B checkpoint'):
        measure.main()


def test_main_exits_zero_and_writes_the_csv_when_everything_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = stub_main(monkeypatch, tmp_path, {'A': MEASURED_A, 'B': MEASURED_B})

    assert measure.main() == 0
    rows = list(csv.reader(path.read_text(encoding='utf-8').splitlines()))
    assert rows[0] == HEADERS
    assert len(rows) == 3


def test_main_exits_non_zero_when_a_figure_drifts_beyond_the_tolerance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    drifted = dict(MEASURED_B)
    drifted['box_map50_95'] = RUN_B.box_map50_95 + 0.05
    path = stub_main(monkeypatch, tmp_path, {'A': MEASURED_A, 'B': drifted})

    assert measure.main() == 1
    # Written anyway: a mismatched run is exactly the one whose numbers someone
    # needs to open. The exit code is what marks it, not a missing file.
    assert path.exists()


def test_main_exits_non_zero_when_the_k1_verdict_reverses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Both runs' figures swapped, so all six figures still appear among the
    # measurements and only the verdict catches it.
    stub_main(monkeypatch, tmp_path, {'A': MEASURED_B, 'B': MEASURED_A})

    assert measure.main() == 1


def test_the_exit_code_consults_the_verdict_and_not_only_the_figures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The verdict's contribution to the exit code cannot be isolated through
    # real figures: the registered values are 0.0317 apart, thirty times the
    # tolerance, so a reversal always drags the figure comparison down with it.
    # The comparison is therefore stubbed to report everything as matching,
    # leaving the reversed verdict as the only possible cause of a non-zero
    # exit. Without this, dropping verdict_ok from main() passes the whole file.
    stub_main(monkeypatch, tmp_path, {'A': MEASURED_B, 'B': MEASURED_A})
    monkeypatch.setattr(measure, 'collect_comparisons', lambda results: [])

    assert measure.main() == 1


def test_a_reversed_verdict_fails_even_though_every_figure_is_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    stub_main(monkeypatch, tmp_path, {'A': MEASURED_B, 'B': MEASURED_A})

    measure.main()
    out = capsys.readouterr().out

    # The figure comparison reports six mismatches (each run measured the
    # other's numbers) AND the verdict reverses; both are printed, so neither
    # failure hides the other.
    assert 'MISMATCH' in out
    assert 'REVERSED' in out
