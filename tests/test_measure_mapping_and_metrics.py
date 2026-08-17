"""Unit tests for the Stage 8 and Stage 9 count measurement (scripts/measure_mapping_and_metrics.py)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import scripts.measure_mapping_and_metrics as measure
from basketball.homography.court_mapper import MappingReport
from basketball.metrics.player_metrics import MetricsReport
from scripts.measure_mapping_and_metrics import (
    MAPPING_HEADERS,
    METRICS_HEADERS,
    TOTAL_LABEL,
    collect_comparisons,
    mapping_row,
    mapping_total_row,
    metrics_row,
    metrics_total_row,
    print_comparisons,
    require_reconciliation,
    write_csv,
)

# The pre-registered figures themselves, so a fixture that is meant to match
# expectation matches it exactly rather than approximately.
CLIP_1 = {'n_frames': 117, 'mapped': 107, 'tracks': 13, 'displacements': 651, 'speeds': 643, 'suppressed': 8}
CLIP_2 = {'n_frames': 174, 'mapped': 174, 'tracks': 14, 'displacements': 1609, 'speeds': 1606, 'suppressed': 3}
CLIP_3 = {'n_frames': 243, 'mapped': 198, 'tracks': 17, 'displacements': 1581, 'speeds': 1572, 'suppressed': 9}


def mapping_report(spec: dict[str, int]) -> MappingReport:
    """A reconciling MappingReport carrying one clip's expected frame counts."""
    return MappingReport(
        n_frames=spec['n_frames'],
        mapped_frames=spec['mapped'],
        insufficient_keypoints=spec['n_frames'] - spec['mapped'],
        positions_mapped=spec['displacements'],
        positions_dropped_out_of_bounds=2,
        positions_dropped_at_horizon=1,
    )


def metrics_report(spec: dict[str, int]) -> MetricsReport:
    """A MetricsReport carrying one clip's expected track and displacement counts."""
    return MetricsReport(
        n_frames=spec['n_frames'],
        total_distance_m={track_id: 1.0 for track_id in range(spec['tracks'])},
        displacements_measured=spec['displacements'],
        speeds_computed=spec['speeds'],
        speeds_suppressed_by_gap=spec['suppressed'],
    )


def all_mapping_reports() -> dict[str, MappingReport]:
    """One matching MappingReport per clip."""
    return {
        'clip_1': mapping_report(CLIP_1),
        'clip_2': mapping_report(CLIP_2),
        'clip_3': mapping_report(CLIP_3),
    }


def all_metrics_reports() -> dict[str, MetricsReport]:
    """One matching MetricsReport per clip."""
    return {
        'clip_1': metrics_report(CLIP_1),
        'clip_2': metrics_report(CLIP_2),
        'clip_3': metrics_report(CLIP_3),
    }


# --- CSV shape and headers -------------------------------------------------

def test_a_mapping_row_has_one_field_per_header() -> None:
    row = mapping_row('clip_1', mapping_report(CLIP_1))

    assert len(row) == len(MAPPING_HEADERS)
    assert row[0] == 'clip_1'


def test_a_metrics_row_has_one_field_per_header() -> None:
    row = metrics_row('clip_1', metrics_report(CLIP_1))

    assert len(row) == len(METRICS_HEADERS)
    assert row[0] == 'clip_1'


def test_the_mapping_headers_are_the_columns_the_brief_specifies() -> None:
    assert MAPPING_HEADERS == [
        'clip', 'n_frames', 'mapped_frames', 'unmapped_frames',
        'insufficient_keypoints', 'degenerate_keypoints', 'malformed_input',
        'positions_mapped', 'positions_dropped_out_of_bounds',
        'positions_dropped_at_horizon', 'reconciles',
    ]


def test_the_metrics_headers_are_the_columns_the_brief_specifies() -> None:
    assert METRICS_HEADERS == [
        'clip', 'n_frames', 'tracks', 'displacements_measured',
        'speeds_computed', 'speeds_suppressed_by_gap', 'speed_rate',
    ]


def test_tracks_is_the_number_of_tracks_that_covered_any_distance() -> None:
    # len(total_distance_m), not a frame count or a max track id: a track that
    # never appeared has no entry, and ids are not contiguous.
    report = metrics_report(CLIP_3)
    row = metrics_row('clip_3', report)

    assert row[METRICS_HEADERS.index('tracks')] == 17
    assert row[METRICS_HEADERS.index('tracks')] == len(report.total_distance_m)


def test_written_csvs_round_trip_with_a_header_and_every_row(tmp_path: Path) -> None:
    rows = [mapping_row(clip, mapping_report(spec)) for clip, spec in
            (('clip_1', CLIP_1), ('clip_2', CLIP_2), ('clip_3', CLIP_3))]
    rows.append(mapping_total_row(rows))
    path = tmp_path / 'nested' / 'stage8.csv'

    write_csv(path, MAPPING_HEADERS, rows)
    read_back = list(csv.reader(path.read_text(encoding='utf-8').splitlines()))

    assert read_back[0] == MAPPING_HEADERS
    assert len(read_back) == 5, 'a header, three clips and a TOTAL row'
    assert read_back[-1][0] == TOTAL_LABEL
    assert all(len(row) == len(MAPPING_HEADERS) for row in read_back)


def test_write_csv_creates_a_missing_output_directory(tmp_path: Path) -> None:
    path = tmp_path / 'does' / 'not' / 'exist' / 'out.csv'

    write_csv(path, METRICS_HEADERS, [metrics_row('clip_1', metrics_report(CLIP_1))])

    assert path.exists()


# --- TOTAL row arithmetic --------------------------------------------------

def test_the_mapping_total_sums_every_count_across_clips() -> None:
    rows = [mapping_row(clip, mapping_report(spec)) for clip, spec in
            (('clip_1', CLIP_1), ('clip_2', CLIP_2), ('clip_3', CLIP_3))]

    total = mapping_total_row(rows)

    assert total[0] == TOTAL_LABEL
    assert total[MAPPING_HEADERS.index('n_frames')] == 117 + 174 + 243
    assert total[MAPPING_HEADERS.index('mapped_frames')] == 107 + 174 + 198
    assert total[MAPPING_HEADERS.index('unmapped_frames')] == (117 - 107) + 0 + (243 - 198)


def test_the_metrics_total_sums_every_count_across_clips() -> None:
    rows = [metrics_row(clip, metrics_report(spec)) for clip, spec in
            (('clip_1', CLIP_1), ('clip_2', CLIP_2), ('clip_3', CLIP_3))]

    total = metrics_total_row(rows)

    assert total[0] == TOTAL_LABEL
    assert total[METRICS_HEADERS.index('displacements_measured')] == 3841
    assert total[METRICS_HEADERS.index('speeds_computed')] == 3821
    assert total[METRICS_HEADERS.index('speeds_suppressed_by_gap')] == 20


def test_the_total_speed_rate_is_recomputed_not_averaged() -> None:
    # The clips carry very different displacement counts, so a mean of the three
    # per-clip rates weights a 651-displacement clip equally with a 1609 one.
    # These fixtures make the two answers differ: one clip is tiny and perfect,
    # the other large and poor.
    small = MetricsReport(n_frames=10, total_distance_m={1: 1.0}, displacements_measured=10,
                          speeds_computed=10, speeds_suppressed_by_gap=0)
    large = MetricsReport(n_frames=100, total_distance_m={1: 1.0}, displacements_measured=100,
                          speeds_computed=50, speeds_suppressed_by_gap=50)
    rows = [metrics_row('small', small), metrics_row('large', large)]

    total = metrics_total_row(rows)
    pooled = 60 / 110
    averaged = (1.0 + 0.5) / 2

    assert total[METRICS_HEADERS.index('speed_rate')] == round(pooled, 4)
    assert total[METRICS_HEADERS.index('speed_rate')] != round(averaged, 4)


def test_the_total_tracks_cell_is_empty_rather_than_a_sum(tmp_path: Path) -> None:
    # A track id identifies a player within one clip and carries no meaning
    # across clips, so summing counts the same person once per clip they appear
    # in and reads as a squad size no clip has. The pre-registration leaves this
    # cell as a dash, and EXPECTED_METRICS_TOTALS registers no tracks total.
    rows = [metrics_row(clip, metrics_report(spec)) for clip, spec in
            (('clip_1', CLIP_1), ('clip_2', CLIP_2), ('clip_3', CLIP_3))]

    total = metrics_total_row(rows)
    cell = total[METRICS_HEADERS.index('tracks')]

    assert cell == ''
    assert cell != 13 + 14 + 17, 'the tracks column must not be summed across clips'
    # Empty through serialisation too, since the CSV is what a reader opens.
    rows.append(total)
    path = tmp_path / 'stage9.csv'
    write_csv(path, METRICS_HEADERS, rows)
    read_back = list(csv.reader(path.read_text(encoding='utf-8').splitlines()))

    assert read_back[-1][METRICS_HEADERS.index('tracks')] == ''
    # The rest of the row still carries its counts, so this is an empty cell
    # rather than a truncated or misaligned row.
    assert len(read_back[-1]) == len(METRICS_HEADERS)
    assert read_back[-1][METRICS_HEADERS.index('displacements_measured')] == '3841'


def test_the_total_reconciles_flag_is_derived_from_the_summed_counts() -> None:
    rows = [mapping_row(clip, mapping_report(spec)) for clip, spec in
            (('clip_1', CLIP_1), ('clip_2', CLIP_2), ('clip_3', CLIP_3))]

    total = mapping_total_row(rows)

    assert total[MAPPING_HEADERS.index('reconciles')] is True


def test_a_non_reconciling_clip_makes_the_total_flag_false() -> None:
    broken = MappingReport(n_frames=117, mapped_frames=100, insufficient_keypoints=5)
    rows = [mapping_row('clip_1', broken)]

    total = mapping_total_row(rows)

    assert total[MAPPING_HEADERS.index('reconciles')] is False


def test_a_zero_attempt_total_speed_rate_is_not_a_division_error() -> None:
    empty = MetricsReport(n_frames=0, total_distance_m={}, displacements_measured=0,
                          speeds_computed=0, speeds_suppressed_by_gap=0)

    total = metrics_total_row([metrics_row('clip_1', empty)])
    rate = total[METRICS_HEADERS.index('speed_rate')]

    assert rate != rate, 'an unattempted rate is NaN, not a fabricated 0.0'


# --- the reconciliation assertion ------------------------------------------

def test_require_reconciliation_passes_a_reconciling_report() -> None:
    require_reconciliation('clip_1', mapping_report(CLIP_1))


def test_require_reconciliation_raises_naming_the_clip_and_the_counts() -> None:
    # Frames are unaccounted for: 100 mapped + 5 unmapped against 117 frames.
    broken = MappingReport(n_frames=117, mapped_frames=100, insufficient_keypoints=5)

    with pytest.raises(ValueError, match=r'clip_2: mapping report does not reconcile'):
        require_reconciliation('clip_2', broken)


def test_the_reconciliation_error_reports_the_three_numbers() -> None:
    broken = MappingReport(n_frames=117, mapped_frames=100, insufficient_keypoints=5)

    with pytest.raises(ValueError) as error:
        require_reconciliation('clip_2', broken)

    message = str(error.value)
    assert '100' in message and '5' in message and '117' in message


# --- the pre-registered comparison -----------------------------------------

def test_matching_reports_produce_only_passes() -> None:
    comparisons = collect_comparisons(all_mapping_reports(), all_metrics_reports())

    assert comparisons, 'the comparison must not be silently empty'
    assert all(row[4] for row in comparisons)


def test_every_pre_registered_quantity_is_compared() -> None:
    comparisons = collect_comparisons(all_mapping_reports(), all_metrics_reports())
    scoped = {(scope, quantity) for scope, quantity, _, _, _ in comparisons}

    for clip in ('clip_1', 'clip_2', 'clip_3'):
        assert (clip, 'mapped_frames') in scoped
        for quantity in ('tracks', 'displacements_measured', 'speeds_computed', 'speeds_suppressed_by_gap'):
            assert (clip, quantity) in scoped
    for quantity in ('displacements_measured', 'speeds_computed', 'speeds_suppressed_by_gap'):
        assert (TOTAL_LABEL, quantity) in scoped


def test_a_drifted_speed_count_is_reported_as_a_mismatch() -> None:
    metrics = all_metrics_reports()
    metrics['clip_2'].speeds_computed += 1

    comparisons = collect_comparisons(all_mapping_reports(), metrics)
    failed = [row for row in comparisons if not row[4]]

    # The per-clip count and the total both move, so both must be reported.
    assert ('clip_2', 'speeds_computed', 1606, 1607, False) in comparisons
    assert (TOTAL_LABEL, 'speeds_computed', 3821, 3822, False) in comparisons
    assert len(failed) == 2


def test_a_drifted_mapped_frame_count_is_reported_as_a_mismatch() -> None:
    mapping = all_mapping_reports()
    mapping['clip_3'].mapped_frames = 197

    comparisons = collect_comparisons(mapping, all_metrics_reports())

    assert ('clip_3', 'mapped_frames', 198, 197, False) in comparisons


def test_print_comparisons_returns_true_and_prints_pass_when_everything_matches(capsys: pytest.CaptureFixture) -> None:
    comparisons = collect_comparisons(all_mapping_reports(), all_metrics_reports())

    ok = print_comparisons(comparisons)
    out = capsys.readouterr().out

    assert ok is True
    assert 'PASS' in out
    assert 'MISMATCH' not in out


def test_print_comparisons_returns_false_and_names_the_mismatch(capsys: pytest.CaptureFixture) -> None:
    metrics = all_metrics_reports()
    metrics['clip_1'].displacements_measured = 650

    ok = print_comparisons(collect_comparisons(all_mapping_reports(), metrics))
    out = capsys.readouterr().out

    assert ok is False
    assert 'MISMATCH' in out
    assert 'displacements_measured' in out
    # Both the registered and the measured value are shown, so the reader can
    # see the size of the drift without re-running anything.
    assert '651' in out and '650' in out


def test_a_mismatch_is_called_a_finding_rather_than_an_error(capsys: pytest.CaptureFixture) -> None:
    # The wording matters: a drifted figure means an input or a stage behaviour
    # changed, which needs investigating rather than silently re-registering.
    metrics = all_metrics_reports()
    metrics['clip_1'].speeds_suppressed_by_gap = 99

    print_comparisons(collect_comparisons(all_mapping_reports(), metrics))

    assert 'A mismatch is a finding' in capsys.readouterr().out


# --- the exit code ---------------------------------------------------------

def test_main_exits_non_zero_when_an_expectation_does_not_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The whole point of the pre-registration: a drifted figure must not be
    # written quietly and reported as success.
    mapping_csv = tmp_path / 'stage8.csv'
    monkeypatch.setattr(measure, 'MAPPING_CSV', mapping_csv)
    monkeypatch.setattr(measure, 'METRICS_CSV', tmp_path / 'stage9.csv')
    monkeypatch.setattr(measure, 'load_config', lambda path: {'metrics': {'speed_window': 5}})

    specs = {'clip_1': CLIP_1, 'clip_2': CLIP_2, 'clip_3': CLIP_3}

    def drifted(clip: str, speed_window: int) -> tuple[MappingReport, MetricsReport]:
        report = metrics_report(specs[clip])
        if clip == 'clip_1':
            report.speeds_computed -= 1
        return mapping_report(specs[clip]), report

    monkeypatch.setattr(measure, 'measure_clip', drifted)

    assert measure.main() == 1
    # Written anyway, deliberately: a mismatched run is exactly the one whose
    # numbers someone needs to open and compare. The non-zero exit is what
    # stops it being mistaken for a clean result, not the absence of a file.
    assert mapping_csv.exists()


def test_main_exits_zero_and_writes_both_csvs_when_everything_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mapping_csv = tmp_path / 'stage8.csv'
    metrics_csv = tmp_path / 'stage9.csv'
    monkeypatch.setattr(measure, 'MAPPING_CSV', mapping_csv)
    monkeypatch.setattr(measure, 'METRICS_CSV', metrics_csv)
    monkeypatch.setattr(measure, 'load_config', lambda path: {'metrics': {'speed_window': 5}})

    specs = {'clip_1': CLIP_1, 'clip_2': CLIP_2, 'clip_3': CLIP_3}
    monkeypatch.setattr(
        measure, 'measure_clip',
        lambda clip, speed_window: (mapping_report(specs[clip]), metrics_report(specs[clip])),
    )

    assert measure.main() == 0
    for path, headers in ((mapping_csv, MAPPING_HEADERS), (metrics_csv, METRICS_HEADERS)):
        rows = list(csv.reader(path.read_text(encoding='utf-8').splitlines()))
        assert rows[0] == headers
        assert len(rows) == 5
        assert rows[-1][0] == TOTAL_LABEL


def test_main_raises_rather_than_writing_when_a_clip_does_not_reconcile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(measure, 'MAPPING_CSV', tmp_path / 'stage8.csv')
    monkeypatch.setattr(measure, 'METRICS_CSV', tmp_path / 'stage9.csv')
    monkeypatch.setattr(measure, 'load_config', lambda path: {'metrics': {'speed_window': 5}})
    monkeypatch.setattr(
        measure, 'measure_clip',
        lambda clip, speed_window: (
            MappingReport(n_frames=117, mapped_frames=100, insufficient_keypoints=5),
            metrics_report(CLIP_1),
        ),
    )

    with pytest.raises(ValueError, match='does not reconcile'):
        measure.main()

    assert not (tmp_path / 'stage8.csv').exists(), 'nothing may be written from an unreconciled run'


def test_the_speed_window_reaches_player_metrics_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Read from config rather than defaulted, matching main.py: the committed
    # value equals PlayerMetrics' own default, so only a non-default proves it.
    monkeypatch.setattr(measure, 'MAPPING_CSV', tmp_path / 'stage8.csv')
    monkeypatch.setattr(measure, 'METRICS_CSV', tmp_path / 'stage9.csv')
    monkeypatch.setattr(measure, 'load_config', lambda path: {'metrics': {'speed_window': 11}})

    seen: list[int] = []
    specs = {'clip_1': CLIP_1, 'clip_2': CLIP_2, 'clip_3': CLIP_3}

    def record(clip: str, speed_window: int) -> tuple[MappingReport, MetricsReport]:
        seen.append(speed_window)
        return mapping_report(specs[clip]), metrics_report(specs[clip])

    monkeypatch.setattr(measure, 'measure_clip', record)
    measure.main()

    assert seen == [11, 11, 11]
