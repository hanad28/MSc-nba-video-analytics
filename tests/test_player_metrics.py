"""Unit tests for Stage 9's PlayerMetrics (basketball/metrics/player_metrics.py)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from basketball.metrics.player_metrics import MetricsReport, PlayerMetrics, TrackSpeed

FPS = 30.0
SPEED_WINDOW = 5

# The measured gap distribution across all three clips, pooled over 3,841
# displacements: gap 1 is 97.3%, gaps 1-5 cover 99.5%, and above 5 the counts
# fall to single digits. The 46-frame gaps are clip_3's contiguous unmapped
# run, where the camera panned to midcourt during a fast break.
CLIP_3_RUN_GAP = 46


def metrics(speed_window: int = SPEED_WINDOW) -> PlayerMetrics:
    """A PlayerMetrics at the measured 30 fps and the configured speed window."""
    return PlayerMetrics(fps=FPS, speed_window=speed_window)


def straight_line(n_frames: int, metres_per_frame: float = 1.0) -> list[dict[int, tuple[float, float]]]:
    """A single track moving a fixed distance along x on every consecutive frame, the 97.3% case."""
    return [{1: (index * metres_per_frame, 0.0)} for index in range(n_frames)]


def seen_at(frames_and_positions: dict[int, tuple[float, float]], n_frames: int) -> list[dict[int, tuple[float, float]]]:
    """A clip of the given length where track 1 appears only on the listed frames, every other frame being an empty dict."""
    # An unmapped frame yields an EMPTY DICT, never None; that is what
    # CourtMapper produces for a frame with no homography, and roughly 10% of
    # frames are in that state.
    return [
        {1: frames_and_positions[index]} if index in frames_and_positions else {}
        for index in range(n_frames)
    ]


# --- distance ------------------------------------------------------------

def test_a_one_metre_displacement_records_one_metre():
    distances, _, report = metrics().compute(straight_line(2))

    assert distances[1][1] == pytest.approx(1.0)
    assert report.total_distance_m[1] == pytest.approx(1.0)


def test_distance_is_euclidean_not_axis_wise():
    # A 3-4-5 triangle: an implementation summing the axes would report 7.
    positions = [{1: (0.0, 0.0)}, {1: (3.0, 4.0)}]

    distances, _, _ = metrics().compute(positions)

    assert distances[1][1] == pytest.approx(5.0)


def test_distance_is_cumulative_across_frames():
    distances, _, report = metrics().compute(straight_line(5))

    assert [distances[index][1] for index in range(5)] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert report.total_distance_m[1] == pytest.approx(4.0)


def test_a_first_sighting_records_zero_distance_not_absence():
    # The track exists and has covered nothing yet, which is a real zero
    # rather than an unmeasurable quantity.
    distances, _, _ = metrics().compute(straight_line(2))

    assert distances[0][1] == pytest.approx(0.0)


def test_cumulative_distance_includes_gap_spanning_displacements():
    # The player did travel that distance; only the RATE is uncertain.
    # Excluding it would understate distance covered for no good reason.
    positions = seen_at({0: (0.0, 0.0), CLIP_3_RUN_GAP: (20.0, 0.0)}, CLIP_3_RUN_GAP + 1)

    distances, speeds, report = metrics().compute(positions)

    assert distances[CLIP_3_RUN_GAP][1] == pytest.approx(20.0)
    assert report.total_distance_m[1] == pytest.approx(20.0)
    # And that same displacement contributed no speed.
    assert 1 not in speeds[CLIP_3_RUN_GAP]


def test_a_track_appearing_partway_through_starts_from_its_first_sighting():
    positions = seen_at({3: (0.0, 0.0), 4: (1.0, 0.0)}, 6)

    distances, _, report = metrics().compute(positions)

    assert 1 not in distances[0]
    assert distances[3][1] == pytest.approx(0.0)
    assert distances[4][1] == pytest.approx(1.0)
    assert report.total_distance_m[1] == pytest.approx(1.0)


# --- speed ---------------------------------------------------------------

def test_five_metres_over_five_frames_at_thirty_fps_is_thirty_metres_per_second():
    _, speeds, _ = metrics().compute(straight_line(6))

    assert speeds[5][1].speed_ms == pytest.approx(30.0)


def test_the_denominator_is_elapsed_time_not_the_sighting_count():
    # The central arithmetic error to avoid. A track seen ONLY at frames 0 and
    # 5 covered 5 m over 5 frames of elapsed time, so 30 m/s. Dividing by
    # frames_present / fps (one sighting) would give 150 m/s, fivefold
    # higher, and reads perfectly plausibly in code.
    positions = seen_at({0: (0.0, 0.0), 5: (5.0, 0.0)}, 6)

    _, speeds, _ = metrics().compute(positions)

    assert speeds[5][1].speed_ms == pytest.approx(30.0)
    assert speeds[5][1].speed_ms != pytest.approx(150.0)


def test_a_sparsely_seen_track_is_not_inflated_relative_to_a_densely_seen_one():
    # Two tracks covering the same ground over the same elapsed time, one seen
    # every frame and one seen twice, must report the same speed.
    dense, _, _ = metrics().compute(straight_line(6))
    sparse_positions = seen_at({0: (0.0, 0.0), 5: (5.0, 0.0)}, 6)
    _, sparse_speeds, _ = metrics().compute(sparse_positions)
    _, dense_speeds, _ = metrics().compute(straight_line(6))

    assert dense_speeds[5][1].speed_ms == pytest.approx(sparse_speeds[5][1].speed_ms)


def test_speed_is_reported_in_metres_per_second_not_kilometres_per_hour():
    # 1 m per frame at 30 fps is 30 m/s, which would be 108 in km/h.
    _, speeds, _ = metrics().compute(straight_line(3))

    assert speeds[2][1].speed_ms == pytest.approx(30.0)
    assert speeds[2][1].speed_ms != pytest.approx(108.0)


def test_no_correction_factor_is_applied_to_the_measurement():
    # The reference multiplies every distance by 0.4 with no measurement
    # behind it, most likely compensating for its inflated denominator. A
    # 1 m displacement must record exactly 1 m.
    distances, speeds, _ = metrics().compute(straight_line(2))

    assert distances[1][1] == pytest.approx(1.0)
    assert distances[1][1] != pytest.approx(0.4)
    assert speeds[1][1].speed_ms == pytest.approx(30.0)


def test_a_stationary_player_records_zero_speed_and_zero_distance():
    positions = [{1: (5.0, 5.0)} for _ in range(4)]

    distances, speeds, _ = metrics().compute(positions)

    assert distances[3][1] == pytest.approx(0.0)
    # Genuinely zero, and PRESENT: the state the gap rule's absence must be
    # distinguishable from.
    assert speeds[3][1].speed_ms == pytest.approx(0.0)


# --- the gap rule --------------------------------------------------------

@pytest.mark.parametrize('gap', [1, 2, 3, 4, 5])
def test_a_gap_within_the_threshold_yields_a_speed(gap):
    # Gaps of 1 to 5 frames cover 99.5% of measured displacements.
    positions = seen_at({0: (0.0, 0.0), gap: (float(gap), 0.0)}, gap + 1)

    _, speeds, report = metrics().compute(positions)

    assert 1 in speeds[gap]
    assert speeds[gap][1].speed_ms == pytest.approx(FPS)
    assert report.speeds_suppressed_by_gap == 0


@pytest.mark.parametrize('gap', [6, 7, 19, CLIP_3_RUN_GAP])
def test_a_gap_beyond_the_threshold_yields_no_speed_entry(gap):
    # Above 5 the measured counts fall to single digits and scatter: 6, 7, 8,
    # 9, 10, 11, 16, 17, 19, then 46 with nothing between. A speed averaged
    # over 1.53 seconds of a fast break is a different quantity.
    positions = seen_at({0: (0.0, 0.0), gap: (float(gap), 0.0)}, gap + 1)

    _, speeds, report = metrics().compute(positions)

    assert 1 not in speeds[gap]
    assert report.speeds_suppressed_by_gap == 1


def test_a_suppressed_speed_is_absent_rather_than_zero():
    # Absence and zero are different measurements. A zero here would be
    # indistinguishable from a stationary player, which is a real state this
    # class reports elsewhere.
    positions = seen_at({0: (0.0, 0.0), 6: (6.0, 0.0)}, 7)

    _, speeds, _ = metrics().compute(positions)

    assert 1 not in speeds[6]
    assert speeds[6].get(1) is None
    assert speeds[6] == {}


def test_the_threshold_boundary_falls_between_five_and_six():
    within = seen_at({0: (0.0, 0.0), 5: (5.0, 0.0)}, 6)
    beyond = seen_at({0: (0.0, 0.0), 6: (6.0, 0.0)}, 7)

    _, within_speeds, _ = metrics().compute(within)
    _, beyond_speeds, _ = metrics().compute(beyond)

    assert 1 in within_speeds[5]
    assert 1 not in beyond_speeds[6]
    assert PlayerMetrics.MAX_SPEED_GAP == 5


def test_the_gap_threshold_is_a_class_constant_not_a_constructor_argument():
    # A reasoned choice from a measured distribution rather than a swept
    # parameter; a config key or constructor argument would imply it had
    # been tuned.

    assert 'max_speed_gap' not in inspect.signature(PlayerMetrics.__init__).parameters


# --- the gap carried alongside each speed --------------------------------

@pytest.mark.parametrize('gap', [1, 2, 5])
def test_the_gap_length_is_carried_with_the_speed(gap):
    # A speed derived across a 4-frame gap is a different measurement from one
    # across a single frame, and a consumer that cannot tell them apart reads
    # both as instantaneous.
    positions = seen_at({0: (0.0, 0.0), gap: (float(gap), 0.0)}, gap + 1)

    _, speeds, _ = metrics().compute(positions)

    assert speeds[gap][1].last_gap_frames == gap


def test_a_suppressed_gap_discards_the_tracks_earlier_history():
    # The reported case, at speed_window = 10. A track seen at frames 0-3,
    # absent 4-9, then seen at 10 and 11: without discarding, the window at
    # frame 11 spans frames 2-11 and still holds the pre-gap displacements
    # while excluding the 17 m covered across the suppressed interval. That
    # reported 9.0 m/s against a true 30.0, diluted by whatever fraction of
    # the window the gap occupies.
    positions = (
        [{1: (float(index), 0.0)} for index in range(4)]
        + [{} for _ in range(6)]
        + [{1: (20.0, 0.0)}, {1: (21.0, 0.0)}]
    )

    _, speeds, _ = metrics(speed_window=10).compute(positions)

    assert speeds[11][1].speed_ms == pytest.approx(FPS)
    assert speeds[11][1].speed_ms != pytest.approx(9.0)


@pytest.mark.parametrize('window', [5, 10, 20])
def test_the_discard_holds_at_any_speed_window(window):
    # Structural rather than arithmetic: at the default 5 a suppressed gap of
    # 6+ frames already pushes the older entries out, so the defect is
    # unreachable there, but speed_window is a config key, and the fix must
    # not depend on the relationship between two constants.
    positions = (
        [{1: (float(index), 0.0)} for index in range(4)]
        + [{} for _ in range(6)]
        + [{1: (20.0, 0.0)}, {1: (21.0, 0.0)}]
    )

    _, speeds, _ = metrics(speed_window=window).compute(positions)

    assert speeds[11][1].speed_ms == pytest.approx(FPS)


def test_a_suppressed_gap_does_not_discard_another_tracks_history():
    # The discard is per track: one player's lost sighting must not reset
    # another's window.
    #
    # Track 1 is inserted FIRST on the suppression frame so its discard runs
    # before track 2 is computed; a clear-everything bug is otherwise
    # invisible, since dict iteration is insertion-ordered. And the two are
    # distinguished by window_frames rather than by speed: at constant
    # velocity a track averaging over 5 frames and one averaging over its last
    # hop both report 30 m/s, so asserting the speed alone cannot tell them
    # apart. This exact shape let a history.clear() mutation survive.
    positions = []
    for index in range(8):
        frame: dict[int, tuple[float, float]] = {}
        if index == 0:
            frame[1] = (0.0, 0.0)
        if index == 7:
            frame[1] = (50.0, 0.0)
        frame[2] = (float(index), 5.0)
        positions.append(frame)

    _, speeds, report = metrics().compute(positions)

    assert 1 not in speeds[7], 'the gap-spanning track loses its speed'
    assert report.speeds_suppressed_by_gap == 1
    assert speeds[7][2].speed_ms == pytest.approx(FPS), 'the other track is unaffected'
    assert speeds[7][2].window_frames == SPEED_WINDOW, (
        'the other track must keep its full window, not be reset to its last hop'
    )


# --- what the two spans mean ---------------------------------------------

def test_window_frames_reports_the_span_the_speed_was_averaged_over():
    # A densely-seen track's speed is a mean over the whole window, not over
    # the last hop. Reporting only the last gap would understate the span.
    _, speeds, _ = metrics().compute(straight_line(6))

    entry = speeds[5][1]
    assert entry.window_frames == 5
    assert entry.last_gap_frames == 1


def test_the_two_spans_differ_when_a_track_is_seen_densely_then_sparsely():
    # Seen every frame to 3, then not until 8. The last hop is 5 frames, and
    # the window at frame 8 spans from the earliest displacement still in it.
    positions = (
        [{1: (float(index), 0.0)} for index in range(4)]
        + [{} for _ in range(4)]
        + [{1: (8.0, 0.0)}]
    )

    _, speeds, _ = metrics().compute(positions)

    entry = speeds[8][1]
    assert entry.last_gap_frames == 5
    assert entry.window_frames >= entry.last_gap_frames


def test_a_speed_entry_is_a_track_speed_carrying_both_fields():
    _, speeds, _ = metrics().compute(straight_line(3))

    entry = speeds[2][1]
    assert isinstance(entry, TrackSpeed)
    assert entry.speed_ms > 0.0
    assert entry.last_gap_frames == 1
    assert entry.window_frames >= entry.last_gap_frames


# --- output shape --------------------------------------------------------

def test_output_length_equals_the_frame_count():
    positions = straight_line(4) + [{}, {}]

    distances, speeds, report = metrics().compute(positions)

    assert len(distances) == len(speeds) == len(positions) == 6
    assert report.n_frames == 6


def test_trailing_frames_with_no_positions_yield_empty_dicts():
    positions = straight_line(3) + [{}, {}]

    distances, speeds, _ = metrics().compute(positions)

    assert distances[3] == {} and distances[4] == {}
    assert speeds[3] == {} and speeds[4] == {}


def test_a_clip_of_entirely_unmapped_frames_produces_empty_output():
    # Roughly 10% of frames carry no positions, and on clip_3 that is a
    # contiguous run; a clip could in principle open with one.
    distances, speeds, report = metrics().compute([{} for _ in range(5)])

    assert distances == [{}] * 5
    assert speeds == [{}] * 5
    assert report.total_distance_m == {}
    assert report.displacements_measured == 0


def test_multiple_tracks_are_measured_independently():
    positions = [
        {1: (0.0, 0.0), 2: (10.0, 5.0)},
        {1: (1.0, 0.0), 2: (10.0, 5.0)},
        {1: (2.0, 0.0), 2: (12.0, 5.0)},
    ]

    distances, speeds, report = metrics().compute(positions)

    assert report.total_distance_m[1] == pytest.approx(2.0)
    assert report.total_distance_m[2] == pytest.approx(2.0)
    assert speeds[1][2].speed_ms == pytest.approx(0.0)
    assert distances[2][1] == pytest.approx(2.0)


# --- the report ----------------------------------------------------------

def test_the_report_counts_displacements_speeds_and_suppressions():
    positions = seen_at({0: (0.0, 0.0), 1: (1.0, 0.0), 8: (8.0, 0.0)}, 9)

    _, _, report = metrics().compute(positions)

    assert report.displacements_measured == 2
    assert report.speeds_computed == 1
    assert report.speeds_suppressed_by_gap == 1
    assert 'suppressed by the gap rule' in report.summary()


def test_the_report_totals_distance_per_track():
    positions = [
        {1: (0.0, 0.0), 2: (0.0, 0.0)},
        {1: (3.0, 0.0), 2: (0.0, 4.0)},
    ]

    _, _, report = metrics().compute(positions)

    assert report.total_distance_m == {1: pytest.approx(3.0), 2: pytest.approx(4.0)}


def test_an_empty_report_summarises_without_dividing_by_zero():
    report = MetricsReport(n_frames=0)

    assert 'metrics' in report.summary()


# --- construction guards -------------------------------------------------

@pytest.mark.parametrize('fps', [0.0, -1.0])
def test_a_non_positive_fps_raises(fps):
    with pytest.raises(ValueError, match='fps must be positive'):
        PlayerMetrics(fps=fps)


@pytest.mark.parametrize('window', [0, -1])
def test_a_non_positive_speed_window_raises(window):
    with pytest.raises(ValueError, match='speed_window must be at least 1'):
        PlayerMetrics(fps=FPS, speed_window=window)


def test_the_speed_window_changes_what_the_window_spans():
    # A longer window averages over more of the track's recent motion, so an
    # accelerating player reports a lower speed under it.
    positions = [{1: (0.0, 0.0)}, {1: (1.0, 0.0)}, {1: (3.0, 0.0)}, {1: (6.0, 0.0)}]

    _, narrow, _ = metrics(speed_window=1).compute(positions)
    _, wide, _ = metrics(speed_window=4).compute(positions)

    assert narrow[3][1].speed_ms > wide[3][1].speed_ms


def test_the_configured_speed_window_matches_the_default_config():

    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    assert config['metrics']['speed_window'] == SPEED_WINDOW
