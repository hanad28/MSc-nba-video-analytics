"""Unit tests for BallInterpolator: detection gates, diagnostic traces, and gap filling."""

from __future__ import annotations

import pytest

from basketball.detection.ball_detector import BallDetection
from basketball.detection.ball_interpolation import (
    MAX_BALL_TRAVEL,
    BallInterpolator,
)


@pytest.fixture
def interpolator() -> BallInterpolator:
    return BallInterpolator()


def detection(
    cx: float,
    cy: float,
    size: float = 4.0,
    confidence: float = 0.9,
) -> dict[int, BallDetection]:
    half = size / 2.0
    return {1: BallDetection(bbox=[cx - half, cy - half, cx + half, cy + half], confidence=confidence)}


def centres(frames: list[dict[int, BallDetection]]) -> list[float | None]:
    """Return the ball centre x per frame, or None where the frame carries no detection."""
    out: list[float | None] = []
    for frame in frames:
        if 1 in frame:
            x1, _, x2, _ = frame[1].bbox
            out.append((x1 + x2) / 2.0)
        else:
            out.append(None)
    return out


def present(frames: list[dict[int, BallDetection]]) -> list[bool]:
    return [1 in frame for frame in frames]


# --- filter_detections (frozen last-accepted-point gate) --------------------

def test_filter_detections_keeps_a_plausible_track(interpolator):
    frames = [detection(x, 0) for x in (0, 10, 20, 30)]
    assert present(interpolator.filter_detections(frames)) == [True] * 4


def test_filter_detections_rejects_a_teleport(interpolator):
    frames = [detection(0, 0), detection(10, 0), detection(1000, 0), detection(20, 0)]
    assert present(interpolator.filter_detections(frames)) == [True, True, False, True]


def test_filter_detections_tolerance_grows_with_the_gap(interpolator):
    # 40px in one frame is rejected; the same jump across two frames is allowed.
    one_frame = [detection(0, 0), detection(1.5 * MAX_BALL_TRAVEL, 0)]
    assert present(interpolator.filter_detections(one_frame)) == [True, False]

    two_frames = [detection(0, 0), {}, detection(1.5 * MAX_BALL_TRAVEL, 0)]
    assert present(interpolator.filter_detections(two_frames)) == [True, False, True]


def test_filter_detections_does_not_mutate_its_input(interpolator):
    frames = [detection(0, 0), detection(1000, 0)]
    interpolator.filter_detections(frames)
    assert present(frames) == [True, True]


def test_filter_detections_handles_empty_input(interpolator):
    assert interpolator.filter_detections([]) == []
    assert interpolator.filter_detections([{}, {}]) == [{}, {}]


# --- filter_detections_predicted (velocity extrapolation gate) --------------

def test_filter_detections_predicted_seeds_then_follows_constant_velocity(interpolator):
    frames = [detection(x, 0) for x in (0, 20, 40, 60, 80)]
    assert present(interpolator.filter_detections_predicted(frames)) == [True] * 5


def test_filter_detections_predicted_rejects_a_detection_off_the_prediction(interpolator):
    frames = [detection(0, 0), detection(20, 0), detection(40, 0), detection(2000, 0)]
    assert present(interpolator.filter_detections_predicted(frames)) == [True, True, True, False]


def test_filter_detections_predicted_bootstrap_uses_the_frozen_point_gate(interpolator):
    """With only one anchor, the frozen-point fallback applies rather than the looser tolerance."""
    frames = [detection(0, 0), detection(35, 0)]
    assert present(interpolator.filter_detections_predicted(frames)) == [True, False]


def test_filter_detections_predicted_falls_back_when_anchors_are_too_sparse(interpolator):
    frames = [{} for _ in range(30)]
    frames[0] = detection(0, 0)
    frames[20] = detection(10, 0)     # accepted: within MAX_BALL_TRAVEL * 20
    frames[21] = detection(2000, 0)   # anchor span exceeded -> frozen-point gate -> rejected
    assert present(interpolator.filter_detections_predicted(frames))[21] is False


# --- _predict_position -----------------------------------------------------

def test_predict_position_needs_two_anchors(interpolator):
    assert interpolator._predict_position([(0, 0.0, 0.0)], 1, 8) is None


def test_predict_position_rejects_a_too_wide_anchor_span(interpolator):
    anchors = [(0, 0.0, 0.0), (20, 20.0, 0.0)]
    assert interpolator._predict_position(anchors, 1, 8) is None


def test_predict_position_extrapolates_the_fitted_velocity(interpolator):
    anchors = [(0, 0.0, 0.0), (1, 10.0, 5.0), (2, 20.0, 10.0)]
    px, py = interpolator._predict_position(anchors, 2, 8)
    assert px == pytest.approx(40.0)
    assert py == pytest.approx(20.0)


def test_predict_position_explained_reports_the_path(interpolator):
    _, path = interpolator._predict_position_explained([(0, 0.0, 0.0)], 1, 8)
    assert path == 'fallback: fewer than 2 anchors'

    _, path = interpolator._predict_position_explained([(0, 0.0, 0.0), (20, 1.0, 1.0)], 1, 8)
    assert path == 'fallback: anchor span exceeded'

    _, path = interpolator._predict_position_explained([(0, 0.0, 0.0), (1, 1.0, 1.0)], 1, 8)
    assert path == 'predicted'


def test_trace_predicted_decisions_records_every_detection_frame(interpolator):
    frames = [detection(0, 0), detection(10, 0), {}, detection(2000, 0)]
    records = interpolator.trace_predicted_decisions(frames)

    assert sorted(records) == [0, 1, 3]
    assert records[0]['path'] == 'seed: first anchor'
    assert records[1]['accepted'] is True
    assert records[3]['accepted'] is False
    assert records[3]['frame_gap'] == 2
    assert records[3]['distance'] > records[3]['tolerance']


# --- filter_detections_staleness_aware -------------------------------------

def test_staleness_aware_matches_the_frozen_gate_for_fresh_anchors(interpolator):
    frames = [detection(0, 0), detection(1000, 0), detection(10, 0)]
    assert present(interpolator.filter_detections_staleness_aware(frames)) == [True, False, True]


def test_staleness_aware_accepts_outright_once_the_anchor_is_stale(interpolator):
    frames = [{} for _ in range(12)]
    frames[0] = detection(0, 0)
    frames[10] = detection(5000, 0)  # gap 10 > threshold 8 -> unconditional accept
    assert present(interpolator.filter_detections_staleness_aware(frames))[10] is True


def test_trace_staleness_aware_decisions_paths(interpolator):
    frames = [{} for _ in range(12)]
    frames[0] = detection(0, 0)
    frames[1] = detection(1000, 0)
    frames[10] = detection(5000, 0)

    records = interpolator.trace_staleness_aware_decisions(frames)

    assert records[0]['path'] == 'seed: first accepted detection'
    assert records[1]['path'] == 'frozen-point check'
    assert records[1]['accepted'] is False
    assert records[10]['accepted'] is True
    assert records[10]['path'].startswith('stale-reset accept')
    assert records[10]['frame_gap'] == 10


# --- is_motion_consistent --------------------------------------------------

def test_motion_consistency_is_inconclusive_without_two_anchors(interpolator):
    assert interpolator.is_motion_consistent([], (1, 0.0, 0.0)) is True
    assert interpolator.is_motion_consistent([(0, 0.0, 0.0)], (1, 500.0, 0.0)) is True


def test_motion_consistency_is_inconclusive_for_degenerate_gaps(interpolator):
    history = [(0, 0.0, 0.0), (0, 10.0, 0.0)]
    assert interpolator.is_motion_consistent(history, (1, 10.0, 0.0)) is True

    history = [(0, 0.0, 0.0), (2, 10.0, 0.0)]
    assert interpolator.is_motion_consistent(history, (2, 10.0, 0.0)) is True


def test_motion_consistency_accepts_continued_motion(interpolator):
    history = [(0, 0.0, 0.0), (1, 10.0, 0.0), (2, 20.0, 0.0)]
    assert interpolator.is_motion_consistent(history, (3, 30.0, 0.0)) is True


def test_motion_consistency_rejects_an_implausible_speed(interpolator):
    history = [(0, 0.0, 0.0), (1, 10.0, 0.0), (2, 20.0, 0.0)]
    assert interpolator.is_motion_consistent(history, (3, 1000.0, 0.0)) is False


def test_motion_consistency_rejects_a_direction_reversal(interpolator):
    history = [(0, 0.0, 0.0), (1, 20.0, 0.0), (2, 40.0, 0.0)]
    # Backwards 30px in one frame: within the speed cap, but ~180 deg off.
    assert interpolator.is_motion_consistent(history, (3, 10.0, 0.0)) is False


def test_motion_consistency_ignores_direction_below_the_reference_speed(interpolator):
    history = [(0, 0.0, 0.0), (1, 1.0, 0.0), (2, 2.0, 0.0)]
    assert interpolator.is_motion_consistent(history, (3, 0.0, 0.0)) is True


def test_filter_detections_motion_consistency_drops_the_outlier(interpolator):
    frames = [detection(x, 0) for x in (0, 20, 40, 5000, 60)]
    assert present(interpolator.filter_detections_motion_consistency(frames)) == [
        True, True, True, False, True,
    ]


def test_trace_motion_consistency_decisions_records_reasons(interpolator):
    frames = [detection(x, 0) for x in (0, 20, 40, 5000)]
    records = interpolator.trace_motion_consistency_decisions(frames)

    assert records[0]['reason'].startswith('inconclusive')
    assert records[2]['reason'] == 'consistent'
    assert records[3]['accepted'] is False
    assert 'implied speed' in records[3]['reason']
    assert records[3]['anchors_before'] == 3


# --- filter_detections_combined -------------------------------------------

def test_combined_gate_base_layer_rejects_a_teleport(interpolator):
    frames = [detection(0, 0), detection(5000, 0), detection(10, 0)]
    assert present(interpolator.filter_detections_combined(frames)) == [True, False, True]


def test_combined_gate_screen_rejects_a_reversal_within_the_base_tolerance(interpolator):
    frames = [detection(x, 0) for x in (0, 20, 40, 20)]
    assert present(interpolator.filter_detections_combined(frames)) == [True, True, True, False]


def test_combined_gate_skips_the_screen_when_the_reference_is_not_fresh(interpolator):
    """A candidate 10 frames after the last anchor is accepted on the base check alone."""
    frames = [{} for _ in range(14)]
    frames[0] = detection(0, 0)
    frames[1] = detection(20, 0)
    frames[2] = detection(40, 0)
    frames[12] = detection(20, 0)  # reversal, but frame_gap 10 > 8 -> screen skipped
    assert present(interpolator.filter_detections_combined(frames))[12] is True


def test_trace_combined_decisions_records_each_layer(interpolator):
    frames = [{} for _ in range(14)]
    frames[0] = detection(0, 0)
    frames[1] = detection(20, 0)
    frames[2] = detection(40, 0)
    frames[3] = detection(20, 0)     # screen-inconsistent
    frames[4] = detection(5000, 0)   # base-reject
    frames[13] = detection(30, 0)    # screen skipped (stale gap)

    records = interpolator.trace_combined_decisions(frames)

    assert records[0]['path'] == 'seed: first accepted detection'
    assert records[2]['path'] == 'screen-consistent'
    assert records[3]['path'] == 'screen-inconsistent'
    assert records[3]['motion_check_applied'] is True
    assert records[4]['path'].startswith('base-reject')
    assert records[4]['accepted'] is False
    assert records[13]['fresh'] is False
    assert records[13]['accepted'] is True
    assert 'frame_gap' in records[13]['path']


def test_trace_combined_decisions_reports_a_short_reference_window(interpolator):
    frames = [detection(0, 0), detection(10, 0)]
    records = interpolator.trace_combined_decisions(frames)
    assert 'fewer than 2 accepted anchors' in records[1]['path']


# --- global trajectory solver ---------------------------------------------

def test_global_trajectory_passes_through_short_clips(interpolator):
    assert interpolator.filter_detections_global_trajectory([]) == []
    single = [{}, detection(0, 0)]
    assert present(interpolator.filter_detections_global_trajectory(single)) == [False, True]


def test_global_trajectory_keeps_a_straight_flight(interpolator):
    frames = [detection(x, 0) for x in (0, 20, 40, 60, 80)]
    assert present(interpolator.filter_detections_global_trajectory(frames)) == [True] * 5


def test_global_trajectory_drops_an_off_path_false_positive(interpolator):
    frames = [detection(0, 0), detection(20, 0), detection(40, 900), detection(40, 0), detection(60, 0)]
    kept = present(interpolator.filter_detections_global_trajectory(frames))
    assert kept[2] is False
    assert kept[0] and kept[1] and kept[3] and kept[4]


def test_global_trajectory_forbids_edges_above_the_speed_cap(interpolator):
    frames = [detection(0, 0), detection(20, 0), detection(10_000, 0)]
    kept = present(interpolator.filter_detections_global_trajectory(frames))
    assert kept[2] is False


def test_trajectory_nodes_collects_frame_indices_and_centres(interpolator):
    frames = [detection(5, 7), {}, detection(9, 11)]
    assert interpolator._trajectory_nodes(frames) == [(0, 5.0, 7.0), (2, 9.0, 11.0)]


def test_trajectory_path_indices_walks_parents_in_ascending_order(interpolator):
    assert interpolator._trajectory_path_indices([-1, 0, 1], 2) == [0, 1, 2]


def test_solve_global_trajectory_charges_no_motion_cost_after_a_long_silence(interpolator):
    nodes = [(0, 0.0, 0.0), (100, 5000.0, 0.0)]
    parent, best, end_index = interpolator._solve_global_trajectory(
        nodes, max_speed=100.0, skip_penalty=1000.0, max_edge_gap=45,
    )
    assert parent[1] == 0
    assert best[1] == 0.0
    assert end_index == 1


def test_trace_global_trajectory_decisions_empty_and_single(interpolator):
    assert interpolator.trace_global_trajectory_decisions([{}, {}]) == {}

    records = interpolator.trace_global_trajectory_decisions([{}, detection(0, 0)])
    assert records[1]['accepted'] is True
    assert records[1]['edge_kind'] == 'path start'


def test_trace_global_trajectory_decisions_reports_edges_and_skips(interpolator):
    frames = [detection(0, 0), detection(20, 0), detection(40, 900), detection(40, 0)]
    records = interpolator.trace_global_trajectory_decisions(frames)

    assert records[0]['edge_kind'] == 'path start'
    assert records[0]['previous_accepted_frame'] is None
    assert records[1]['edge_kind'] == 'motion edge'
    assert records[1]['previous_accepted_frame'] == 0
    assert records[1]['edge_gap'] == 1
    assert records[1]['edge_speed'] == pytest.approx(20.0)
    assert records[1]['edge_motion_cost'] == pytest.approx(400.0)
    assert records[2]['accepted'] is False
    assert records[2]['reason'].startswith('skipped')
    assert records[3]['accepted'] is True


def test_trace_global_trajectory_decisions_flags_a_long_silence_edge(interpolator):
    frames = [{} for _ in range(60)]
    frames[0] = detection(0, 0)
    frames[50] = detection(4000, 0)

    records = interpolator.trace_global_trajectory_decisions(frames)

    assert records[50]['edge_kind'] == 'long-silence reset (no motion cost charged)'
    assert records[50]['edge_motion_cost'] == 0.0
    assert records[50]['edge_gap'] == 50


# --- fill_missing ---------------------------------------------------------

def test_fill_missing_interpolates_an_interior_gap(interpolator):
    frames = [detection(0, 0), {}, detection(20, 0)]
    filled = interpolator.fill_missing(frames)

    assert centres(filled) == [0.0, 10.0, 20.0]
    assert filled[1][1].confidence == 0.0


def test_fill_missing_preserves_genuine_detections(interpolator):
    frames = [detection(0, 0, confidence=0.7), {}, detection(20, 0, confidence=0.8)]
    filled = interpolator.fill_missing(frames)

    assert filled[0] is frames[0]
    assert filled[2] is frames[2]


def test_fill_missing_backfills_and_forward_fills_the_edges(interpolator):
    frames = [{}, detection(10, 0), {}]
    filled = interpolator.fill_missing(frames)

    assert centres(filled) == [10.0, 10.0, 10.0]
    assert filled[0][1].confidence == 0.0
    assert filled[2][1].confidence == 0.0


def test_fill_missing_leaves_a_fully_empty_clip_empty(interpolator):
    assert interpolator.fill_missing([{}, {}, {}]) == [{}, {}, {}]


def test_fill_missing_handles_empty_input(interpolator):
    # Guards the empty-clip path through Pandas: a zero-row frame has object
    # dtype columns, so a stricter interpolate() would raise here rather than
    # returning an empty result.
    assert interpolator.fill_missing([]) == []
