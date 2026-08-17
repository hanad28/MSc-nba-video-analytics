"""Unit tests for the CLEAR MOT evaluator, using synthetic tracks and ground truth only."""

from __future__ import annotations

import pytest

from basketball.detection.player_detector import PlayerTrack
from evaluation.ground_truth import GTAnnotation
from evaluation.mot_metrics import MOTEvaluator, MOTResult


def gt(frame: int, track_id: int, bbox: list[float]) -> GTAnnotation:
    return GTAnnotation(frame=frame, track_id=track_id, bbox=list(bbox))


def track(track_id: int, bbox: list[float]) -> PlayerTrack:
    return PlayerTrack(track_id=track_id, bbox=list(bbox), confidence=0.9)


BOX_A = [0.0, 0.0, 10.0, 10.0]
BOX_B = [50.0, 50.0, 60.0, 60.0]
FAR_BOX = [200.0, 200.0, 210.0, 210.0]


@pytest.fixture
def evaluator() -> MOTEvaluator:
    return MOTEvaluator()


def test_perfect_tracking_scores_mota_and_idf1_of_one(evaluator):
    ground_truth = [gt(f, 1, BOX_A) for f in range(3)] + [gt(f, 2, BOX_B) for f in range(3)]
    tracks = [{1: track(1, BOX_A), 2: track(2, BOX_B)} for _ in range(3)]

    result = evaluator.evaluate(ground_truth, tracks)

    assert result == MOTResult(
        mota=1.0, motp=1.0, idf1=1.0, num_switches=0, num_false_positives=0,
        num_misses=0, num_objects=6, num_matches=6, precision=1.0, recall=1.0,
    )


def test_sparse_ground_truth_ignores_unlabelled_frames_entirely(evaluator):
    ground_truth = [gt(0, 1, BOX_A), gt(10, 1, BOX_A), gt(20, 1, BOX_A)]

    # 100 tracker frames with a detection on every one; 97 of them unlabelled.
    full_tracks = [{1: track(1, BOX_A)} for _ in range(100)]

    # The same tracker output truncated to just the three labelled frames.
    truncated_tracks: list[dict[int, PlayerTrack]] = [{} for _ in range(21)]
    for frame in (0, 10, 20):
        truncated_tracks[frame] = {1: track(1, BOX_A)}

    full_result = evaluator.evaluate(ground_truth, full_tracks)

    # If unlabelled frames were scored as zero-object frames, the 97 extra
    # detections would each count as a false positive and the results diverge.
    assert full_result == evaluator.evaluate(ground_truth, truncated_tracks)
    assert full_result.mota == 1.0
    assert full_result.num_false_positives == 0


def test_a_mid_sequence_id_change_counts_one_switch(evaluator):
    ground_truth = [gt(f, 1, BOX_A) for f in range(6)]
    tracks = [{7: track(7, BOX_A)} for _ in range(3)] + [{8: track(8, BOX_A)} for _ in range(3)]

    result = evaluator.evaluate(ground_truth, tracks)

    assert result.num_switches == 1
    assert result.num_false_positives == 0
    assert result.num_misses == 0
    assert result.mota == pytest.approx(1 - 1 / 6)


def test_an_unmatched_tracker_box_counts_one_false_positive(evaluator):
    ground_truth = [gt(f, 1, BOX_A) for f in range(3)]
    tracks = [{1: track(1, BOX_A)} for _ in range(3)]
    tracks[1] = {1: track(1, BOX_A), 9: track(9, FAR_BOX)}

    result = evaluator.evaluate(ground_truth, tracks)

    assert result.num_false_positives == 1
    assert result.num_misses == 0
    assert result.num_switches == 0
    assert result.mota == pytest.approx(1 - 1 / 3)


def test_a_missing_tracker_box_counts_one_miss(evaluator):
    ground_truth = [gt(f, 1, BOX_A) for f in range(3)]
    tracks = [{1: track(1, BOX_A)}, {}, {1: track(1, BOX_A)}]

    result = evaluator.evaluate(ground_truth, tracks)

    assert result.num_misses == 1
    assert result.num_false_positives == 0
    assert result.num_switches == 0
    assert result.mota == pytest.approx(1 - 1 / 3)


def test_association_flips_across_the_iou_threshold(evaluator):
    ground_truth = [gt(0, 1, [0.0, 0.0, 10.0, 10.0])]

    # IoU 70/130 ≈ 0.538, just above the 0.5 threshold.
    above = evaluator.evaluate(ground_truth, [{5: track(5, [3.0, 0.0, 13.0, 10.0])}])
    # IoU 60/140 ≈ 0.429, just below it.
    below = evaluator.evaluate(ground_truth, [{5: track(5, [4.0, 0.0, 14.0, 10.0])}])

    assert (above.num_matches, above.num_misses, above.num_false_positives) == (1, 0, 0)
    assert (below.num_matches, below.num_misses, below.num_false_positives) == (0, 1, 1)
    assert above.motp == pytest.approx(7 / 13)  # overlap convention, not 1 - IoU


def test_pooling_accumulates_events_rather_than_averaging_per_clip_metrics(evaluator):
    # Clip A: 4 objects, all matched (MOTA 1.0). Clip B: 1 object, missed (MOTA 0.0).
    # A per-clip average would report 0.5; event-level pooling must report
    # 1 - 1/5 = 0.8, weighting clips by their object counts.
    gt_a = [gt(f, 1, BOX_A) for f in (0, 10, 20, 30)]
    tracks_a = [{1: track(1, BOX_A)} for _ in range(31)]
    gt_b = [gt(0, 1, BOX_A)]
    tracks_b = [{}]

    pooled = evaluator.evaluate_pooled([(gt_a, tracks_a), (gt_b, tracks_b)])

    assert pooled.mota == pytest.approx(0.8)
    assert pooled.num_objects == 5
    assert pooled.num_matches == 4
    assert pooled.num_misses == 1
    assert pooled.recall == pytest.approx(4 / 5)
    # The matchless clip B contributes motp = NaN to motmetrics' own OVERALL
    # aggregation (NaN * 0 weight), which used to poison the pooled MOTP into
    # a masked 0.0; it must equal the perfect clip's MOTP instead.
    assert pooled.motp == 1.0


def test_pooled_motp_is_a_matched_pair_weighted_mean_skipping_matchless_clips(evaluator):
    gt_a = [gt(f, 1, BOX_A) for f in range(4)]
    tracks_a = [{1: track(1, BOX_A)} for _ in range(4)]              # 4 matches at overlap 1.0
    gt_b = [gt(0, 1, [0.0, 0.0, 10.0, 10.0])]
    tracks_b = [{5: track(5, [3.0, 0.0, 13.0, 10.0])}]               # 1 match at overlap 7/13
    gt_c = [gt(0, 1, BOX_B)]
    tracks_c: list[dict] = [{}]                                      # matchless: contributes no weight

    pooled = evaluator.evaluate_pooled([(gt_a, tracks_a), (gt_b, tracks_b), (gt_c, tracks_c)])

    assert pooled.motp == pytest.approx((4 * 1.0 + 7 / 13) / 5)


def test_pooled_motp_is_zero_when_no_sequence_has_any_matches(evaluator):
    gt_a = [gt(0, 1, BOX_A)]
    gt_b = [gt(0, 2, BOX_B)]

    pooled = evaluator.evaluate_pooled([(gt_a, [{}]), (gt_b, [{}])])

    assert pooled.motp == 0.0    # genuinely no matched pairs anywhere; 0.0 is correct here
    assert pooled.num_matches == 0
    assert pooled.num_misses == 2


def test_pooling_a_single_sequence_equals_evaluating_it_directly(evaluator):
    ground_truth = [gt(f, 1, BOX_A) for f in range(3)]
    tracks = [{1: track(1, BOX_A)} for _ in range(3)]

    assert evaluator.evaluate_pooled([(ground_truth, tracks)]) == evaluator.evaluate(ground_truth, tracks)


def test_pooling_nothing_returns_a_zeroed_result(evaluator):
    zero = MOTResult(
        mota=0.0, motp=0.0, idf1=0.0, num_switches=0, num_false_positives=0,
        num_misses=0, num_objects=0, num_matches=0, precision=0.0, recall=0.0,
    )

    assert evaluator.evaluate_pooled([]) == zero
    assert evaluator.evaluate_pooled([([], [{1: track(1, BOX_A)}])]) == zero


def test_switch_frames_names_the_frames_where_switches_are_scored(evaluator):
    ground_truth = [gt(f, 1, BOX_A) for f in range(6)]
    tracks = [{7: track(7, BOX_A)} for _ in range(3)] + [{8: track(8, BOX_A)} for _ in range(3)]

    assert evaluator.switch_frames(ground_truth, tracks) == [3]
    assert evaluator.switch_frames([], tracks) == []


def test_empty_ground_truth_returns_a_zeroed_result(evaluator):
    tracks = [{1: track(1, BOX_A)} for _ in range(5)]

    assert evaluator.evaluate([], tracks) == MOTResult(
        mota=0.0, motp=0.0, idf1=0.0, num_switches=0, num_false_positives=0,
        num_misses=0, num_objects=0, num_matches=0, precision=0.0, recall=0.0,
    )


def test_a_tracker_that_produced_nothing_scores_all_misses_without_nan(evaluator):
    ground_truth = [gt(f, 1, BOX_A) for f in range(3)]

    # A full-length run of empty frames is a tracker that found nothing,
    # distinct from a truncated tracks list, which raises (see below).
    # Dataclass equality doubles as a NaN check: NaN never compares equal.
    assert evaluator.evaluate(ground_truth, [{}, {}, {}]) == MOTResult(
        mota=0.0, motp=0.0, idf1=0.0, num_switches=0, num_false_positives=0,
        num_misses=3, num_objects=3, num_matches=0, precision=0.0, recall=0.0,
    )


def test_ground_truth_beyond_the_tracker_output_raises_naming_the_mismatch(evaluator):
    ground_truth = [gt(0, 1, BOX_A), gt(10, 1, BOX_A), gt(20, 1, BOX_A)]
    tracks = [{1: track(1, BOX_A)} for _ in range(5)]  # frames 10 and 20 out of range

    with pytest.raises(ValueError) as exc_info:
        evaluator.evaluate(ground_truth, tracks)

    message = str(exc_info.value)
    assert '2 of 3' in message          # out-of-range GT frame count
    assert '5 tracker frames' in message
    assert '20' in message              # highest GT frame index
