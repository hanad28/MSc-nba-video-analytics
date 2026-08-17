"""Unit tests for Stage 8's CourtMapper (basketball/homography/court_mapper.py)."""

from __future__ import annotations

import inspect

import pytest

from basketball.detection.player_detector import PlayerTrack
from basketball.homography.court_mapper import COURT_MARGIN_M, CourtMapper
from basketball.keypoints.court_keypoints import Keypoint
from basketball.keypoints.court_template import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NUM_KEYPOINTS,
    TEMPLATE_POINTS_M,
)

SCALE = 20.0
OFFSET = (100.0, 50.0)

# The measured framing behaviour: a left-half frame yields 0-5, 8, 9 and a
# right-half frame 10-17, both within one clip. Six confident keypoints is the
# mode; four is the minimum viable; three cannot produce a homography.
LEFT_HALF_SIX = [0, 1, 2, 3, 8, 9]
RIGHT_HALF_SIX = [10, 11, 12, 13, 16, 17]
VIABLE_FOUR = [0, 5, 8, 9]
BELOW_THRESHOLD_THREE = [0, 5, 8]
ONE_BASELINE_FOUR = [0, 1, 2, 3]

# A real broadcast frame's per-keypoint confidences: mostly exactly 1.0 or
# exactly 0.0 with a thin middle band. Six of these clear the 0.5 threshold.
REAL_CONFIDENCES = (
    1.0, 1.0, 1.0, 0.991, 0.306, 0.108, 0.204, 0.154, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)


def to_image(index: int) -> tuple[float, float]:
    """Map a keypoint's template position into image space under the exact synthetic transform."""
    x_m, y_m = TEMPLATE_POINTS_M[index]
    return (x_m * SCALE + OFFSET[0], y_m * SCALE + OFFSET[1])


def keypoint_frame(indices: list[int], confidence: float = 1.0) -> list[Keypoint]:
    """A fixed-length 18-keypoint frame with the given indices confident, matching CourtKeypoints' contract."""
    frame = [Keypoint(index=i, x=0.0, y=0.0, confidence=0.0) for i in range(NUM_KEYPOINTS)]
    for index in indices:
        x, y = to_image(index)
        frame[index] = Keypoint(index=index, x=x, y=y, confidence=confidence)
    return frame


def track_at_court_position(track_id: int, position_m: tuple[float, float], height_px: float = 80.0) -> PlayerTrack:
    """A track whose FOOT POINT sits at the given court position under the synthetic transform."""
    foot_x = position_m[0] * SCALE + OFFSET[0]
    foot_y = position_m[1] * SCALE + OFFSET[1]
    return PlayerTrack(
        track_id=track_id,
        bbox=[foot_x - 10.0, foot_y - height_px, foot_x + 10.0, foot_y],
        confidence=0.9,
    )


# --- the mapping ---------------------------------------------------------

def test_a_player_maps_to_their_true_court_position():
    expected = (5.79, 7.62)
    tracks = [{7: track_at_court_position(7, expected)}]

    positions, report = CourtMapper().map_to_court(tracks, [keypoint_frame(LEFT_HALF_SIX)])

    assert positions[0][7] == pytest.approx(expected, abs=1e-4)
    assert report.mapped_frames == 1
    assert report.positions_mapped == 1


@pytest.mark.parametrize('indices', [LEFT_HALF_SIX, RIGHT_HALF_SIX, VIABLE_FOUR])
def test_every_viable_index_set_produces_a_mapping(indices):
    tracks = [{7: track_at_court_position(7, (10.0, 7.0))}]

    positions, report = CourtMapper().map_to_court(tracks, [keypoint_frame(indices)])

    assert report.mapped_frames == 1
    assert positions[0][7] == pytest.approx((10.0, 7.0), abs=1e-4)


def test_a_real_confidence_vector_yields_a_mapping():
    # Guards against a fixture that only ever uses one synthetic confidence:
    # the real distribution is near-binary with a thin middle band, and the
    # middle band is where a threshold comparison can go wrong.
    frame = [
        Keypoint(index=i, x=to_image(i)[0], y=to_image(i)[1], confidence=confidence)
        for i, confidence in enumerate(REAL_CONFIDENCES)
    ]
    tracks = [{7: track_at_court_position(7, (5.0, 5.0))}]

    positions, report = CourtMapper().map_to_court(tracks, [frame])

    assert report.mapped_frames == 1
    assert positions[0][7] == pytest.approx((5.0, 5.0), abs=1e-4)


# --- the foot point ------------------------------------------------------

def test_the_foot_point_is_used_rather_than_the_box_centre():
    # A homography maps the ground plane, and only the feet are on it. The box
    # is made tall enough that the centre maps to a visibly different court
    # position, so the two answers cannot be confused.
    expected = (5.0, 6.0)
    track = track_at_court_position(7, expected, height_px=200.0)
    centre_y = (track.bbox[1] + track.bbox[3]) / 2.0
    centre_court_y = (centre_y - OFFSET[1]) / SCALE

    positions, _ = CourtMapper().map_to_court(
        [{7: track}], [keypoint_frame(LEFT_HALF_SIX)],
    )

    assert positions[0][7] == pytest.approx(expected, abs=1e-4)
    # The centre would have given a materially different answer, so this test
    # discriminates between the two rather than passing either way.
    assert abs(centre_court_y - expected[1]) > 1.0
    assert positions[0][7][1] != pytest.approx(centre_court_y, abs=0.5)


# --- frames that cannot be mapped ----------------------------------------

def test_a_frame_below_the_threshold_yields_an_empty_dict_not_none():
    tracks = [{7: track_at_court_position(7, (5.0, 5.0))}]

    positions, report = CourtMapper().map_to_court(
        tracks, [keypoint_frame(BELOW_THRESHOLD_THREE)],
    )

    assert positions[0] == {}
    assert positions[0] is not None
    assert report.insufficient_keypoints == 1
    assert report.mapped_frames == 0


def test_a_degenerate_frame_is_counted_apart_from_an_insufficient_one():
    # Four points from one baseline is a different failure from three points:
    # the frame had enough keypoints, they just cannot determine a transform.
    tracks = [{7: track_at_court_position(7, (5.0, 5.0))}] * 2

    _, report = CourtMapper().map_to_court(
        tracks, [keypoint_frame(ONE_BASELINE_FOUR), keypoint_frame(BELOW_THRESHOLD_THREE)],
    )

    assert report.degenerate_keypoints == 1
    assert report.insufficient_keypoints == 1


def test_there_is_no_fallback_to_the_previous_frames_homography():
    # The behaviour a plausible simplification would break silently. Carrying
    # the last valid matrix would map the middle frame's player to a position
    # rather than none, and Stage 9 would read the resulting jump as speed.
    tracks = [{7: track_at_court_position(7, (5.0, 5.0))}] * 3
    frames = [
        keypoint_frame(LEFT_HALF_SIX),
        keypoint_frame(BELOW_THRESHOLD_THREE),
        keypoint_frame(LEFT_HALF_SIX),
    ]

    positions, report = CourtMapper().map_to_court(tracks, frames)

    assert positions[1] == {}, 'a failed frame must not inherit the previous matrix'
    # The successful frames on either side are unaffected by the gap.
    assert positions[0][7] == pytest.approx((5.0, 5.0), abs=1e-4)
    assert positions[2][7] == pytest.approx((5.0, 5.0), abs=1e-4)
    assert report.mapped_frames == 2


def test_a_contiguous_run_of_failures_maps_nothing_throughout():
    # clip_3's shortfall arrives as one contiguous 45-frame window, not as
    # scattered frames, so the run is the case that matters.
    run_length = 45
    tracks = [{7: track_at_court_position(7, (5.0, 5.0))}] * (run_length + 2)
    frames = (
        [keypoint_frame(LEFT_HALF_SIX)]
        + [keypoint_frame(BELOW_THRESHOLD_THREE)] * run_length
        + [keypoint_frame(LEFT_HALF_SIX)]
    )

    positions, report = CourtMapper().map_to_court(tracks, frames)

    assert all(positions[i] == {} for i in range(1, run_length + 1))
    assert report.mapped_frames == 2
    assert report.insufficient_keypoints == run_length


# --- bounds --------------------------------------------------------------

def test_a_position_beyond_the_margin_is_dropped_and_counted():
    far_outside = (COURT_LENGTH_M + COURT_MARGIN_M + 5.0, COURT_WIDTH_M / 2)
    tracks = [{
        7: track_at_court_position(7, (5.0, 5.0)),
        8: track_at_court_position(8, far_outside),
    }]

    positions, report = CourtMapper().map_to_court(tracks, [keypoint_frame(LEFT_HALF_SIX)])

    assert 7 in positions[0]
    assert 8 not in positions[0]
    assert report.positions_dropped_out_of_bounds == 1
    assert report.positions_mapped == 1


def test_a_player_just_past_the_sideline_is_kept():
    # Players legitimately step past the line; the margin exists for them.
    just_outside = (5.0, COURT_WIDTH_M + COURT_MARGIN_M / 2)
    tracks = [{7: track_at_court_position(7, just_outside)}]

    positions, report = CourtMapper().map_to_court(tracks, [keypoint_frame(LEFT_HALF_SIX)])

    assert positions[0][7] == pytest.approx(just_outside, abs=1e-4)
    assert report.positions_dropped_out_of_bounds == 0


def test_a_negative_court_position_within_the_margin_is_kept():
    tracks = [{7: track_at_court_position(7, (-COURT_MARGIN_M / 2, 5.0))}]

    positions, _ = CourtMapper().map_to_court(tracks, [keypoint_frame(LEFT_HALF_SIX)])

    assert 7 in positions[0]


# --- output shape and reporting ------------------------------------------

def test_output_length_always_equals_the_frame_count():
    tracks = [{7: track_at_court_position(7, (5.0, 5.0))}] * 5
    frames = [
        keypoint_frame(LEFT_HALF_SIX),
        keypoint_frame(BELOW_THRESHOLD_THREE),
        keypoint_frame(RIGHT_HALF_SIX),
        keypoint_frame(ONE_BASELINE_FOUR),
        keypoint_frame(VIABLE_FOUR),
    ]

    positions, report = CourtMapper().map_to_court(tracks, frames)

    assert len(positions) == len(frames) == 5
    assert all(isinstance(entry, dict) for entry in positions)
    assert report.n_frames == 5


def test_the_report_reconciles_every_frame():
    tracks = [{}] * 4
    frames = [
        keypoint_frame(LEFT_HALF_SIX),
        keypoint_frame(BELOW_THRESHOLD_THREE),
        keypoint_frame(ONE_BASELINE_FOUR),
        keypoint_frame(RIGHT_HALF_SIX),
    ]

    _, report = CourtMapper().map_to_court(tracks, frames)

    assert report.reconciles()
    assert report.mapped_frames == 2
    assert report.unmapped_frames == 2
    assert 'frames mapped' in report.summary()


def test_an_empty_frame_of_tracks_still_counts_as_mapped():
    # A frame with a homography but no players is not the same as a frame
    # without a homography, and the report must not conflate them.
    _, report = CourtMapper().map_to_court([{}], [keypoint_frame(LEFT_HALF_SIX)])

    assert report.mapped_frames == 1
    assert report.positions_mapped == 0


def test_the_report_flags_which_frames_produced_a_homography():
    # MinimapAnnotator captions on this, so it must distinguish a homography
    # failure from a homography that simply had no players to map.
    tracks = [{7: track_at_court_position(7, (5.0, 5.0))}, {}, {}]
    frames = [
        keypoint_frame(LEFT_HALF_SIX),          # mapped, with a player
        keypoint_frame(LEFT_HALF_SIX),          # mapped, no players tracked
        keypoint_frame(BELOW_THRESHOLD_THREE),  # no homography
    ]

    positions, report = CourtMapper().map_to_court(tracks, frames)

    assert report.frame_has_homography == [True, True, False]
    # The middle frame is empty for a reason that is NOT a failure, which is
    # exactly the state the annotator must not caption as one.
    assert positions[1] == {} and report.frame_has_homography[1] is True


def test_the_homography_flags_have_one_entry_per_frame():
    tracks = [{}] * 4
    frames = [
        keypoint_frame(LEFT_HALF_SIX),
        keypoint_frame(BELOW_THRESHOLD_THREE),
        keypoint_frame(ONE_BASELINE_FOUR),
        keypoint_frame(RIGHT_HALF_SIX),
    ]

    _, report = CourtMapper().map_to_court(tracks, frames)

    assert len(report.frame_has_homography) == report.n_frames == 4
    assert sum(report.frame_has_homography) == report.mapped_frames


def test_a_frame_whose_only_position_was_dropped_still_flags_a_homography():
    # The third cause of an empty dict: fitted, but every position was out of
    # bounds. Still not a homography failure.
    far_outside = (COURT_LENGTH_M + COURT_MARGIN_M + 5.0, COURT_WIDTH_M / 2)
    tracks = [{7: track_at_court_position(7, far_outside)}]

    positions, report = CourtMapper().map_to_court(tracks, [keypoint_frame(LEFT_HALF_SIX)])

    assert positions[0] == {}
    assert report.frame_has_homography == [True]
    assert report.positions_dropped_out_of_bounds == 1


def test_mismatched_input_lengths_raise():
    with pytest.raises(ValueError, match='aligned frame-for-frame'):
        CourtMapper().map_to_court([{}, {}], [keypoint_frame(LEFT_HALF_SIX)])


def test_the_keypoint_confidence_threshold_gates_which_keypoints_are_used():
    # Six keypoints just below the threshold leave nothing to fit.
    tracks = [{7: track_at_court_position(7, (5.0, 5.0))}]
    frame = keypoint_frame(LEFT_HALF_SIX, confidence=0.49)

    _, report = CourtMapper(keypoint_confidence_threshold=0.5).map_to_court(tracks, [frame])

    assert report.insufficient_keypoints == 1

    _, permissive = CourtMapper(keypoint_confidence_threshold=0.4).map_to_court(tracks, [frame])

    assert permissive.mapped_frames == 1


def test_the_threshold_is_named_for_the_per_keypoint_quantity_not_the_instance_one():
    # config's keypoints.confidence_threshold is CourtKeypoints' INSTANCE
    # threshold, a deliberately different quantity that gates whether a court
    # is detected at all. Both default to 0.5, so passing the config value into
    # a parameter called confidence_threshold would look correct and be wrong.
    # The longer name is what makes that mistake visible at the wiring site.
    parameters = inspect.signature(CourtMapper.__init__).parameters

    assert 'keypoint_confidence_threshold' in parameters
    assert 'confidence_threshold' not in parameters
