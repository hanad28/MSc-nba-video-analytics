"""Unit tests for PossessionTracker."""

from __future__ import annotations

import json

import pytest

from basketball.detection.ball_detector import BallDetection
from basketball.detection.player_detector import PlayerTrack
from basketball.possession.possession_tracker import PossessionTracker

CONFIG = {
    'possession': {
        'max_ball_distance': 50.0,
        'hold_threshold': 3,
        'bbox_overlap_min': 0.5,
    }
}


def make_tracker(**overrides: float | int) -> PossessionTracker:
    config = {'possession': dict(CONFIG['possession'], **overrides)}
    return PossessionTracker(config)


def player(track_id: int, bbox: list[float], confidence: float = 0.9) -> PlayerTrack:
    return PlayerTrack(track_id=track_id, bbox=bbox, confidence=confidence)


def ball_frame(bbox: list[float], confidence: float = 0.9) -> dict[int, BallDetection]:
    return {1: BallDetection(bbox=bbox, confidence=confidence)}


@pytest.fixture
def tracker() -> PossessionTracker:
    return make_tracker()


def test_config_is_read_from_possession_section(tracker):
    assert tracker.max_ball_distance == 50.0
    assert tracker.hold_threshold == 3
    assert tracker.bbox_overlap_min == 0.5


def test_overlap_ratio_full_containment(tracker):
    assert tracker.overlap_ratio([0, 0, 100, 100], [10, 10, 20, 20]) == 1.0


def test_overlap_ratio_no_overlap(tracker):
    assert tracker.overlap_ratio([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0


def test_overlap_ratio_partial_containment(tracker):
    # Ball spans x 5..15 against a player ending at x=10: half the ball area.
    assert tracker.overlap_ratio([0, 0, 10, 20], [5, 5, 15, 15]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    'ball_bbox',
    [
        [10, 10, 10, 20],   # zero width
        [10, 10, 20, 10],   # zero height
        [10, 10, 10, 10],   # a point
    ],
)
def test_overlap_ratio_returns_zero_for_a_zero_area_ball_box(tracker, ball_bbox: list[float]) -> None:
    # Each of these previously divided by a zero ball area and raised
    # ZeroDivisionError, which would take down a whole clip's run on one bad
    # box. A box with no area carries no containment evidence, so 0.0 and let
    # find_holder() fall through to its proximity test. Verified to fail
    # against the pre-fix tracker.
    assert tracker.overlap_ratio([0, 0, 100, 100], ball_bbox) == 0.0


def test_overlap_ratio_returns_zero_for_an_inverted_ball_box(tracker) -> None:
    # NOT a regression case, and deliberately separated from the zero-area
    # ones above rather than being a fourth parameter of them: an inverted box
    # cannot reach the area division at all, so it returned 0.0 before the
    # zero-area guard existed too. max(px1, bx1) >= bx1 > bx2 >= min(px2, bx2)
    # whenever bx2 < bx1, so the empty-intersection check above it always
    # fires first. Pinned so that a future reordering of the two guards still
    # returns 0.0 rather than dividing by a negative area.
    assert tracker.overlap_ratio([0, 0, 100, 100], [20, 20, 10, 10]) == 0.0


def test_find_holder_survives_a_degenerate_ball_box(tracker) -> None:
    # The end-to-end consequence of the guard above: a degenerate box must not
    # crash holder resolution, and proximity still decides the answer.
    tracks = {4: PlayerTrack(track_id=4, bbox=[0, 0, 100, 100], confidence=0.9)}

    holder = tracker.find_holder((50.0, 50.0), tracks, [50, 50, 50, 50])

    assert holder == 4


def test_closest_distance_is_zero_for_point_on_the_box_edge(tracker):
    assert tracker.closest_distance((0.0, 5.0), [0, 0, 10, 10]) == 0.0


def test_closest_distance_grows_with_separation(tracker):
    near = tracker.closest_distance((20.0, 5.0), [0, 0, 10, 10])
    far = tracker.closest_distance((100.0, 5.0), [0, 0, 10, 10])
    assert near == pytest.approx(10.0)
    assert far > near


def test_get_proximity_points_adds_side_points_when_vertically_aligned(tracker):
    points = tracker.get_proximity_points([0, 0, 10, 10], (5.0, 5.0))
    assert (0, 5.0) in points and (10, 5.0) in points
    assert (5.0, 0) in points and (5.0, 10) in points


def test_get_proximity_points_omits_side_points_when_outside_the_box(tracker):
    points = tracker.get_proximity_points([0, 0, 10, 10], (50.0, 50.0))
    assert (0, 50.0) not in points
    assert (50.0, 0) not in points


def test_find_holder_prefers_containment_over_proximity(tracker):
    ball_bbox = [10, 10, 14, 14]
    tracks = {
        1: player(1, [0, 0, 100, 100]),   # contains the ball
        2: player(2, [14, 10, 20, 16]),   # touching, but no containment
    }
    assert tracker.find_holder((12.0, 12.0), tracks, ball_bbox) == 1


def test_find_holder_picks_highest_containment(tracker):
    ball_bbox = [0, 0, 10, 10]
    tracks = {
        1: player(1, [0, 0, 8, 100]),    # 80% of the ball
        2: player(2, [0, 0, 100, 100]),  # 100% of the ball
    }
    assert tracker.find_holder((5.0, 5.0), tracks, ball_bbox) == 2


def test_find_holder_falls_back_to_nearest_player(tracker):
    ball_bbox = [100, 100, 104, 104]
    tracks = {
        1: player(1, [90, 90, 98, 98]),
        2: player(2, [130, 130, 140, 140]),
    }
    assert tracker.find_holder((102.0, 102.0), tracks, ball_bbox) == 1


def test_find_holder_returns_minus_one_when_all_players_are_too_far(tracker):
    tracks = {1: player(1, [0, 0, 10, 10])}
    assert tracker.find_holder((1000.0, 1000.0), tracks, [1000, 1000, 1004, 1004]) == -1


def test_find_holder_returns_minus_one_for_empty_frame(tracker):
    assert tracker.find_holder((5.0, 5.0), {}, [0, 0, 10, 10]) == -1


def test_find_holder_skips_players_with_empty_bbox(tracker):
    tracks = {
        1: player(1, []),
        2: player(2, [0, 0, 100, 100]),
    }
    assert tracker.find_holder((5.0, 5.0), tracks, [0, 0, 10, 10]) == 2


def test_assign_possession_backfills_the_confirmed_streak(tracker):
    frames = 4
    player_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(frames)]
    ball_detections = [ball_frame([10, 10, 14, 14]) for _ in range(frames)]

    assert tracker.assign_possession(player_tracks, ball_detections) == [1, 1, 1, 1]


def test_assign_possession_requires_the_hold_threshold(tracker):
    player_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(2)]
    ball_detections = [ball_frame([10, 10, 14, 14]) for _ in range(2)]

    assert tracker.assign_possession(player_tracks, ball_detections) == [-1, -1]


def test_assign_possession_requires_a_real_anchor(tracker):
    """A streak built only from interpolated (confidence 0.0) detections is not confirmed."""
    frames = 5
    player_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(frames)]
    ball_detections = [ball_frame([10, 10, 14, 14], confidence=0.0) for _ in range(frames)]

    assert tracker.assign_possession(player_tracks, ball_detections) == [-1] * frames


def test_assign_possession_is_gap_tolerant_on_frames_without_a_ball(tracker):
    player_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(4)]
    ball_detections = [
        ball_frame([10, 10, 14, 14]),
        {},
        ball_frame([10, 10, 14, 14]),
        ball_frame([10, 10, 14, 14]),
    ]

    # The streak survives the ball-less frame, and confirmation backfills across it.
    assert tracker.assign_possession(player_tracks, ball_detections) == [1, 1, 1, 1]


def test_assign_possession_resets_when_no_player_clears_the_gates(tracker):
    holder = {1: player(1, [0, 0, 100, 100])}
    far_ball = ball_frame([1000, 1000, 1004, 1004])
    near_ball = ball_frame([10, 10, 14, 14])

    player_tracks = [holder] * 6
    ball_detections = [near_ball, near_ball, far_ball, near_ball, near_ball, near_ball]

    assert tracker.assign_possession(player_tracks, ball_detections) == [-1, -1, -1, 1, 1, 1]


def test_assign_possession_resets_on_candidate_switch(tracker):
    """Documented limitation: switching candidate restarts the streak from zero."""
    both = {
        1: player(1, [0, 0, 20, 20]),
        2: player(2, [80, 80, 100, 100]),
    }
    ball_with_one = ball_frame([5, 5, 9, 9])
    ball_with_two = ball_frame([85, 85, 89, 89])

    player_tracks = [both] * 4
    ball_detections = [ball_with_one, ball_with_one, ball_with_two, ball_with_one]

    assert tracker.assign_possession(player_tracks, ball_detections) == [-1, -1, -1, -1]


def test_assign_possession_empty_input(tracker):
    assert tracker.assign_possession([], []) == []


def test_assign_possession_uses_and_writes_the_cache(tracker, tmp_path):
    from basketball.cache.cache_utils import save_cache

    cache_path = str(tmp_path / 'nested' / 'possession.pkl')
    player_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(3)]
    ball_detections = [ball_frame([10, 10, 14, 14]) for _ in range(3)]

    first = tracker.assign_possession(player_tracks, ball_detections, cache_path=cache_path)
    assert first == [1, 1, 1]

    # Overwrite the pickle with a sentinel while keeping the matching sidecar:
    # getting the sentinel back proves the second call was served from the cache.
    save_cache([9, 9, 9], cache_path)

    assert tracker.assign_possession(player_tracks, ball_detections, cache_path=cache_path) == [9, 9, 9]


def test_assign_possession_recomputes_when_the_logic_revision_changes(tracker, tmp_path):
    """Generic mechanism test: ANY change to LOGIC_REVISION invalidates a cache written under the previous value, whatever that value currently is."""
    from basketball.cache.cache_utils import save_cache

    cache_path = str(tmp_path / 'possession.pkl')
    player_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(3)]
    ball_detections = [ball_frame([10, 10, 14, 14]) for _ in range(3)]
    tracker.assign_possession(player_tracks, ball_detections, cache_path=cache_path)

    # Sentinel under a matching sidecar: served back only while the revision matches.
    save_cache([9, 9, 9], cache_path)
    assert tracker.assign_possession(player_tracks, ball_detections, cache_path=cache_path) == [9, 9, 9]

    # Relative to whatever LOGIC_REVISION currently is, not a hardcoded next
    # value: a hardcoded 2 stopped being a real change the moment the class
    # constant itself reached 2 (see the overlap_ratio zero-area guard bump),
    # which would have made this assert a no-op comparison against the same
    # cache rather than a genuine revision-change test.
    tracker.LOGIC_REVISION += 1  # instance attribute shadows the class constant

    assert tracker.assign_possession(player_tracks, ball_detections, cache_path=cache_path) == [1, 1, 1]


def test_assign_possession_recomputes_a_cache_written_under_logic_revision_1(tracker, tmp_path):
    """Pins the actual 1 -> 2 bump made for overlap_ratio's zero-area guard: a real cache written by the pre-fix tracker (fingerprint logic_revision=1) is not served by today's tracker."""
    from basketball.cache.cache_utils import data_digest, save_cache_with_meta

    cache_path = str(tmp_path / 'possession.pkl')
    player_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(3)]
    ball_detections = [ball_frame([10, 10, 14, 14]) for _ in range(3)]

    # A fingerprint exactly as the pre-fix tracker (LOGIC_REVISION 1) would
    # have written it -- same keys assign_possession() itself builds, with
    # logic_revision hardcoded to the OLD value rather than read from the
    # class, so this reproduces a real leftover cache rather than whatever
    # the class happens to say today.
    stale_fingerprint = {
        'player_tracks_digest': data_digest(player_tracks),
        'ball_detections_digest': data_digest(ball_detections),
        'hold_threshold': tracker.hold_threshold,
        'bbox_overlap_min': tracker.bbox_overlap_min,
        'max_ball_distance': tracker.max_ball_distance,
        'n_frames': len(ball_detections),
        'logic_revision': 1,
    }
    save_cache_with_meta([9, 9, 9], cache_path, stale_fingerprint)

    # The stale revision-1 cache must be rejected outright: real output,
    # never the sentinel a hit would have returned. This is the behavioural
    # assertion the test exists for, and comes before the literal-value pin
    # below so a real regression fails here first, with the informative
    # [9, 9, 9] mismatch, rather than at a premise check.
    assert tracker.assign_possession(player_tracks, ball_detections, cache_path=cache_path) == [1, 1, 1]

    # The fingerprint assign_possession() just wrote over the stale one now
    # carries today's actual revision -- chained equality against the
    # literal, the same convention test_detectors.py/test_team_classifier.py
    # use for INFERENCE_REVISION/CLASSIFIER_REVISION, so a future bump that
    # forgets to update this literal fails loudly here.
    with open(f'{cache_path}.meta.json') as f:
        fingerprint = json.load(f)
    assert fingerprint['logic_revision'] == PossessionTracker.LOGIC_REVISION == 2


def test_assign_possession_recomputes_for_same_length_but_different_inputs(tracker, tmp_path):
    """The pre-fingerprint guard reused this cache: same frame count, different inputs."""
    cache_path = str(tmp_path / 'possession.pkl')
    player_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(3)]
    ball_detections = [ball_frame([10, 10, 14, 14]) for _ in range(3)]
    assert tracker.assign_possession(player_tracks, ball_detections, cache_path=cache_path) == [1, 1, 1]

    # Same length, but no players or ball anywhere: a returned [1, 1, 1] could
    # only be stale reuse, so recomputation must yield [-1, -1, -1].
    assert tracker.assign_possession([{}] * 3, [{}] * 3, cache_path=cache_path) == [-1, -1, -1]


def test_assign_possession_recomputes_when_the_cache_length_mismatches(tracker, tmp_path):
    cache_path = str(tmp_path / 'possession.pkl')
    player_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(3)]
    ball_detections = [ball_frame([10, 10, 14, 14]) for _ in range(3)]
    tracker.assign_possession(player_tracks, ball_detections, cache_path=cache_path)

    # A 4-frame input against the 3-frame cache is stale: the 4-frame result
    # can only come from recomputation, never from the cached [1, 1, 1].
    longer_tracks = [{1: player(1, [0, 0, 100, 100])} for _ in range(4)]
    longer_balls = [ball_frame([10, 10, 14, 14]) for _ in range(4)]

    assert tracker.assign_possession(longer_tracks, longer_balls, cache_path=cache_path) == [1, 1, 1, 1]
