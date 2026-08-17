"""Unit tests for EventDetector: possession-change detection and its team-comparison labelling."""
from __future__ import annotations

import pytest

from basketball.events.event_detector import (
    EventDetector,
    InterceptionEvent,
    PassEvent,
    UnclassifiedTransition,
)


def teams(*per_frame: dict[int, int]) -> list[dict[int, int]]:
    """One team-assignment dict per frame, in frame order."""
    return list(per_frame)


def constant_teams(mapping: dict[int, int], n_frames: int) -> list[dict[int, int]]:
    """The same team mapping repeated for every frame, for cases where the labels never change."""
    return [dict(mapping) for _ in range(n_frames)]


def test_same_team_transition_emits_one_pass():
    possession = [1, 1, 2, 2]
    team_assignment = constant_teams({1: 1, 2: 1}, 4)

    passes = EventDetector().identify_passes(possession, team_assignment)

    assert passes == [
        PassEvent(
            frame_idx=2, sender_track_id=1, receiver_track_id=2, sender_team=1, receiver_team=1,
        )
    ]


def test_cross_team_transition_emits_one_interception_with_the_receiver_as_interceptor():
    possession = [1, 1, 2, 2]
    team_assignment = constant_teams({1: 1, 2: 2}, 4)

    interceptions = EventDetector().identify_interceptions(possession, team_assignment)

    assert interceptions == [
        InterceptionEvent(
            frame_idx=2, passer_track_id=1, interceptor_track_id=2, passer_team=1, interceptor_team=2,
        )
    ]


def test_unresolved_previous_holder_team_emits_an_unclassified_transition():
    possession = [1, 1, 2, 2]
    team_assignment = constant_teams({1: 0, 2: 1}, 4)

    detector = EventDetector()

    assert detector.identify_unclassified(possession, team_assignment) == [
        UnclassifiedTransition(
            frame_idx=2, from_track_id=1, to_track_id=2, from_team=0, to_team=1,
        )
    ]
    assert detector.identify_passes(possession, team_assignment) == []
    assert detector.identify_interceptions(possession, team_assignment) == []


def test_unresolved_current_holder_team_emits_an_unclassified_transition():
    possession = [1, 1, 2, 2]
    team_assignment = constant_teams({1: 1, 2: 0}, 4)

    unclassified = EventDetector().identify_unclassified(possession, team_assignment)

    assert unclassified == [
        UnclassifiedTransition(
            frame_idx=2, from_track_id=1, to_track_id=2, from_team=1, to_team=0,
        )
    ]


def test_a_track_absent_from_the_team_dict_is_treated_exactly_like_an_explicit_zero():
    # PlayerAnnotator resolves a missing key to 0; this must match, so an
    # untracked player cannot silently become a labelled participant.
    possession = [1, 1, 2, 2]
    absent = constant_teams({1: 1}, 4)               # track 2 never appears
    explicit_zero = constant_teams({1: 1, 2: 0}, 4)

    detector = EventDetector()

    assert (
        detector.identify_unclassified(possession, absent)
        == detector.identify_unclassified(possession, explicit_zero)
    )


def test_transition_across_a_gap_within_the_limit_is_detected():
    # Possession legitimately reads -1 for long stretches -- clip_2 frames 92 to
    # 134 is a measured 43-frame gap across which the ball changed hands twice.
    possession = [1] + [-1] * 5 + [2]
    team_assignment = constant_teams({1: 1, 2: 1}, 7)

    passes = EventDetector(max_transition_gap=10).identify_passes(possession, team_assignment)

    assert [event.frame_idx for event in passes] == [6]


def test_transition_across_a_gap_beyond_the_limit_is_not_detected_but_the_new_holder_is_adopted():
    # The stale holder must not be paired with the new one, but the new one
    # still becomes the carried holder -- otherwise the NEXT real transition
    # would be lost too.
    possession = [1] + [-1] * 20 + [2, 2, 3]
    team_assignment = constant_teams({1: 1, 2: 1, 3: 1}, 24)

    detector = EventDetector(max_transition_gap=5)
    events = detector._find_transitions(possession, team_assignment)

    # The 1 -> 2 transition spans 21 frames and is suppressed; the 2 -> 3
    # transition one frame later is detected, proving 2 was adopted.
    assert [event.frame_idx for event in events] == [23]
    assert isinstance(events[0], PassEvent)
    assert events[0].sender_track_id == 2


def test_a_continuous_run_of_one_holder_emits_nothing():
    possession = [7] * 10
    team_assignment = constant_teams({7: 1}, 10)

    detector = EventDetector()

    assert detector._find_transitions(possession, team_assignment) == []


def test_flicker_emits_two_events_because_there_is_no_debounce():
    # Pins the deliberate no-debounce decision: measuring without one is what
    # makes the possession-versus-events error decomposition interpretable.
    possession = [1, 2, 1]
    team_assignment = constant_teams({1: 1, 2: 1}, 3)

    passes = EventDetector().identify_passes(possession, team_assignment)

    assert [(event.sender_track_id, event.receiver_track_id) for event in passes] == [(1, 2), (2, 1)]


def test_previous_holders_team_is_read_from_their_last_seen_frame_not_the_current_frame():
    # Track 1 is team 2 while holding (frames 0-1) but is mislabelled team 1 by
    # the time track 2 takes over -- reading the current frame would call this a
    # pass, reading the last-seen frame correctly calls it an interception.
    possession = [1, 1, 2]
    team_assignment = teams(
        {1: 2, 2: 1},
        {1: 2, 2: 1},
        {1: 1, 2: 1},
    )

    detector = EventDetector()

    assert detector.identify_passes(possession, team_assignment) == []
    assert detector.identify_interceptions(possession, team_assignment) == [
        InterceptionEvent(
            frame_idx=2, passer_track_id=1, interceptor_track_id=2, passer_team=2, interceptor_team=1,
        )
    ]


def test_length_mismatch_raises_naming_both_counts():
    with pytest.raises(ValueError, match='3 possession entries for 2 team assignment frames'):
        EventDetector().identify_passes([1, 1, 2], constant_teams({1: 1, 2: 1}, 2))


def test_empty_inputs_return_empty_lists_without_raising():
    detector = EventDetector()

    assert detector.identify_passes([], []) == []
    assert detector.identify_interceptions([], []) == []
    assert detector.identify_unclassified([], []) == []


def test_a_possession_list_of_all_minus_one_returns_empty_lists():
    possession = [-1] * 8
    team_assignment = constant_teams({1: 1}, 8)

    detector = EventDetector()

    assert detector.identify_passes(possession, team_assignment) == []
    assert detector.identify_interceptions(possession, team_assignment) == []
    assert detector.identify_unclassified(possession, team_assignment) == []


def test_an_out_of_range_team_value_is_treated_as_unresolved_rather_than_raising(capsys):
    # A single bad label must not kill a whole clip's run, but it must be
    # visible rather than silently folded into the unresolved bucket.
    possession = [1, 1, 2]
    team_assignment = constant_teams({1: 5, 2: 1}, 3)

    unclassified = EventDetector().identify_unclassified(possession, team_assignment)

    assert [event.from_team for event in unclassified] == [0]
    assert 'out-of-range team' in capsys.readouterr().out


def test_the_three_public_methods_partition_the_same_single_traversal():
    # Every detected change lands in exactly one of the three lists -- nothing
    # is dropped and nothing is double-counted.
    possession = [1, 2, 3, 4]
    team_assignment = constant_teams({1: 1, 2: 1, 3: 2, 4: 0}, 4)

    detector = EventDetector()
    total = (
        len(detector.identify_passes(possession, team_assignment))
        + len(detector.identify_interceptions(possession, team_assignment))
        + len(detector.identify_unclassified(possession, team_assignment))
    )

    assert total == len(detector._find_transitions(possession, team_assignment)) == 3


def test_the_default_gap_is_the_untuned_class_constant():
    assert EventDetector.MAX_TRANSITION_GAP == 30
    assert EventDetector().max_transition_gap == 30
    assert EventDetector(max_transition_gap=4).max_transition_gap == 4
