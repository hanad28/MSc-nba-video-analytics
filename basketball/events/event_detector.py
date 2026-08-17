"""
event_detector.py

Detects possession changes from the per-frame possession list and labels each
one a pass, an interception, or an unclassified transition by comparing the two
participants' team labels.
"""
from __future__ import annotations

from dataclasses import dataclass

RESOLVED_TEAMS = (1, 2)
VALID_TEAMS = (0, 1, 2)


@dataclass
class PassEvent:
    frame_idx: int
    sender_track_id: int
    receiver_track_id: int
    sender_team: int
    receiver_team: int


@dataclass
class InterceptionEvent:
    frame_idx: int
    passer_track_id: int
    interceptor_track_id: int
    passer_team: int
    interceptor_team: int


@dataclass
class UnclassifiedTransition:
    frame_idx: int
    from_track_id: int
    to_track_id: int
    from_team: int
    to_team: int


class EventDetector:
    """
    Emits a pass, interception or unclassified transition at each possession
    change, classifying it by comparing the previous and current holders' team
    labels.
    """

    # Not a tuned value: 30 frames is one second at 30fps, picked only so the
    # gate has a defined value. The leave-one-clip-out sweep found the
    # parameter inert at or above 20 with the folds disagreeing, so no value
    # was adopted and the starting value stands.
    MAX_TRANSITION_GAP: int = 30

    def __init__(self, max_transition_gap: int | None = None) -> None:
        self.max_transition_gap = (
            self.MAX_TRANSITION_GAP if max_transition_gap is None else max_transition_gap
        )

    def _resolve_team(self, team_assignment_frame: dict[int, int], track_id: int, frame_idx: int) -> int:
        """Return a track's team for one frame, treating an absent key and any out-of-range value alike as unresolved."""
        # An absent key resolves to 0 exactly as PlayerAnnotator.draw() does --
        # there is no -1 sentinel in this project's team data.
        team = team_assignment_frame.get(track_id, 0)

        if team not in VALID_TEAMS:
            # PlayerAnnotator raises on this, but a single bad label must not
            # kill a whole clip's run here; treat it as unresolved and say so.
            print(
                f'[events] Frame {frame_idx}: track {track_id} has out-of-range team '
                f'{team!r}; treating as unresolved.'
            )
            return 0

        return team

    def _classify(
        self,
        frame_idx: int,
        from_track_id: int,
        to_track_id: int,
        from_team: int,
        to_team: int,
    ) -> PassEvent | InterceptionEvent | UnclassifiedTransition:
        """Label one possession change by comparing the two participants' resolved team values."""
        if from_team not in RESOLVED_TEAMS or to_team not in RESOLVED_TEAMS:
            # A change the system detected but cannot label is still a real
            # detection; dropping it would hide recall failures.
            return UnclassifiedTransition(
                frame_idx=frame_idx,
                from_track_id=from_track_id,
                to_track_id=to_track_id,
                from_team=from_team,
                to_team=to_team,
            )

        if from_team == to_team:
            return PassEvent(
                frame_idx=frame_idx,
                sender_track_id=from_track_id,
                receiver_track_id=to_track_id,
                sender_team=from_team,
                receiver_team=to_team,
            )

        return InterceptionEvent(
            frame_idx=frame_idx,
            passer_track_id=from_track_id,
            interceptor_track_id=to_track_id,
            passer_team=from_team,
            interceptor_team=to_team,
        )

    def _find_transitions(
        self,
        possession: list[int],
        team_assignment: list[dict[int, int]],
    ) -> list[PassEvent | InterceptionEvent | UnclassifiedTransition]:
        """Walk the possession list once and return every detected possession change, of all three types."""
        if len(possession) != len(team_assignment):
            raise ValueError(
                f'Got {len(possession)} possession entries for {len(team_assignment)} team '
                f'assignment frames — the two must be aligned frame-for-frame.'
            )

        events: list[PassEvent | InterceptionEvent | UnclassifiedTransition] = []

        held_track_id: int | None = None
        last_seen_frame: int | None = None

        for frame_idx, holder_id in enumerate(possession):
            # -1 carries the previous holder forward rather than resetting:
            # possession legitimately reads -1 for long stretches, and a reset
            # would lose the transition that spans the gap.
            if holder_id == -1:
                continue

            if held_track_id is None:
                held_track_id = holder_id
                last_seen_frame = frame_idx
                continue

            if holder_id == held_track_id:
                last_seen_frame = frame_idx
                continue

            # Beyond the gap limit the carried holder is stale -- the ball may
            # have been uncontrolled throughout, so pairing them with the
            # current holder would fabricate an event spanning nothing.
            if frame_idx - last_seen_frame <= self.max_transition_gap:
                events.append(
                    self._classify(
                        frame_idx=frame_idx,
                        from_track_id=held_track_id,
                        to_track_id=holder_id,
                        # The previous holder's team is read from the frame they
                        # were last seen holding the ball: they may not be
                        # tracked at all in the current frame, where their label
                        # would be absent or 0.
                        from_team=self._resolve_team(
                            team_assignment[last_seen_frame], held_track_id, last_seen_frame,
                        ),
                        to_team=self._resolve_team(
                            team_assignment[frame_idx], holder_id, frame_idx,
                        ),
                    )
                )

            held_track_id = holder_id
            last_seen_frame = frame_idx

        return events

    def identify_passes(
        self,
        possession: list[int],
        team_assignment: list[dict[int, int]],
    ) -> list[PassEvent]:
        """Return every possession change between two players resolved to the same team."""
        return [
            event for event in self._find_transitions(possession, team_assignment)
            if isinstance(event, PassEvent)
        ]

    def identify_interceptions(
        self,
        possession: list[int],
        team_assignment: list[dict[int, int]],
    ) -> list[InterceptionEvent]:
        """Return every possession change between two players resolved to opposing teams."""
        return [
            event for event in self._find_transitions(possession, team_assignment)
            if isinstance(event, InterceptionEvent)
        ]

    def identify_unclassified(
        self,
        possession: list[int],
        team_assignment: list[dict[int, int]],
    ) -> list[UnclassifiedTransition]:
        """Return every possession change where either participant's team is unresolved."""
        return [
            event for event in self._find_transitions(possession, team_assignment)
            if isinstance(event, UnclassifiedTransition)
        ]
