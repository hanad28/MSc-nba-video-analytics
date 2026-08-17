"""
event_annotator.py

Renders each detected pass, interception and unclassified transition once, as a
transient top-right caption held for a few frames at the frame it fired on.
"""
from __future__ import annotations

import cv2
import numpy as np

from basketball.events.event_detector import (
    InterceptionEvent,
    PassEvent,
    UnclassifiedTransition,
)


class EventAnnotator:
    """
    Annotates video frames with a short-lived caption per detected event,
    rendering each detection where it fired rather than accumulating a running
    per-team total.
    """

    # A caption is held for this many frames from the event's own frame_idx so
    # it is legible at playback speed; an event near the clip end simply gets
    # however many frames remain.
    HOLD_FRAMES: int = 15

    # Deliberately transient, never a running tally: this stage measured 2 true
    # positives against 12 false positives, and a per-team counter would present
    # a phantom-inflated total as a summary statistic while letting a missed real
    # event and a phantom one cancel into a correct-looking number.

    # None of these collide with PlayerAnnotator's DEFAULT_TEAM_1_COLOUR,
    # DEFAULT_TEAM_2_COLOUR or UNKNOWN_TEAM_COLOUR, clip_3's red team-2 override
    # or PossessionAnnotator's cyan holder highlight, so an event caption can
    # never be read as a team or possession marker. Asserted by a test.
    PASS_COLOUR: tuple[int, int, int] = (0, 255, 0)        # BGR: green
    INTERCEPTION_COLOUR: tuple[int, int, int] = (0, 165, 255)  # BGR: orange
    UNCLASSIFIED_COLOUR: tuple[int, int, int] = (255, 0, 255)  # BGR: magenta

    # Beyond this many concurrent captions the oldest is dropped rather than
    # rendered: more lines than this cannot be read at playback speed anyway.
    MAX_SIMULTANEOUS_LINES: int = 3

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    THICKNESS = 2

    def draw(
        self,
        frames: list[np.ndarray],
        passes: list[PassEvent],
        interceptions: list[InterceptionEvent],
        unclassified: list[UnclassifiedTransition],
    ) -> list[np.ndarray]:
        """Return annotated copies of all frames with each event captioned on the frames it is held for."""
        captions = self._build_captions(frames, passes, interceptions, unclassified)

        output: list[np.ndarray] = []
        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()

            # Most recent last, so dropping the oldest keeps the newest visible.
            active = [
                (event_frame, text, colour)
                for event_frame, text, colour in captions
                if event_frame <= frame_idx < event_frame + self.HOLD_FRAMES
            ]
            active = active[-self.MAX_SIMULTANEOUS_LINES:]

            for line_index, (_, text, colour) in enumerate(active):
                self._draw_caption(frame, text, colour, line_index)

            output.append(frame)
        return output

    def _build_captions(
        self,
        frames: list[np.ndarray],
        passes: list[PassEvent],
        interceptions: list[InterceptionEvent],
        unclassified: list[UnclassifiedTransition],
    ) -> list[tuple[int, str, tuple[int, int, int]]]:
        """Return every event as a (frame_idx, caption text, colour) triple in chronological order, validating each frame index."""
        captions: list[tuple[int, str, tuple[int, int, int]]] = []

        for event in passes:
            self._validate_frame_idx(event.frame_idx, frames)
            captions.append((
                event.frame_idx,
                f'PASS  team {event.sender_team}  {event.sender_track_id} -> {event.receiver_track_id}',
                self.PASS_COLOUR,
            ))

        for event in interceptions:
            self._validate_frame_idx(event.frame_idx, frames)
            # The interceptor's team, not the passer's, matching how the
            # scoring notebook credits an interception.
            captions.append((
                event.frame_idx,
                f'INTERCEPTION  team {event.interceptor_team}  '
                f'{event.passer_track_id} -> {event.interceptor_track_id}',
                self.INTERCEPTION_COLOUR,
            ))

        for event in unclassified:
            self._validate_frame_idx(event.frame_idx, frames)
            # No team by definition: at least one participant is unresolved.
            captions.append((
                event.frame_idx,
                f'UNCLASSIFIED  {event.from_track_id} -> {event.to_track_id}',
                self.UNCLASSIFIED_COLOUR,
            ))

        captions.sort(key=lambda caption: caption[0])
        return captions

    def _validate_frame_idx(self, frame_idx: int, frames: list[np.ndarray]) -> None:
        """Raise if an event's frame index falls outside the frame list, which means mismatched inputs rather than something to skip."""
        if frame_idx < 0 or frame_idx >= len(frames):
            raise ValueError(
                f'Got an event at frame {frame_idx} for {len(frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )

    def _draw_caption(
        self,
        frame: np.ndarray,
        text: str,
        colour: tuple[int, int, int],
        line_index: int,
    ) -> None:
        """Draw one right-aligned caption line in the top-right corner, clamped inside the frame for any frame size."""
        height, width = frame.shape[:2]

        (text_w, text_h), baseline = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.THICKNESS)
        margin = 10
        line_height = text_h + baseline + 8

        # Right-aligned by measuring the text and offsetting from the frame
        # width, then clamped to 0 so a caption wider than the frame starts at
        # the left edge instead of at a negative x. PossessionAnnotator owns the
        # top-left, so on a frame too narrow to hold both they can meet; the
        # clamp keeps this one on-frame rather than off it.
        x = max(0, width - text_w - margin)
        y_origin = margin + text_h + line_index * line_height
        y = min(y_origin, height - baseline - 1) if height > text_h + baseline else text_h

        cv2.rectangle(
            frame,
            (x - 4, y - text_h - 4),
            (x + text_w + 4, y + baseline + 4),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(frame, text, (x, y), self.FONT, self.FONT_SCALE, colour, self.THICKNESS)
