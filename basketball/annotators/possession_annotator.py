"""
possession_annotator.py

Renders possession state onto video frames: a confirmed holder's box is
highlighted distinctly from PlayerAnnotator's team-colour ellipse, and every
frame is captioned with which of the three possession states it is in.
"""
from __future__ import annotations

import cv2
import numpy as np

from basketball.detection.ball_detector import BallDetection
from basketball.detection.player_detector import PlayerTrack


class PossessionAnnotator:
    """Annotates video frames with a possession highlight box and a caption naming the frame's possession state."""

    # Bright cyan, absent from PlayerAnnotator.DEFAULT_TEAM_1_COLOUR (off-white),
    # DEFAULT_TEAM_2_COLOUR (dark blue) and UNKNOWN_TEAM_COLOUR (grey), and from
    # clip_3's red team-2 override, so the highlight can never be mistaken for a
    # team-colour ellipse.
    HOLDER_COLOUR: tuple[int, int, int] = (255, 255, 0)  # BGR: cyan
    CAPTION_COLOUR: tuple[int, int, int] = (255, 255, 255)  # BGR: white
    NO_HOLDER_CAPTION = 'NO CONFIRMED HOLDER'
    INTERPOLATED_BALL_CAPTION = 'BALL POSITION INTERPOLATED'
    NO_BALL_CAPTION = 'BALL NOT DETECTED THIS FRAME'
    # Second-line caption for a confirmed-possession frame whose ball position
    # is fabricated rather than observed -- these are exactly the frames that
    # need visual inspection, since a streak can confirm off a single real
    # detection while every other frame in it is interpolated.
    CONFIRMED_HOLDER_INTERPOLATED_BALL_CAPTION = 'HOLDER CONFIRMED ON INTERPOLATED BALL'

    def draw(
        self,
        frames: list[np.ndarray],
        player_tracks: list[dict[int, PlayerTrack]],
        possession: list[int],
        ball_detections: list[dict[int, BallDetection]],
    ) -> list[np.ndarray]:
        """Return annotated copies of all frames with the confirmed holder highlighted and a possession-state caption."""
        if len(possession) != len(frames):
            raise ValueError(
                f'Got {len(possession)} possession entries for {len(frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )

        if len(player_tracks) != len(frames):
            raise ValueError(
                f'Got {len(player_tracks)} track frames for {len(frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )

        if len(ball_detections) != len(frames):
            raise ValueError(
                f'Got {len(ball_detections)} ball detection frames for {len(frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )

        output: list[np.ndarray] = []
        for frame, frame_tracks, holder_id, ball_frame in zip(frames, player_tracks, possession, ball_detections):
            frame = frame.copy()

            ball = ball_frame.get(1)

            if holder_id != -1:
                caption = f'POSSESSION: PLAYER {holder_id}'
                if holder_id in frame_tracks:
                    self._draw_holder_box(frame, frame_tracks[holder_id].bbox)
            elif ball is not None and ball.confidence > 0.0:
                caption = self.NO_HOLDER_CAPTION
            elif ball is not None:
                caption = self.INTERPOLATED_BALL_CAPTION
            else:
                caption = self.NO_BALL_CAPTION

            self._draw_caption(frame, caption)

            # Confirmed possession does not by itself say whether the ball was
            # observed that frame -- surface it as a second line rather than
            # folding it into the first, so the existing three captions and
            # their trigger conditions are untouched.
            if holder_id != -1 and ball is not None and ball.confidence == 0.0:
                self._draw_caption(frame, self.CONFIRMED_HOLDER_INTERPOLATED_BALL_CAPTION, line_index=1)

            output.append(frame)
        return output

    def _draw_holder_box(self, frame: np.ndarray, bbox: list[float]) -> None:
        """Draw a highlight rectangle around the confirmed holder's bounding box."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.HOLDER_COLOUR, thickness=3)

    def _draw_caption(self, frame: np.ndarray, text: str, line_index: int = 0) -> None:
        """Draw a possession-state caption line in the top-left corner, clamped inside the frame for any frame size; line_index stacks additional lines below the first."""
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        margin = 10
        line_height = text_h + baseline + 8  # matches the caption box's own padding below

        # Clamp rather than assume a fixed origin fits: a caption wider than a
        # narrow frame would otherwise run off the right edge, and a fixed
        # y-origin would sit above frame 0 for a very short frame. A second or
        # later line is clamped to this same last fitting position, so on a
        # frame too short to hold every line it overlaps the line above it
        # rather than running off the bottom.
        x = min(margin, max(0, width - text_w - margin))
        y_origin = margin + text_h + line_index * line_height
        y = min(y_origin, height - baseline - 1) if height > text_h + baseline else text_h

        cv2.rectangle(
            frame,
            (x - 4, y - text_h - 4),
            (x + text_w + 4, y + baseline + 4),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(frame, text, (x, y), font, font_scale, self.CAPTION_COLOUR, thickness)
