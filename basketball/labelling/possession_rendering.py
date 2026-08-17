"""
possession_rendering.py

Composes the single image the possession labelling notebook shows per frame:
tracked player boxes plus the real, gated ball detection or an explicit
no-detection caption.
"""
from __future__ import annotations

import cv2
import numpy as np

from basketball.detection.ball_detector import BallDetection
from basketball.labelling.frame_rendering import draw_tracked_boxes

# White, deliberately absent from frame_rendering.BOX_COLOURS so the ball can
# never be confused with a player box whatever the per-frame rank assignment
# happens to be, and drawn thicker than the player boxes for the same reason.
BALL_BOX_COLOUR: tuple[int, int, int] = (255, 255, 255)
BALL_BOX_THICKNESS = 3

BALL_CAPTION = 'BALL (detected)'
NO_BALL_CAPTION = 'NO BALL DETECTION THIS FRAME'

CAPTION_ORIGIN = (10, 28)
CAPTION_SCALE = 0.8
CAPTION_THICKNESS = 2
# Same "drop below the top edge when there is no room above" RULE as
# frame_rendering._label_y(), with its own thresholds rather than that
# function's (the ball caption is a different size), so the caption cannot
# render off-frame for a detection near the top of the image.
BALL_CAPTION_OFFSET_ABOVE_PX = -8
BALL_CAPTION_OFFSET_BELOW_PX = 20
BALL_TOP_EDGE_THRESHOLD_PX = 30


def draw_ball_box(frame: np.ndarray, ball_bbox: list[float]) -> np.ndarray:
    """Return a copy of frame with one real ball detection outlined in BALL_BOX_COLOUR and captioned as the ball."""
    annotated = frame.copy()
    x1, y1, x2, y2 = (int(value) for value in ball_bbox)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), BALL_BOX_COLOUR, BALL_BOX_THICKNESS)
    caption_y = (
        y1 + BALL_CAPTION_OFFSET_BELOW_PX
        if y1 < BALL_TOP_EDGE_THRESHOLD_PX
        else y1 + BALL_CAPTION_OFFSET_ABOVE_PX
    )
    cv2.putText(
        annotated,
        BALL_CAPTION,
        (x1, caption_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        BALL_BOX_COLOUR,
        2,
    )
    return annotated


def gated_ball_bbox(gated_frame: dict[int, BallDetection]) -> list[float] | None:
    """The bbox of the real ball detection on one gated frame, or None when the gate left that frame empty."""
    detection = gated_frame.get(1)
    return None if detection is None else list(detection.bbox)


def render_labelling_frame(
    frame: np.ndarray,
    tracks: dict[int, object],
    gated_frame: dict[int, BallDetection],
) -> np.ndarray:
    """Return the full annotated frame for one labelling decision: every tracked player box, plus the gated ball box or a no-detection caption."""
    # ONLY ever the gated (real) detection. An interpolated bbox would put a
    # confident-looking ball on a frame the detector never saw one in, and a
    # labeller shown that would answer from a fabricated position -- writing
    # the interpolator's guess into the ground truth that is supposed to
    # score it. Callers must pass a frame from BallInput.gated, never .filled.
    annotated = draw_tracked_boxes(frame, tracks)
    ball_bbox = gated_ball_bbox(gated_frame)
    if ball_bbox is not None:
        return draw_ball_box(annotated, ball_bbox)

    cv2.putText(
        annotated,
        NO_BALL_CAPTION,
        CAPTION_ORIGIN,
        cv2.FONT_HERSHEY_SIMPLEX,
        CAPTION_SCALE,
        BALL_BOX_COLOUR,
        CAPTION_THICKNESS,
    )
    return annotated


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to RGB for matplotlib's imshow, which expects RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
