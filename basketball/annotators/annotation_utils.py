"""
annotation_utils.py

Drawing primitives for player and ball annotations. Used by PlayerAnnotator
and BallAnnotator to render tracking output onto video frames.
"""
from typing import Tuple

import cv2
import numpy as np

from basketball.utils.box_utils import bbox_width
from basketball.utils.geometry import foot_point


def annotate_player(
    frame: np.ndarray,
    bbox: list[float],
    track_id: int,
    colour: Tuple[int, int, int],
) -> np.ndarray:
    """Draw a foot-position ellipse and track ID label onto the frame for a single player."""
    x1, y1, x2, y2 = bbox
    cx, fy = foot_point(x1, y1, x2, y2)
    width = bbox_width(x1, x2)

    cv2.ellipse(
        frame,
        center=(int(cx), int(fy)),
        axes=(int(width), int(0.35 * width)),
        angle=0,
        startAngle=45,
        endAngle=235,
        color=colour,
        thickness=2,
        lineType=cv2.LINE_4,
    )

    rect_w, rect_h = 40, 20
    x1_rect = int(cx) - rect_w // 2
    x2_rect = int(cx) + rect_w // 2
    y1_rect = int(fy) - rect_h // 2 + 15
    y2_rect = int(fy) + rect_h // 2 + 15

    cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), colour, cv2.FILLED)

    x1_text = x1_rect + 12
    if track_id > 99:
        x1_text -= 10

    cv2.putText(
        frame,
        str(track_id),
        (x1_text, y1_rect + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )

    return frame


def annotate_ball(
    frame: np.ndarray,
    bbox: list[float],
    colour: Tuple[int, int, int],
) -> np.ndarray:
    """Draw an inverted triangle pointer above the ball position onto the frame."""
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    top_y = int(y1)

    triangle_points = np.array([
        [cx, top_y],
        [cx - 10, top_y - 20],
        [cx + 10, top_y - 20],
    ])

    cv2.drawContours(frame, [triangle_points], 0, colour, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame
