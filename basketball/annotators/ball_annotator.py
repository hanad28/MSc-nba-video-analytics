"""
ball_annotator.py

Renders ball detection results onto video frames as inverted triangle
pointer markers. Frames where the ball is not detected are passed through
unchanged; gaps are filled downstream by BallInterpolator.
"""
from typing import Tuple

import numpy as np

from basketball.annotators.annotation_utils import annotate_ball
from basketball.detection.ball_detector import BallDetection


class BallAnnotator:
    """Annotates video frames with a ball position pointer triangle."""

    COLOUR: Tuple[int, int, int] = (0, 255, 0)  # green in BGR

    def __init__(self) -> None:
        self.colour = self.COLOUR

    def draw(
        self,
        frames: list[np.ndarray],
        detections: list[dict[int, BallDetection]],
    ) -> list[np.ndarray]:
        """Return annotated copies of all frames, passing frames with no ball detection through unchanged."""
        if len(detections) != len(frames):
            raise ValueError(
                f'Got {len(detections)} detection frames for {len(frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )

        output: list[np.ndarray] = []
        for frame, frame_dets in zip(frames, detections):
            frame = frame.copy()
            if 1 in frame_dets:
                frame = annotate_ball(frame, frame_dets[1].bbox, self.colour)
            output.append(frame)
        return output
