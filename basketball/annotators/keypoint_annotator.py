"""
keypoint_annotator.py

Renders detected court keypoints as labelled dots, drawing each point's index
beside it so a human can verdict whether that index sits on the right landmark.
"""
from __future__ import annotations

import cv2
import numpy as np

from basketball.keypoints.court_keypoints import Keypoint


class KeypointAnnotator:
    """Annotates video frames with confident court keypoints drawn as dots labelled by keypoint index."""

    # Yellow, absent from every colour already in use: PlayerAnnotator's
    # off-white/dark-blue/grey, PossessionAnnotator's cyan and white,
    # BallAnnotator's green, EventAnnotator's green/orange/magenta, and
    # clip_3's red team-2 override. Asserted by a non-collision test.
    KEYPOINT_COLOUR: tuple[int, int, int] = (0, 255, 255)  # BGR: yellow
    LABEL_COLOUR: tuple[int, int, int] = (0, 255, 255)     # BGR: yellow

    # Below this a point is not drawn at all: rendering all 18 when 8 are real
    # buries the confident points among predictions the model does not stand
    # behind, which is exactly what the verdicted audit must not be shown.
    DEFAULT_RENDER_THRESHOLD: float = 0.5

    RADIUS = 4
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    THICKNESS = 1

    def __init__(self, render_threshold: float | None = None) -> None:
        self.render_threshold = (
            self.DEFAULT_RENDER_THRESHOLD if render_threshold is None else render_threshold
        )

    def draw(
        self,
        frames: list[np.ndarray],
        keypoints_per_frame: list[list[Keypoint]],
    ) -> list[np.ndarray]:
        """Return annotated copies of all frames with each confident keypoint drawn as a dot labelled by its index."""
        if len(keypoints_per_frame) != len(frames):
            raise ValueError(
                f'Got {len(keypoints_per_frame)} keypoint frames for {len(frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )

        output: list[np.ndarray] = []
        for frame, frame_keypoints in zip(frames, keypoints_per_frame):
            frame = frame.copy()
            for keypoint in frame_keypoints:
                if keypoint.confidence < self.render_threshold:
                    continue
                self._draw_keypoint(frame, keypoint)
            output.append(frame)
        return output

    def _draw_keypoint(self, frame: np.ndarray, keypoint: Keypoint) -> None:
        """Draw one keypoint as a filled dot with its index rendered beside it, clamped inside the frame."""
        height, width = frame.shape[:2]
        x, y = int(round(keypoint.x)), int(round(keypoint.y))

        cv2.circle(frame, (x, y), self.RADIUS, self.KEYPOINT_COLOUR, thickness=-1)

        # The index label is the point of this annotator, not decoration: the
        # verdicted audit asks whether index k is on the correct landmark, and
        # that is unanswerable from unlabelled dots.
        label = str(keypoint.index)
        (text_w, text_h), _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, self.THICKNESS)
        label_x = min(max(0, x + self.RADIUS + 2), max(0, width - text_w))
        label_y = min(max(text_h, y - self.RADIUS - 2), max(text_h, height - 1))

        cv2.putText(
            frame, label, (label_x, label_y), self.FONT, self.FONT_SCALE,
            self.LABEL_COLOUR, self.THICKNESS,
        )
