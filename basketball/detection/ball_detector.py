"""
ball_detector.py

YOLOv8-based ball detection. No tracker is used; gaps in detection are
filled downstream by BallInterpolator. Produces per-frame dicts of 1 → BallDetection,
cached to disk to avoid repeated inference during development.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import supervision as sv
from ultralytics import YOLO

from basketball.cache.cache_utils import file_digest, load_valid_cache, save_cache_with_meta


@dataclass
class BallDetection:
    bbox: list[float]      # [x1, y1, x2, y2]
    confidence: float


class BallDetector:
    """Detects the ball across video frames using YOLOv8 with no tracker; gaps are filled by BallInterpolator."""

    CLASS_NAME = 'ball'
    CONF_THRESHOLD = 0.3

    # Increment whenever the inference call, the filtering logic or anything
    # else the parameter keys cannot see changes; same rule as
    # PossessionTracker.LOGIC_REVISION. 1: conf passed through to inference,
    # deliberately invalidating every detection cache written before it.
    INFERENCE_REVISION = 1

    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.class_id = self._resolve_class_index()

    def _resolve_class_index(self) -> int:
        """Return the checkpoint's own class index for CLASS_NAME, matched case-insensitively."""
        # 'Ball' is index 0 in the current fy4c2 checkpoint only because its class
        # list happens to sort that way; the index is read from model.names so a
        # future checkpoint swap cannot silently zero out detections.
        for class_id, name in self.model.names.items():
            if name.lower() == self.CLASS_NAME.lower():
                return class_id
        raise ValueError(
            f'No class named {self.CLASS_NAME!r} in this checkpoint; '
            f'available classes: {self.model.names}'
        )

    def filter_detections(self, detections: sv.Detections) -> sv.Detections:
        """Remove non-ball detections and boxes below confidence threshold."""
        return detections[
            (detections.class_id == self.class_id) &
            (detections.confidence >= self.CONF_THRESHOLD)
        ]

    def _cache_fingerprint(self, video_path: str, n_frames: int) -> dict:
        """Fingerprint of every input that determines this stage's output, used to invalidate same-length stale caches."""
        return {
            'video_digest': file_digest(video_path),
            'model_digest': file_digest(self.model_path),
            'class_name': self.CLASS_NAME,
            'class_id': self.class_id,
            'conf_threshold': self.CONF_THRESHOLD,
            'n_frames': n_frames,
            'inference_revision': self.INFERENCE_REVISION,
        }

    def _validate_cached(self, cached: list, cache_path: str) -> None:
        """Raise if a cache entry count matches but its contents are not ball detections."""
        # The frame-count check alone cannot tell a ball cache from a same-length
        # cache of another stage, which would otherwise be used as if it were one.
        for entry in cached:
            if not isinstance(entry, dict):
                raise ValueError(f'Cache at {cache_path} does not hold per-frame detection dicts.')
            for detection in entry.values():
                if not isinstance(detection, BallDetection):
                    raise ValueError(
                        f'Cache at {cache_path} holds {type(detection).__name__} values, '
                        f'expected BallDetection.'
                    )

    def run_detection(
        self,
        frames: list[np.ndarray],
        video_path: str,
        cache_path: str,
    ) -> list[dict[int, BallDetection]]:
        """Run YOLOv8 inference on all frames and filter to the ball class, served from cache when its fingerprint matches."""
        fingerprint = self._cache_fingerprint(video_path, len(frames))
        cached = load_valid_cache(cache_path, fingerprint)
        if cached is not None:
            self._validate_cached(cached, cache_path)
            return cached

        all_detections: list[dict[int, BallDetection]] = []
        for frame in frames:
            # conf is passed explicitly, mirroring PlayerDetector: at 0.3 it was
            # never clipped by ultralytics' 0.25 default NMS floor, but a bare
            # call would silently clip any future threshold below 0.25.
            # filter_detections() keeps its >= gate as a defensive double check.
            results = self.model(frame, conf=self.CONF_THRESHOLD, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            ball_dets = self.filter_detections(detections)

            frame_dict: dict[int, BallDetection] = {}
            if len(ball_dets) > 0:
                best = int(np.argmax(ball_dets.confidence))
                frame_dict[1] = BallDetection(
                    bbox=ball_dets.xyxy[best].tolist(),
                    confidence=float(ball_dets.confidence[best]),
                )
            all_detections.append(frame_dict)

        save_cache_with_meta(all_detections, cache_path, fingerprint)
        return all_detections
