"""
player_detector.py

YOLOv8-based player detection and ByteTrack multi-object tracking.
Produces per-frame dicts of track_id → PlayerTrack, cached to disk to
avoid repeated inference during development.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import supervision as sv
from ultralytics import YOLO

from basketball.utils.io_utils import get_video_metadata
from basketball.cache.cache_utils import file_digest, load_valid_cache, save_cache_with_meta


@dataclass
class PlayerTrack:
    track_id: int
    bbox: list[float]      # [x1, y1, x2, y2]
    confidence: float


class PlayerDetector:
    """Detects and tracks players across video frames using YOLOv8 and ByteTrack."""

    CLASS_NAME = 'player'
    CONF_THRESHOLD = 0.5

    # Increment whenever the inference call, the filtering logic or anything
    # else the parameter keys cannot see changes; same rule as
    # PossessionTracker.LOGIC_REVISION. 1: conf passed through to inference,
    # deliberately invalidating every detection cache written before it.
    INFERENCE_REVISION = 1

    # ByteTrack settings, shared by the tracker construction and the cache
    # fingerprint so the two cannot drift apart. The class constants are the
    # production operating point; the constructor can override them per
    # instance for evaluation sweeps (scripts/run_evaluation.py), and the
    # overrides flow into the fingerprint so sweep caches never collide.
    TRACK_ACTIVATION_THRESHOLD = 0.5
    LOST_TRACK_BUFFER = 30
    MINIMUM_MATCHING_THRESHOLD = 0.8
    MINIMUM_CONSECUTIVE_FRAMES = 2

    def __init__(
        self,
        model_path: str,
        conf_threshold: float | None = None,
        track_activation_threshold: float | None = None,
        lost_track_buffer: int | None = None,
        minimum_matching_threshold: float | None = None,
        minimum_consecutive_frames: int | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.class_id = self._resolve_class_index()
        self.conf_threshold = self.CONF_THRESHOLD if conf_threshold is None else conf_threshold
        self.track_activation_threshold = (
            self.TRACK_ACTIVATION_THRESHOLD if track_activation_threshold is None else track_activation_threshold
        )
        self.lost_track_buffer = self.LOST_TRACK_BUFFER if lost_track_buffer is None else lost_track_buffer
        self.minimum_matching_threshold = (
            self.MINIMUM_MATCHING_THRESHOLD if minimum_matching_threshold is None else minimum_matching_threshold
        )
        self.minimum_consecutive_frames = (
            self.MINIMUM_CONSECUTIVE_FRAMES if minimum_consecutive_frames is None else minimum_consecutive_frames
        )

    def _resolve_class_index(self) -> int:
        """Return the checkpoint's own class index for CLASS_NAME, matched case-insensitively."""
        # The player class index differs between checkpoints (1 in players.pt, 4 in
        # the fy4c2 ball.pt), and a wrong hardcoded index silently yields zero
        # detections, so the index is read from model.names, never assumed.
        for class_id, name in self.model.names.items():
            if name.lower() == self.CLASS_NAME.lower():
                return class_id
        raise ValueError(
            f'No class named {self.CLASS_NAME!r} in this checkpoint; '
            f'available classes: {self.model.names}'
        )

    def filter_detections(self, detections: sv.Detections) -> sv.Detections:
        """Remove non-player detections and boxes below confidence threshold."""
        return detections[
            (detections.class_id == self.class_id) &
            (detections.confidence >= self.conf_threshold)
        ]

    def _cache_fingerprint(self, video_path: str, fps: float, n_frames: int) -> dict:
        """Fingerprint of every input that determines this stage's output, used to invalidate same-length stale caches."""
        return {
            'video_digest': file_digest(video_path),
            'model_digest': file_digest(self.model_path),
            'class_name': self.CLASS_NAME,
            'class_id': self.class_id,
            'conf_threshold': self.conf_threshold,
            'n_frames': n_frames,
            'track_activation_threshold': self.track_activation_threshold,
            'lost_track_buffer': self.lost_track_buffer,
            'minimum_matching_threshold': self.minimum_matching_threshold,
            'minimum_consecutive_frames': self.minimum_consecutive_frames,
            'frame_rate': fps,
            'inference_revision': self.INFERENCE_REVISION,
        }

    def _validate_cached(self, cached: list, cache_path: str) -> None:
        """Raise if a cache entry count matches but its contents are not player tracks."""
        # The frame-count check alone cannot tell a player cache from a same-length
        # cache of another stage, which would otherwise be used as if it were one.
        for entry in cached:
            if not isinstance(entry, dict):
                raise ValueError(f'Cache at {cache_path} does not hold per-frame track dicts.')
            for track in entry.values():
                if not isinstance(track, PlayerTrack):
                    raise ValueError(
                        f'Cache at {cache_path} holds {type(track).__name__} values, '
                        f'expected PlayerTrack.'
                    )

    def run_detection(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        """
        Run YOLOv8 inference at the instance's confidence floor and filter to
        the player class. Passing conf into inference makes the requested
        threshold the real NMS floor; the bare call silently clipped it at
        ultralytics' 0.25 default. At the production 0.5 the detection set is
        unchanged (everything in [0.25, 0.5) was filtered out anyway), but NMS
        now operates on a smaller candidate pool, which can in principle alter
        which overlapping boxes survive.
        """
        # filter_detections() keeps its >= gate as a defensive double check.
        all_detections = []
        for frame in frames:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            all_detections.append(self.filter_detections(detections))
        return all_detections

    def run_tracking(
        self,
        frames: list[np.ndarray],
        video_path: str,
        cache_path: str,
    ) -> list[dict[int, PlayerTrack]]:
        """Run detection and ByteTrack on all frames, served from cache when its fingerprint matches."""
        metadata = get_video_metadata(video_path)
        fps = metadata['fps']

        fingerprint = self._cache_fingerprint(video_path, fps, len(frames))
        cached = load_valid_cache(cache_path, fingerprint)
        if cached is not None:
            self._validate_cached(cached, cache_path)
            return cached

        tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            frame_rate=fps,
        )

        all_tracks: list[dict[int, PlayerTrack]] = []
        for detections in self.run_detection(frames):
            detections = tracker.update_with_detections(detections)

            frame_dict: dict[int, PlayerTrack] = {}
            if detections.tracker_id is not None:
                for i in range(len(detections)):
                    tid = int(detections.tracker_id[i])
                    frame_dict[tid] = PlayerTrack(
                        track_id=tid,
                        bbox=detections.xyxy[i].tolist(),
                        confidence=float(detections.confidence[i]),
                    )
            all_tracks.append(frame_dict)

        save_cache_with_meta(all_tracks, cache_path, fingerprint)
        return all_tracks