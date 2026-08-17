"""
court_keypoints.py

Detects the 18 court landmarks in each frame using a fine-tuned YOLOv8 pose
model, supplying the image-to-template correspondences Homography consumes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from basketball.cache.cache_utils import file_digest, load_valid_cache, save_cache_with_meta
from basketball.keypoints.court_template import NUM_KEYPOINTS


@dataclass
class Keypoint:
    index: int
    x: float
    y: float
    confidence: float


class CourtKeypoints:
    """
    Detects the 18 court landmarks per frame with a fine-tuned YOLOv8 pose
    model, always returning one Keypoint per index so a landmark's identity is
    carried by its position in the list rather than by its presence.
    """

    # Increment whenever the inference call, the parsing or anything else the
    # parameter keys cannot see changes; the standing rule shared with both
    # detectors' INFERENCE_REVISION and PossessionTracker.LOGIC_REVISION.
    # 1 -> 2: confidence_threshold split into an instance-level and a
    # per-keypoint threshold, narrowing the meaning of the value that reaches
    # inference, so a cache written under revision 1 must not be served.
    INFERENCE_REVISION = 2

    # Unswept starting value: frames per inference batch. A class constant
    # rather than a config key precisely because it is untuned: a config key
    # would imply a measured choice.
    BATCH_SIZE = 20

    def __init__(
        self,
        model_path: str,
        instance_confidence_threshold: float = 0.5,
        keypoint_confidence_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        # Two distinct quantities, deliberately not one: the instance
        # threshold gates whether a court is detected at all and determines
        # the cached predictions, while the per-keypoint threshold gates
        # individual landmarks after retrieval.
        self.instance_confidence_threshold = instance_confidence_threshold
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        self._model: YOLO | None = None

    def load_model(self) -> None:
        """Load the fine-tuned YOLOv8 pose checkpoint, raising a clear error naming the path if it is absent."""
        # Checked here rather than left to ultralytics, whose own failure for a
        # missing path surfaces much deeper and names the cause less clearly.
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f'Court keypoint weights do not exist: {self.model_path}. '
                f'Train the pose model or correct the path before running Stage 7.'
            )

        self._model = YOLO(self.model_path)

    def _require_model(self) -> YOLO:
        """Return the loaded model, loading it on first use so callers need not sequence the load themselves."""
        if self._model is None:
            self.load_model()
        return self._model

    def _parse_result(self, result: object) -> list[Keypoint]:
        """Convert one ultralytics pose result into exactly 18 Keypoints in index order, with absent landmarks at confidence 0.0."""
        # The whole Results object is moved off-device once rather than each
        # tensor separately: np.asarray raises outright on a CUDA tensor, and
        # one transfer cannot be left partially applied the way per-tensor
        # calls can. Guarded because a numpy-backed stand-in has no .cpu().
        if hasattr(result, 'cpu'):
            result = result.cpu()

        keypoints = getattr(result, 'keypoints', None)

        # An undetected court yields a Keypoints object whose xy has leading
        # dimension 0, NOT None, so emptiness is read from the array's shape.
        # A `keypoints is None` check alone would miss that case and index into
        # an empty tensor.
        if keypoints is None or getattr(keypoints, 'xy', None) is None or len(keypoints.xy) == 0:
            return [Keypoint(index=i, x=0.0, y=0.0, confidence=0.0) for i in range(NUM_KEYPOINTS)]

        # Plain floats at this boundary: nothing downstream should hold a CUDA
        # tensor or acquire a torch dependency through this stage's output.
        xy = np.asarray(keypoints.xy[0], dtype=float)
        conf = keypoints.conf

        # Raised rather than defaulted to zeros: this stage's contract is
        # fixed-length output filtered by confidence, so a checkpoint that
        # supplies no confidence cannot satisfy it. Zeros would make every
        # frame read as an undetected court: filter_keypoints would zero
        # everything and the annotator would draw nothing, producing a whole
        # clip of silent, plausible-looking emptiness. A warning would scroll
        # past, which is exactly how ultralytics' per-file corrupt-label
        # warnings hid the loss of 25% of this model's training set.
        if conf is None:
            raise ValueError(
                f'Checkpoint {self.model_path} provides no per-keypoint confidence, '
                f'so it cannot be used by this stage: fixed-length output filtered '
                f'by confidence is Stage 7\'s contract. Train or select a pose '
                f'checkpoint that emits keypoint confidences.'
            )

        confidences = np.asarray(conf[0], dtype=float)

        parsed: list[Keypoint] = []
        for index in range(NUM_KEYPOINTS):
            if index < len(xy):
                x, y = float(xy[index][0]), float(xy[index][1])
                confidence = float(confidences[index]) if index < len(confidences) else 0.0
            else:
                # A checkpoint predicting fewer than 18 points must still yield
                # 18 entries: the index is the identity, so the list length is
                # an invariant rather than a detail of the model's output.
                x, y, confidence = 0.0, 0.0, 0.0
            parsed.append(Keypoint(index=index, x=x, y=y, confidence=confidence))
        return parsed

    def detect_keypoints(self, frame: np.ndarray) -> list[Keypoint]:
        """Detect the 18 court landmarks in a single frame, always returning one Keypoint per index."""
        model = self._require_model()
        # conf gates the court INSTANCE, a different quantity from the
        # per-keypoint threshold filter_keypoints() applies; passing it
        # explicitly stops ultralytics' 0.25 default NMS floor silently
        # clipping a lower threshold, as it did on this project's detectors.
        result = model(frame, conf=self.instance_confidence_threshold, verbose=False)[0]
        return self._parse_result(result)

    def filter_keypoints(self, keypoints: list[Keypoint]) -> list[Keypoint]:
        """Return all 18 keypoints with sub-threshold confidences zeroed and every coordinate left untouched."""
        # Entries are never removed and coordinates never erased: the pre-
        # registered filtering-policy ablation compares policies against the
        # unfiltered predictions, and a policy that destroys its own input
        # cannot be compared against another.
        return [
            Keypoint(
                index=keypoint.index,
                x=keypoint.x,
                y=keypoint.y,
                confidence=(
                    keypoint.confidence
                    if keypoint.confidence >= self.keypoint_confidence_threshold else 0.0
                ),
            )
            for keypoint in keypoints
        ]

    def _cache_fingerprint(self, video_path: str, n_frames: int) -> dict:
        """Fingerprint of every input that determines this stage's output, used to invalidate same-length stale caches."""
        return {
            'video_digest': file_digest(video_path),
            'model_digest': file_digest(self.model_path),
            # The INSTANCE threshold only: it is passed to inference and so
            # determines the cached predictions. The per-keypoint threshold
            # is deliberately absent: filter_keypoints() applies it after
            # retrieval, and including it would force a full re-inference for
            # every value the filtering-policy ablation sweeps, defeating the
            # point of sweeping over cached predictions.
            'instance_confidence_threshold': self.instance_confidence_threshold,
            'n_frames': n_frames,
            'inference_revision': self.INFERENCE_REVISION,
        }

    def _validate_cached(self, cached: list, cache_path: str) -> None:
        """Raise if a cache entry count matches but its contents are not per-frame lists of 18 keypoints."""
        # A matching frame count does not prove a cache holds keypoints rather
        # than some other stage's same-length output.
        for entry in cached:
            if not isinstance(entry, list):
                raise ValueError(f'Cache at {cache_path} does not hold per-frame keypoint lists.')
            if len(entry) != NUM_KEYPOINTS:
                raise ValueError(
                    f'Cache at {cache_path} holds a frame with {len(entry)} keypoints, '
                    f'expected {NUM_KEYPOINTS}.'
                )
            for keypoint in entry:
                if not isinstance(keypoint, Keypoint):
                    raise ValueError(
                        f'Cache at {cache_path} holds {type(keypoint).__name__} values, '
                        f'expected Keypoint.'
                    )

    def run_detection(
        self,
        frames: list[np.ndarray],
        video_path: str,
        cache_path: str,
    ) -> list[list[Keypoint]]:
        """Detect court keypoints across all frames in batches, served from cache when its fingerprint matches."""
        # Loaded before the fingerprint, not after. _cache_fingerprint() calls
        # file_digest(self.model_path), which opens the checkpoint directly, so
        # an absent one raised a bare '[Errno 2] No such file or directory'
        # from cache_utils naming neither the stage nor the cause, and did so
        # even when a valid cache existed, since the fingerprint is computed
        # before the lookup. Loading first means load_model()'s stage-naming
        # error is what surfaces, at the point the checkpoint is required.
        model = self._require_model()

        fingerprint = self._cache_fingerprint(video_path, len(frames))
        cached = load_valid_cache(cache_path, fingerprint)
        if cached is not None:
            self._validate_cached(cached, cache_path)
            return cached

        all_keypoints: list[list[Keypoint]] = []
        for start in range(0, len(frames), self.BATCH_SIZE):
            batch = frames[start:start + self.BATCH_SIZE]
            # Batched rather than per frame: the pose model gains materially
            # more from batching than the box detectors do.
            results = model(batch, conf=self.instance_confidence_threshold, verbose=False)
            all_keypoints.extend(self._parse_result(result) for result in results)

        save_cache_with_meta(all_keypoints, cache_path, fingerprint)
        return all_keypoints
