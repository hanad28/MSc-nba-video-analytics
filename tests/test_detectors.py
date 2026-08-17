"""Unit tests for the detector class-name resolution, class/confidence filters and cached-result handling.

Detector instances are built without calling __init__ so no YOLO weights are
loaded; only the pure resolution, filtering and cache logic is exercised.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import supervision as sv

from basketball.detection.ball_detector import BallDetection, BallDetector
from basketball.detection.player_detector import PlayerDetector, PlayerTrack

PLAYERS_PT_NAMES = {0: 'ball', 1: 'player'}
FY4C2_NAMES = {0: 'Ball', 1: 'Clock', 2: 'Hoop', 3: 'Overlay', 4: 'Player', 5: 'Ref', 6: 'Scoreboard'}


def detections(
    boxes: list[list[float]],
    class_ids: list[int],
    confidences: list[float],
) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        class_id=np.array(class_ids),
        confidence=np.array(confidences, dtype=float),
    )


def resolved_class_id(detector_cls: type, names: dict[int, str]) -> int:
    detector = object.__new__(detector_cls)
    detector.model = SimpleNamespace(names=names)
    return detector._resolve_class_index()


@pytest.fixture
def ball_detector() -> BallDetector:
    detector = object.__new__(BallDetector)
    detector.class_id = 0  # what 'ball' resolves to on the current fy4c2 checkpoint
    return detector


@pytest.fixture
def player_detector() -> PlayerDetector:
    detector = object.__new__(PlayerDetector)
    detector.class_id = 4  # what 'player' resolves to on the current fy4c2 checkpoint
    detector.conf_threshold = PlayerDetector.CONF_THRESHOLD
    detector.track_activation_threshold = PlayerDetector.TRACK_ACTIVATION_THRESHOLD
    detector.lost_track_buffer = PlayerDetector.LOST_TRACK_BUFFER
    detector.minimum_matching_threshold = PlayerDetector.MINIMUM_MATCHING_THRESHOLD
    detector.minimum_consecutive_frames = PlayerDetector.MINIMUM_CONSECUTIVE_FRAMES
    return detector


def test_player_class_resolves_to_1_for_players_pt_style_names():
    assert resolved_class_id(PlayerDetector, PLAYERS_PT_NAMES) == 1


def test_player_class_resolves_to_4_for_fy4c2_style_names():
    assert resolved_class_id(PlayerDetector, FY4C2_NAMES) == 4


def test_ball_class_resolves_to_0_for_fy4c2_style_names():
    assert resolved_class_id(BallDetector, FY4C2_NAMES) == 0


def test_resolution_raises_and_names_the_available_classes_when_no_match():
    with pytest.raises(ValueError, match='available classes.*Hoop'):
        resolved_class_id(PlayerDetector, {0: 'Hoop', 1: 'Ref'})


def test_player_detector_defaults_and_overrides_the_sweep_parameters(monkeypatch):
    import basketball.detection.player_detector as pd_module

    monkeypatch.setattr(pd_module, 'YOLO', lambda path: SimpleNamespace(names={0: 'ball', 1: 'player'}))

    default = PlayerDetector('fake.pt')
    assert default.conf_threshold == PlayerDetector.CONF_THRESHOLD
    assert default.minimum_consecutive_frames == PlayerDetector.MINIMUM_CONSECUTIVE_FRAMES

    swept = PlayerDetector('fake.pt', conf_threshold=0.25, minimum_consecutive_frames=1)
    assert swept.conf_threshold == 0.25
    assert swept.minimum_consecutive_frames == 1
    # Untouched settings keep the production constants.
    assert swept.track_activation_threshold == PlayerDetector.TRACK_ACTIVATION_THRESHOLD
    assert swept.lost_track_buffer == PlayerDetector.LOST_TRACK_BUFFER
    assert swept.minimum_matching_threshold == PlayerDetector.MINIMUM_MATCHING_THRESHOLD


class _StopInference(Exception):
    pass


class _RecordingModel:
    """Records inference call kwargs, then aborts before any result parsing."""

    def __init__(self) -> None:
        self.call_kwargs: list[dict] = []

    def __call__(self, frame: np.ndarray, **kwargs: object) -> list:
        self.call_kwargs.append(dict(kwargs))
        raise _StopInference()


def test_player_inference_passes_the_instance_conf_threshold_to_the_model(player_detector):
    # The bare model(frame) call left ultralytics' default 0.25 NMS floor in
    # charge, silently clipping sub-0.25 sweep thresholds: lowscore_010 and
    # lowscore_025 produced identical tracks. conf must reach inference.
    player_detector.conf_threshold = 0.1
    model = _RecordingModel()
    player_detector.model = model
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(_StopInference):
        player_detector.run_detection([frame])

    assert model.call_kwargs == [{'conf': 0.1, 'verbose': False}]


def test_ball_inference_passes_the_class_conf_threshold_to_the_model(ball_detector, tmp_path):
    video_path = tmp_path / 'clip.mp4'
    video_path.write_bytes(b'fake video bytes')
    model_path = tmp_path / 'ball.pt'
    model_path.write_bytes(b'fake weights')
    ball_detector.model_path = str(model_path)
    model = _RecordingModel()
    ball_detector.model = model
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(_StopInference):
        ball_detector.run_detection([frame], str(video_path), str(tmp_path / 'ball.pkl'))

    assert model.call_kwargs == [{'conf': BallDetector.CONF_THRESHOLD, 'verbose': False}]


def test_detector_fingerprints_carry_the_inference_revision(player_detector, ball_detector, tmp_path):
    video_path = tmp_path / 'clip.mp4'
    video_path.write_bytes(b'fake video bytes')
    model_path = tmp_path / 'weights.pt'
    model_path.write_bytes(b'fake weights')
    player_detector.model_path = str(model_path)
    ball_detector.model_path = str(model_path)

    # A pre-revision sidecar lacks this key entirely, so introducing it
    # invalidates every existing detection cache on first run, deliberately:
    # those caches predate the conf pass-through fix, which fingerprint
    # parameter keys cannot see. Same increment rule as LOGIC_REVISION.
    player_fingerprint = player_detector._cache_fingerprint(str(video_path), 25.0, 3)
    ball_fingerprint = ball_detector._cache_fingerprint(str(video_path), 3)

    assert player_fingerprint['inference_revision'] == PlayerDetector.INFERENCE_REVISION == 1
    assert ball_fingerprint['inference_revision'] == BallDetector.INFERENCE_REVISION == 1


def test_overridden_sweep_parameters_flow_into_the_cache_fingerprint(player_detector, tmp_path):
    video_path = tmp_path / 'clip.mp4'
    video_path.write_bytes(b'fake video bytes')
    model_path = tmp_path / 'ball.pt'
    model_path.write_bytes(b'fake weights')
    player_detector.model_path = str(model_path)
    player_detector.conf_threshold = 0.1
    player_detector.minimum_consecutive_frames = 1

    fingerprint = player_detector._cache_fingerprint(str(video_path), 25.0, 3)

    # Overrides must invalidate caches: identical settings at different values
    # would otherwise serve one configuration's tracks to another.
    assert fingerprint['conf_threshold'] == 0.1
    assert fingerprint['minimum_consecutive_frames'] == 1
    assert fingerprint['track_activation_threshold'] == PlayerDetector.TRACK_ACTIVATION_THRESHOLD


def test_ball_detector_keeps_only_confident_ball_boxes(ball_detector):
    dets = detections(
        [[0, 0, 4, 4], [10, 10, 14, 14], [20, 20, 24, 24]],
        [ball_detector.class_id, ball_detector.class_id, ball_detector.class_id + 1],
        [0.9, 0.2, 0.99],
    )

    filtered = ball_detector.filter_detections(dets)

    assert len(filtered) == 1
    assert filtered.xyxy[0].tolist() == [0, 0, 4, 4]


def test_ball_detector_keeps_a_box_exactly_on_the_threshold(ball_detector):
    dets = detections([[0, 0, 4, 4]], [ball_detector.class_id], [0.3])

    assert len(ball_detector.filter_detections(dets)) == 1


def test_ball_detector_can_filter_everything_out(ball_detector):
    dets = detections([[0, 0, 4, 4]], [ball_detector.class_id + 1], [0.99])

    assert len(ball_detector.filter_detections(dets)) == 0


def test_player_detector_keeps_only_confident_player_boxes(player_detector):
    dets = detections(
        [[0, 0, 10, 20], [30, 0, 40, 20], [60, 0, 70, 20]],
        [player_detector.class_id, player_detector.class_id, player_detector.class_id - 1],
        [0.8, 0.4, 0.99],
    )

    filtered = player_detector.filter_detections(dets)

    assert len(filtered) == 1
    assert filtered.xyxy[0].tolist() == [0, 0, 10, 20]


def test_player_detector_keeps_a_box_exactly_on_the_threshold(player_detector):
    dets = detections([[0, 0, 10, 20]], [player_detector.class_id], [0.5])

    assert len(player_detector.filter_detections(dets)) == 1


def test_ball_detector_returns_a_matching_cache_without_inference(ball_detector, tmp_path):
    from basketball.cache.cache_utils import save_cache_with_meta

    video_path = tmp_path / 'clip.mp4'
    video_path.write_bytes(b'fake video bytes')
    model_path = tmp_path / 'ball.pt'
    model_path.write_bytes(b'fake ball weights')
    ball_detector.model_path = str(model_path)

    cache_path = str(tmp_path / 'ball.pkl')
    cached = [{1: BallDetection(bbox=[0, 0, 4, 4], confidence=0.9)}, {}]
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    save_cache_with_meta(cached, cache_path, ball_detector._cache_fingerprint(str(video_path), len(frames)))

    assert ball_detector.run_detection(frames, str(video_path), cache_path) == cached


def test_player_detector_returns_a_matching_cache_without_inference(player_detector, tmp_path):
    from basketball.cache.cache_utils import save_cache_with_meta
    from basketball.utils.io_utils import write_video

    # run_tracking reads the real fps for the fingerprint, so the video must open.
    video_path = str(tmp_path / 'clip.avi')
    frames = [np.zeros((48, 64, 3), dtype=np.uint8)]
    write_video(frames, video_path, fps=25.0)
    model_path = tmp_path / 'players.pt'
    model_path.write_bytes(b'fake player weights')
    player_detector.model_path = str(model_path)

    cache_path = str(tmp_path / 'players.pkl')
    cached = [{3: PlayerTrack(track_id=3, bbox=[0, 0, 10, 20], confidence=0.8)}]
    save_cache_with_meta(cached, cache_path, player_detector._cache_fingerprint(video_path, 25.0, len(frames)))

    assert player_detector.run_tracking(frames, video_path, cache_path) == cached


def test_ball_detection_dataclass_fields():
    detection = BallDetection(bbox=[1, 2, 3, 4], confidence=0.5)

    assert detection.bbox == [1, 2, 3, 4]
    assert detection.confidence == 0.5


def test_player_track_dataclass_fields():
    player_track = PlayerTrack(track_id=9, bbox=[1, 2, 3, 4], confidence=0.5)

    assert player_track.track_id == 9
    assert player_track.bbox == [1, 2, 3, 4]
