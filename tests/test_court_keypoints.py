"""Unit tests for CourtKeypoints: the fixed-length output contract, the filtering policy and cache fingerprinting."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import basketball.keypoints.court_keypoints as ck_module
from basketball.keypoints.court_keypoints import CourtKeypoints, Keypoint
from basketball.keypoints.court_template import NUM_KEYPOINTS


class _FakePose:
    """A minimal stand-in for one ultralytics pose result, faking only the model's forward pass."""

    def __init__(self, xy: np.ndarray | None, conf: np.ndarray | None) -> None:
        self.keypoints = None if xy is None else SimpleNamespace(xy=xy, conf=conf)


def full_court(confidence: float = 0.9) -> _FakePose:
    """A result carrying all 18 keypoints at distinct positions and one shared confidence."""
    xy = np.array([[[float(i * 10), float(i * 5)] for i in range(NUM_KEYPOINTS)]])
    conf = np.full((1, NUM_KEYPOINTS), confidence)
    return _FakePose(xy, conf)


def no_instance() -> _FakePose:
    """The undetected-court case: ultralytics returns a Keypoints object whose xy has leading dimension 0, not None."""
    return _FakePose(np.zeros((0, NUM_KEYPOINTS, 2)), np.zeros((0, NUM_KEYPOINTS)))


class _DeviceArray:
    """Stands in for a CUDA tensor: refuses direct numpy conversion, and yields a numpy-backed equivalent only via .cpu()."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise TypeError(
            "can't convert cuda:0 device type tensor to numpy. "
            'Use Tensor.cpu() to copy the tensor to host memory first.'
        )

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, item: object) -> object:
        return _DeviceArray(self._array[item])

    def cpu(self) -> np.ndarray:
        return self._array


class _DevicePose:
    """A pose result whose tensors live off-host until the whole Results object is moved with .cpu(), as ultralytics does."""

    def __init__(self, xy: np.ndarray, conf: np.ndarray) -> None:
        self._xy = xy
        self._conf = conf
        self.keypoints = SimpleNamespace(xy=_DeviceArray(xy), conf=_DeviceArray(conf))

    def cpu(self) -> _FakePose:
        # Ultralytics' Results.cpu() returns a Results whose tensors are all
        # host-side; the numpy-backed _FakePose is the equivalent here.
        return _FakePose(self._xy, self._conf)


def device_court(confidence: float = 0.9) -> _DevicePose:
    """A full-court result whose tensors raise on direct numpy conversion, mimicking a CUDA-resident prediction."""
    xy = np.array([[[float(i * 10), float(i * 5)] for i in range(NUM_KEYPOINTS)]])
    conf = np.full((1, NUM_KEYPOINTS), confidence)
    return _DevicePose(xy, conf)


def fake_checkpoint(tmp_path) -> str:
    """A real file standing in for the checkpoint, since both the load guard and the cache fingerprint read it from disk."""
    # A real file rather than a stubbed os.path.exists: os.path is one shared
    # module object, so patching it here would also blind cache_utils' own
    # sidecar existence check and break every cache test.
    path = tmp_path / 'keypoints.pt'
    path.write_bytes(b'fake checkpoint bytes')
    return str(path)


def loaded_detector(
    monkeypatch,
    result: _FakePose,
    keypoint_threshold: float = 0.5,
    instance_threshold: float = 0.5,
    model_path: str | None = None,
    tmp_path=None,
) -> CourtKeypoints:
    """A detector whose checkpoint loads through the real, unstubbed guard, with only the forward pass faked."""
    if model_path is None:
        model_path = fake_checkpoint(tmp_path)
    monkeypatch.setattr(
        ck_module, 'YOLO',
        lambda path: (lambda *args, **kwargs: [result] * (
            len(args[0]) if args and isinstance(args[0], list) else 1
        )),
    )
    return CourtKeypoints(
        model_path,
        instance_confidence_threshold=instance_threshold,
        keypoint_confidence_threshold=keypoint_threshold,
    )


def frames(n: int = 2) -> list[np.ndarray]:
    return [np.zeros((40, 60, 3), dtype=np.uint8) for _ in range(n)]


# --- the fixed-length output contract -----------------------------------

def test_detect_keypoints_returns_exactly_eighteen_in_index_order(monkeypatch, tmp_path):
    detector = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)

    detected = detector.detect_keypoints(frames(1)[0])

    assert len(detected) == NUM_KEYPOINTS
    assert [keypoint.index for keypoint in detected] == list(range(NUM_KEYPOINTS))


def test_detect_keypoints_returns_eighteen_when_no_court_instance_is_detected(monkeypatch, tmp_path):
    # The empty case must be read from the array's leading dimension: a
    # `keypoints is None` check would miss it entirely.
    detector = loaded_detector(monkeypatch, no_instance(), tmp_path=tmp_path)

    detected = detector.detect_keypoints(frames(1)[0])

    assert len(detected) == NUM_KEYPOINTS
    assert [keypoint.index for keypoint in detected] == list(range(NUM_KEYPOINTS))
    assert all(keypoint.confidence == 0.0 for keypoint in detected)


def test_detect_keypoints_returns_eighteen_when_only_some_are_confident(monkeypatch, tmp_path):
    xy = np.array([[[float(i), float(i)] for i in range(NUM_KEYPOINTS)]])
    conf = np.array([[0.9 if i < 6 else 0.05 for i in range(NUM_KEYPOINTS)]])
    detector = loaded_detector(monkeypatch, _FakePose(xy, conf), tmp_path=tmp_path)

    detected = detector.detect_keypoints(frames(1)[0])

    # Low-confidence points are still present -- discarding them here would
    # answer the pre-registered filtering-policy question before it is measured.
    assert len(detected) == NUM_KEYPOINTS
    assert sum(1 for keypoint in detected if keypoint.confidence >= 0.5) == 6


def test_detect_keypoints_keeps_zero_confidence_points_in_the_parsed_output(monkeypatch, tmp_path):
    # A keypoint the model scored at exactly 0.0 must still occupy its index.
    # Without this the parse loop is never exercised with a zero, and a
    # `if confidence > 0.0` filter added there would pass every other test
    # while silently making the output length variable.
    xy = np.array([[[float(i), float(i)] for i in range(NUM_KEYPOINTS)]])
    conf = np.array([[0.0 if i % 2 else 0.9 for i in range(NUM_KEYPOINTS)]])
    detector = loaded_detector(monkeypatch, _FakePose(xy, conf), tmp_path=tmp_path)

    detected = detector.detect_keypoints(frames(1)[0])

    assert len(detected) == NUM_KEYPOINTS
    assert [keypoint.index for keypoint in detected] == list(range(NUM_KEYPOINTS))
    assert sum(1 for keypoint in detected if keypoint.confidence == 0.0) == NUM_KEYPOINTS // 2


def test_run_detection_returns_eighteen_per_frame_across_batches(monkeypatch, tmp_path):
    detector = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)
    monkeypatch.setattr(CourtKeypoints, 'BATCH_SIZE', 3)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')

    result = detector.run_detection(frames(7), str(video), str(tmp_path / 'keypoints.pkl'))

    assert len(result) == 7
    assert all(len(per_frame) == NUM_KEYPOINTS for per_frame in result)


# --- filter_keypoints ----------------------------------------------------

def test_filter_keypoints_returns_eighteen_and_zeroes_only_sub_threshold(monkeypatch, tmp_path):
    detector = loaded_detector(monkeypatch, full_court(), keypoint_threshold=0.5, tmp_path=tmp_path)
    keypoints = [
        Keypoint(index=i, x=float(i), y=float(i), confidence=0.9 if i < 9 else 0.2)
        for i in range(NUM_KEYPOINTS)
    ]

    filtered = detector.filter_keypoints(keypoints)

    assert len(filtered) == NUM_KEYPOINTS
    assert [keypoint.confidence for keypoint in filtered[:9]] == [0.9] * 9
    assert [keypoint.confidence for keypoint in filtered[9:]] == [0.0] * 9


def test_filter_keypoints_leaves_coordinates_untouched(monkeypatch, tmp_path):
    # A policy that erases its own inputs cannot be compared against another,
    # which is what the pre-registered filtering ablation requires.
    detector = loaded_detector(monkeypatch, full_court(), keypoint_threshold=0.5, tmp_path=tmp_path)
    keypoints = [Keypoint(index=i, x=float(i * 3), y=float(i * 7), confidence=0.1) for i in range(NUM_KEYPOINTS)]

    filtered = detector.filter_keypoints(keypoints)

    assert [(keypoint.x, keypoint.y) for keypoint in filtered] == [
        (float(i * 3), float(i * 7)) for i in range(NUM_KEYPOINTS)
    ]


def test_filter_keypoints_keeps_confidence_exactly_at_the_threshold(monkeypatch, tmp_path):
    detector = loaded_detector(monkeypatch, full_court(), keypoint_threshold=0.5, tmp_path=tmp_path)
    keypoints = [Keypoint(index=i, x=1.0, y=1.0, confidence=0.5) for i in range(NUM_KEYPOINTS)]

    assert all(keypoint.confidence == 0.5 for keypoint in detector.filter_keypoints(keypoints))


def test_a_confident_keypoint_at_the_origin_survives_filtering(monkeypatch, tmp_path):
    # The (0, 0) non-sentinel property. Index 0's template position is the court
    # corner, so a legitimate prediction there must not be read as absence --
    # confidence is the only absence signal.
    detector = loaded_detector(monkeypatch, full_court(), keypoint_threshold=0.5, tmp_path=tmp_path)
    keypoints = [Keypoint(index=i, x=0.0, y=0.0, confidence=0.95) for i in range(NUM_KEYPOINTS)]

    filtered = detector.filter_keypoints(keypoints)

    assert len(filtered) == NUM_KEYPOINTS
    assert all(keypoint.confidence == 0.95 for keypoint in filtered)
    assert all((keypoint.x, keypoint.y) == (0.0, 0.0) for keypoint in filtered)


# --- load_model ----------------------------------------------------------

def test_load_model_raises_naming_a_missing_checkpoint_path():
    detector = CourtKeypoints('models/does_not_exist.pt')

    with pytest.raises(FileNotFoundError, match='models/does_not_exist.pt'):
        detector.load_model()


def test_detect_keypoints_surfaces_the_missing_checkpoint_through_the_load_guard():
    # The guard must remain reachable through the detection path; stubbing the
    # forward pass must never be able to bypass it.
    detector = CourtKeypoints('models/does_not_exist.pt')

    with pytest.raises(FileNotFoundError, match='models/does_not_exist.pt'):
        detector.detect_keypoints(frames(1)[0])


def test_run_detection_names_the_stage_for_a_missing_checkpoint_rather_than_raising_from_the_digest(tmp_path):
    # The real path, which neither test above traverses: one calls load_model()
    # directly and the other never reaches the cache layer. _cache_fingerprint()
    # calls file_digest(model_path), which opens the checkpoint, so computing
    # the fingerprint first raised a bare '[Errno 2] No such file or directory'
    # from cache_utils naming neither the stage nor the cause.
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    detector = CourtKeypoints(str(tmp_path / 'does_not_exist.pt'))

    with pytest.raises(FileNotFoundError) as raised:
        detector.run_detection(frames(1), str(video), str(tmp_path / 'keypoints.pkl'))

    message = str(raised.value)
    assert 'does_not_exist.pt' in message
    assert 'Stage 7' in message
    assert 'Court keypoint weights do not exist' in message
    # The bare OS error is what this test exists to prevent resurfacing.
    assert 'No such file or directory' not in message


def test_a_missing_checkpoint_is_named_even_when_a_valid_cache_exists(monkeypatch, tmp_path):
    # The fingerprint is computed before the cache lookup, so an absent
    # checkpoint failed even on a cache hit, the case a run that should have
    # been served entirely from cache still hit.
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    cache_path = str(tmp_path / 'keypoints.pkl')

    warm = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)
    warm.run_detection(frames(2), str(video), cache_path)
    assert Path(cache_path).exists(), 'fixture must leave a real cache behind'

    detector = CourtKeypoints(str(tmp_path / 'does_not_exist.pt'))

    with pytest.raises(FileNotFoundError, match='Court keypoint weights do not exist'):
        detector.run_detection(frames(2), str(video), cache_path)


def test_a_cache_hit_still_serves_without_re_running_inference(monkeypatch, tmp_path):
    # Loading before the fingerprint must not turn a cache hit into a second
    # inference pass: the model is loaded, but the forward pass is not run.
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    cache_path = str(tmp_path / 'keypoints.pkl')

    detector = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)
    first = detector.run_detection(frames(2), str(video), cache_path)

    calls: list[object] = []
    real_model = detector._model

    def counting_model(*args: object, **kwargs: object) -> object:
        calls.append(args)
        return real_model(*args, **kwargs)

    detector._model = counting_model
    second = detector.run_detection(frames(2), str(video), cache_path)

    assert second == first
    assert calls == [], 'a cache hit must not run the forward pass'


# --- caching -------------------------------------------------------------

def test_cache_is_served_when_the_fingerprint_matches(monkeypatch, tmp_path):
    detector = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    cache_path = str(tmp_path / 'keypoints.pkl')

    first = detector.run_detection(frames(2), str(video), cache_path)
    second = detector.run_detection(frames(2), str(video), cache_path)

    assert first == second


def test_cache_misses_when_the_instance_threshold_changes(monkeypatch, tmp_path):
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    cache_path = str(tmp_path / 'keypoints.pkl')

    checkpoint = fake_checkpoint(tmp_path)
    loaded_detector(monkeypatch, full_court(0.9), instance_threshold=0.5, model_path=checkpoint).run_detection(
        frames(2), str(video), cache_path,
    )
    stored = json.loads((tmp_path / 'keypoints.pkl.meta.json').read_text(encoding='utf-8'))

    loaded_detector(monkeypatch, full_court(0.9), instance_threshold=0.7, model_path=checkpoint).run_detection(
        frames(2), str(video), cache_path,
    )
    rewritten = json.loads((tmp_path / 'keypoints.pkl.meta.json').read_text(encoding='utf-8'))

    assert stored['instance_confidence_threshold'] == 0.5
    assert rewritten['instance_confidence_threshold'] == 0.7


def test_cache_misses_when_the_inference_revision_changes(monkeypatch, tmp_path):
    detector = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    cache_path = str(tmp_path / 'keypoints.pkl')

    detector.run_detection(frames(2), str(video), cache_path)
    monkeypatch.setattr(CourtKeypoints, 'INFERENCE_REVISION', CourtKeypoints.INFERENCE_REVISION + 1)

    fingerprint = detector._cache_fingerprint(str(video), 2)
    stored = json.loads((tmp_path / 'keypoints.pkl.meta.json').read_text(encoding='utf-8'))

    assert fingerprint['inference_revision'] != stored['inference_revision']


def test_a_same_length_cache_of_the_wrong_type_is_rejected(monkeypatch, tmp_path):
    detector = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)
    cache_path = str(tmp_path / 'keypoints.pkl')

    with pytest.raises(ValueError, match='expected Keypoint'):
        detector._validate_cached([['not a keypoint'] * NUM_KEYPOINTS], cache_path)


def test_a_cache_frame_of_the_wrong_length_is_rejected(monkeypatch, tmp_path):
    detector = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)
    cache_path = str(tmp_path / 'keypoints.pkl')
    short = [[Keypoint(index=i, x=0.0, y=0.0, confidence=0.5) for i in range(4)]]

    with pytest.raises(ValueError, match='expected 18'):
        detector._validate_cached(short, cache_path)


def test_the_fingerprint_carries_every_key_that_determines_the_output(monkeypatch, tmp_path):
    detector = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')

    fingerprint = detector._cache_fingerprint(str(video), 5)

    assert set(fingerprint) == {
        'video_digest', 'model_digest', 'instance_confidence_threshold', 'n_frames',
        'inference_revision',
    }


# --- inference call ------------------------------------------------------

def test_the_confidence_threshold_is_passed_into_inference(monkeypatch, tmp_path):
    # A bare call leaves ultralytics' 0.25 default NMS floor in charge, which
    # silently clipped a sub-0.25 threshold on this project's detectors before.
    recorded: dict[str, object] = {}

    def fake_model(*args: object, **kwargs: object) -> list[_FakePose]:
        recorded.update(kwargs)
        return [full_court()]

    monkeypatch.setattr(ck_module, 'YOLO', lambda path: fake_model)

    CourtKeypoints(
        fake_checkpoint(tmp_path), instance_confidence_threshold=0.15,
    ).detect_keypoints(frames(1)[0])

    assert recorded['conf'] == 0.15


def test_parsed_coordinates_are_plain_floats_not_tensors(monkeypatch, tmp_path):
    # Nothing downstream should hold a CUDA tensor or acquire a torch
    # dependency through this stage's output.
    detector = loaded_detector(monkeypatch, full_court(), tmp_path=tmp_path)

    detected = detector.detect_keypoints(frames(1)[0])

    assert all(type(keypoint.x) is float and type(keypoint.y) is float for keypoint in detected)
    assert all(type(keypoint.confidence) is float for keypoint in detected)


# --- a checkpoint without keypoint confidence ----------------------------

def test_a_checkpoint_without_keypoint_confidence_raises_naming_the_path(monkeypatch, tmp_path):
    # Zeros here would be indistinguishable from a genuinely undetected court
    # on every frame: filter_keypoints would zero everything and the annotator
    # would draw nothing, so a whole clip of emptiness would read as a real
    # result. Failing at detection is the only outcome that cannot be mistaken
    # for one.
    checkpoint = fake_checkpoint(tmp_path)
    xy = np.array([[[float(i), float(i)] for i in range(NUM_KEYPOINTS)]])
    detector = loaded_detector(monkeypatch, _FakePose(xy, None), model_path=checkpoint)

    with pytest.raises(ValueError, match='no per-keypoint confidence'):
        detector.detect_keypoints(frames(1)[0])


def test_the_missing_confidence_error_names_the_offending_checkpoint(monkeypatch, tmp_path):
    checkpoint = fake_checkpoint(tmp_path)
    xy = np.array([[[float(i), float(i)] for i in range(NUM_KEYPOINTS)]])
    detector = loaded_detector(monkeypatch, _FakePose(xy, None), model_path=checkpoint)

    with pytest.raises(ValueError, match=re.escape(checkpoint)):
        detector.detect_keypoints(frames(1)[0])


def test_run_detection_surfaces_the_missing_confidence_rather_than_caching_zeros(monkeypatch, tmp_path):
    # The batched path must fail too, otherwise a full clip of zeros would be
    # written to the cache and served on every later run.
    checkpoint = fake_checkpoint(tmp_path)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    cache_path = tmp_path / 'keypoints.pkl'
    xy = np.array([[[float(i), float(i)] for i in range(NUM_KEYPOINTS)]])
    detector = loaded_detector(monkeypatch, _FakePose(xy, None), model_path=checkpoint)

    with pytest.raises(ValueError, match='no per-keypoint confidence'):
        detector.run_detection(frames(2), str(video), str(cache_path))

    assert not cache_path.exists()


# --- device transfer -----------------------------------------------------

def test_device_resident_tensors_are_transferred_before_numpy_conversion(monkeypatch, tmp_path):
    # np.asarray raises outright on a CUDA tensor, so the whole Results object
    # must be moved off-device first. The stand-in refuses direct conversion
    # and yields host arrays only through .cpu(), exactly as a CUDA tensor does.
    detector = loaded_detector(monkeypatch, device_court(), tmp_path=tmp_path)

    detected = detector.detect_keypoints(frames(1)[0])

    assert len(detected) == NUM_KEYPOINTS
    assert [keypoint.index for keypoint in detected] == list(range(NUM_KEYPOINTS))
    assert all(type(keypoint.x) is float for keypoint in detected)
    assert detected[1].x == 10.0


def test_a_device_resident_result_raises_without_the_transfer(monkeypatch, tmp_path):
    # Pins that the stand-in genuinely reproduces the failure: parsing its
    # keypoints directly, bypassing the transfer, must raise.
    pose = device_court()

    with pytest.raises(TypeError, match='Tensor.cpu'):
        np.asarray(pose.keypoints.xy[0], dtype=float)


# --- the threshold split -------------------------------------------------

def test_the_per_keypoint_threshold_is_absent_from_the_fingerprint(monkeypatch, tmp_path):
    # A3 sweeps the per-keypoint policy over cached predictions. Including it
    # here would invalidate the cache on every sweep value and force a full
    # re-inference, making the ablation compare policies over DIFFERENT
    # detections rather than different policies over the same ones.
    detector = loaded_detector(
        monkeypatch, full_court(), keypoint_threshold=0.4, instance_threshold=0.6, tmp_path=tmp_path,
    )
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')

    fingerprint = detector._cache_fingerprint(str(video), 3)

    assert fingerprint['instance_confidence_threshold'] == 0.6
    assert 'keypoint_confidence_threshold' not in fingerprint
    assert not any('keypoint_confidence' in key for key in fingerprint)


def test_changing_only_the_per_keypoint_threshold_still_serves_the_cache(monkeypatch, tmp_path):
    checkpoint = fake_checkpoint(tmp_path)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    cache_path = str(tmp_path / 'keypoints.pkl')

    first = loaded_detector(
        monkeypatch, full_court(), keypoint_threshold=0.3, model_path=checkpoint,
    ).run_detection(frames(2), str(video), cache_path)
    second = loaded_detector(
        monkeypatch, full_court(), keypoint_threshold=0.8, model_path=checkpoint,
    ).run_detection(frames(2), str(video), cache_path)

    assert first == second


def test_the_two_thresholds_are_independent(monkeypatch, tmp_path):
    detector = loaded_detector(
        monkeypatch, full_court(), keypoint_threshold=0.8, instance_threshold=0.2, tmp_path=tmp_path,
    )

    assert detector.instance_confidence_threshold == 0.2
    assert detector.keypoint_confidence_threshold == 0.8
    # filter_keypoints uses the per-keypoint value, not the instance one.
    modest = [Keypoint(index=i, x=1.0, y=1.0, confidence=0.5) for i in range(NUM_KEYPOINTS)]
    assert all(keypoint.confidence == 0.0 for keypoint in detector.filter_keypoints(modest))


def test_a_cache_written_under_inference_revision_1_is_rejected(monkeypatch, tmp_path):
    # Pins the actual 1 -> 2 bump made for the threshold split: a real cache
    # written by the pre-split detector is not served by today's detector.
    from basketball.cache.cache_utils import file_digest, save_cache_with_meta

    checkpoint = fake_checkpoint(tmp_path)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    cache_path = str(tmp_path / 'keypoints.pkl')

    # The pre-split fingerprint: the old single key name, and inference_revision
    # hardcoded to the OLD value rather than read from the class, so this is a
    # real leftover cache rather than whatever the class says today.
    stale_fingerprint = {
        'video_digest': file_digest(str(video)),
        'model_digest': file_digest(checkpoint),
        'confidence_threshold': 0.5,
        'n_frames': 2,
        'inference_revision': 1,
    }
    sentinel = [[Keypoint(index=i, x=-1.0, y=-1.0, confidence=1.0) for i in range(NUM_KEYPOINTS)]] * 2
    save_cache_with_meta(sentinel, cache_path, stale_fingerprint)

    detector = loaded_detector(monkeypatch, full_court(), model_path=checkpoint)
    result = detector.run_detection(frames(2), str(video), cache_path)

    # Regenerated, not served: the sentinel's -1.0 coordinates are gone.
    assert all(keypoint.x != -1.0 for per_frame in result for keypoint in per_frame)
