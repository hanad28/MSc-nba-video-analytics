"""Unit tests for video I/O helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from basketball.utils.io_utils import get_video_metadata, load_video, write_video


@pytest.fixture
def sample_frames() -> list[np.ndarray]:
    return [np.full((32, 48, 3), i * 20, dtype=np.uint8) for i in range(5)]


@pytest.fixture
def sample_video(tmp_path, sample_frames: list[np.ndarray]) -> str:
    path = str(tmp_path / 'clip.avi')
    write_video(sample_frames, path, fps=25.0)
    return path


def test_write_video_creates_the_file_and_parent_directories(tmp_path, sample_frames):
    path = tmp_path / 'nested' / 'out' / 'clip.avi'

    write_video(sample_frames, str(path), fps=30.0)

    assert path.exists()
    assert path.stat().st_size > 0


def test_get_video_metadata_reports_geometry_and_fps(sample_video, sample_frames):
    metadata = get_video_metadata(sample_video)

    assert metadata['width'] == 48
    assert metadata['height'] == 32
    assert metadata['fps'] == pytest.approx(25.0, abs=0.5)
    assert metadata['frame_count'] == len(sample_frames)


def test_get_video_metadata_raises_for_an_unopenable_path(tmp_path):
    with pytest.raises(IOError):
        get_video_metadata(str(tmp_path / 'missing.avi'))


def test_load_video_yields_indexed_frames(sample_video, sample_frames):
    loaded = list(load_video(sample_video))

    assert [idx for idx, _ in loaded] == list(range(len(sample_frames)))
    assert all(frame.shape == (32, 48, 3) for _, frame in loaded)


def test_load_video_raises_for_an_unopenable_path(tmp_path):
    with pytest.raises(IOError):
        list(load_video(str(tmp_path / 'missing.avi')))


def test_write_video_then_load_video_round_trip(tmp_path):
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
    path = str(tmp_path / 'round_trip.avi')

    write_video(frames, path, fps=10.0)
    loaded = [frame for _, frame in load_video(path)]

    assert len(loaded) == len(frames)


def test_written_video_is_readable_by_opencv_directly(sample_video):
    cap = cv2.VideoCapture(sample_video)
    try:
        assert cap.isOpened()
    finally:
        cap.release()
