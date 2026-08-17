"""
io_utils.py

Utilities for video I/O: frame loading, video writing, and metadata
retrieval (fps, resolution, frame count).
"""
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np


def get_video_metadata(video_path: str) -> dict:
    """Return fps, width, height, and frame_count metadata for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')

    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()

    # A zero or NaN fps propagates into the tracker frame rate and the output
    # writer, where it degrades results without raising anything.
    if not metadata['fps'] > 0:
        raise IOError(f'Video reports an unusable fps of {metadata["fps"]!r}: {video_path}')

    if metadata['width'] <= 0 or metadata['height'] <= 0:
        raise IOError(
            f'Video reports unusable frame dimensions '
            f'{metadata["width"]}x{metadata["height"]}: {video_path}'
        )

    return metadata


def load_video(video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield (frame_idx, frame) tuples for every frame in a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        raise IOError(f'Video opened but yielded no frames: {video_path}')


def write_video(frames: list, output_path: str, fps: float) -> None:
    """Write a list of frames to a video file at the given fps."""
    if not frames:
        raise ValueError(f'No frames to write to {output_path}')

    if not fps > 0:
        raise ValueError(f'Cannot write {output_path} at a non-positive fps of {fps!r}')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise IOError(f'Cannot open video writer for {output_path} at {width}x{height}, {fps} fps')

    try:
        for frame_idx, frame in enumerate(frames):
            # cv2.VideoWriter drops frames whose shape differs from the writer's
            # size without reporting anything, so mismatches are caught here.
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f'Frame {frame_idx} has shape {frame.shape[:2]}, '
                    f'expected {(height, width)} — cannot write {output_path}'
                )
            out.write(frame)
    finally:
        out.release()
