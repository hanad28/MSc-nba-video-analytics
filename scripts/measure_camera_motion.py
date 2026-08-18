"""
measure_camera_motion.py measures the per-frame global camera motion of the three
evaluation clips, so the pan and zoom the single-camera pipeline has to absorb is
a measured quantity rather than an impression. Shi-Tomasi corners tracked by
pyramidal Lucas-Kanade optical flow give point correspondences between each pair
of consecutive frames, and a RANSAC similarity fit reduces each pair to a
horizontal displacement, measured at the image centre, and a scale change.
Unlike the other measurement scripts this one needs no caches, no checkpoints
and no GPU: the clips it reads are committed, so it runs from a fresh clone and
writes straight into results/.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Running this file directly (`python scripts/measure_camera_motion.py`) puts
# scripts/ on sys.path[0], not the repo root, so `basketball` would not be
# importable. Insert the repo root explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basketball.utils.io_utils import get_video_metadata, load_video

CLIPS = ('clip_1', 'clip_2', 'clip_3')
RAW_VIDEO_TEMPLATE = 'data/raw/{clip}.mp4'

OUTPUT_CSV = Path('results/camera_motion/camera_motion.csv')

# One series file per clip alongside the summary. Each row describes the motion
# between a frame and the one before it, indexed by the LATER frame: the row for
# frame f is the displacement from f - 1 to f. Indexing by the later frame is
# what lets the series be joined directly against any other per-frame record,
# such as the Stage 8 mapping outcome in results/homography/.
SERIES_CSV_TEMPLATE = 'results/camera_motion/camera_motion_{clip}.csv'
SERIES_HEADERS = ['frame', 'abs_dx_px', 'scale_change']

HEADERS = [
    'clip', 'width', 'height', 'fps', 'frames', 'duration_s',
    'frame_pairs', 'pairs_estimated',
    'median_abs_dx_px', 'p95_abs_dx_px',
    'median_abs_scale_change', 'max_abs_scale_change',
    'frames_abs_dx_over_10px',
]

# Shi-Tomasi corner detection, re-run on every frame rather than tracking one
# fixed set: a pan continuously replaces what is on screen, so a set chosen once
# would empty out as the camera moves. 400 corners is far more than the two
# point pairs a similarity transform needs, leaving RANSAC a large inlier pool
# after the players and the ball are rejected. The 10 px separation spreads the
# corners over the court instead of clustering them along one painted line.
MAX_CORNERS = 400
CORNER_QUALITY_LEVEL = 0.01
MIN_CORNER_DISTANCE_PX = 10
CORNER_BLOCK_SIZE = 7

# Pyramidal Lucas-Kanade. Three pyramid levels above the base carry the large
# displacements a broadcast pan produces at 1280x720, which a single-level
# window would lose entirely.
LK_WINDOW_SIZE = (21, 21)
LK_MAX_PYRAMID_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

# Forward-backward consistency: a point tracked into the next frame and back
# again must return to within this distance of where it started. The court's
# repeated markings and the crowd both produce confident but wrong matches, and
# those survive the tracker's own status flag.
MAX_FORWARD_BACKWARD_ERROR_PX = 1.0

# RANSAC inlier threshold for the similarity fit. Camera motion is the consensus
# of the static court, so independently moving players are the outliers this
# rejects rather than a signal to fit.
RANSAC_REPROJECTION_THRESHOLD_PX = 3.0

# Below this many surviving correspondences a fit is not attempted at all: a
# similarity transform through a handful of points is recoverable but not
# trustworthy, and a wrong estimate is worse here than a missing one.
MIN_TRACKED_POINTS = 10

# The threshold the reported large-motion frame count is taken at.
LARGE_TRANSLATION_PX = 10.0

# estimateAffinePartial2D samples its candidate subsets at random, so the seed
# is fixed: the same clips must yield the same figures on every run.
RNG_SEED = 0


@dataclass
class ClipMotion:
    """One clip's measured camera motion, alongside the metadata it was measured against."""

    clip: str
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float
    frame_pairs: int
    pairs_estimated: int
    median_abs_dx_px: float
    p95_abs_dx_px: float
    median_abs_scale_change: float
    max_abs_scale_change: float
    frames_abs_dx_over_threshold: int
    series: list[tuple[int, float, float]]


def require_clip(clip: str) -> str:
    """Return one clip's path, raising a clear error when the committed video is absent."""
    path = RAW_VIDEO_TEMPLATE.format(clip=clip)
    if not Path(path).exists():
        raise FileNotFoundError(
            f'{path} not found. The three evaluation clips are committed under '
            f'data/raw/, so an absent one means an incomplete checkout rather than '
            f'a missing generated artefact.'
        )
    return path


def track_points(previous_gray: np.ndarray, current_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return matched (previous, current) point arrays between two frames, or None when too few survive tracking."""
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUALITY_LEVEL,
        minDistance=MIN_CORNER_DISTANCE_PX,
        blockSize=CORNER_BLOCK_SIZE,
    )
    if corners is None or len(corners) < MIN_TRACKED_POINTS:
        return None

    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, corners, None,
        winSize=LK_WINDOW_SIZE, maxLevel=LK_MAX_PYRAMID_LEVEL, criteria=LK_CRITERIA,
    )
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, forward, None,
        winSize=LK_WINDOW_SIZE, maxLevel=LK_MAX_PYRAMID_LEVEL, criteria=LK_CRITERIA,
    )

    round_trip_error = np.linalg.norm(corners.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
    keep = (
        (forward_status.reshape(-1) == 1)
        & (backward_status.reshape(-1) == 1)
        & (round_trip_error <= MAX_FORWARD_BACKWARD_ERROR_PX)
    )
    if int(keep.sum()) < MIN_TRACKED_POINTS:
        return None

    return corners.reshape(-1, 2)[keep], forward.reshape(-1, 2)[keep]


def frame_pair_motion(previous_gray: np.ndarray, current_gray: np.ndarray) -> tuple[float, float] | None:
    """Return one frame pair's (horizontal displacement of the image centre in pixels, scale change as a fraction), or None when no similarity transform could be fitted."""
    matched = track_points(previous_gray, current_gray)
    if matched is None:
        return None

    previous_points, current_points = matched
    transform, _ = cv2.estimateAffinePartial2D(
        previous_points, current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD_PX,
    )
    if transform is None:
        return None

    # Displacement is measured at the image CENTRE, deliberately not read off as
    # the transform's own tx. A partial affine is [[a, -b, tx], [b, a, ty]], and
    # tx is the displacement of the point at the origin, the top-left corner. As
    # soon as the scale differs from 1 the displacement varies across the frame,
    # so tx describes a corner no viewer is watching and overstates the motion by
    # (scale - 1) * width / 2. On clip_2 that is the difference between a median
    # of 1.48 px and one of 0.19 px, and the centre figure is the one an
    # independent windowed phase correlation reproduces.
    height, width = previous_gray.shape[:2]
    centre = np.array([width / 2.0, height / 2.0, 1.0])
    horizontal_displacement = float((transform @ centre)[0] - centre[0])

    # Scale is reported as its distance from 1, so 0.004 is a 0.4% zoom over one
    # frame in either direction, which is what makes the median and the maximum
    # comparable across clips.
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    return horizontal_displacement, abs(scale - 1.0)


def measure_clip(clip: str) -> ClipMotion:
    """Measure one clip's per-frame camera motion and return it alongside the clip's own metadata."""
    path = require_clip(clip)
    metadata = get_video_metadata(path)

    horizontal_translations: list[float] = []
    scale_changes: list[float] = []
    series: list[tuple[int, float, float]] = []
    frame_pairs = 0
    previous_gray = None
    frames = 0

    for frame_index, frame in load_video(path):
        frames += 1
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            frame_pairs += 1
            motion = frame_pair_motion(previous_gray, current_gray)
            if motion is not None:
                horizontal_translation, scale_change = motion
                horizontal_translations.append(horizontal_translation)
                scale_changes.append(scale_change)
                series.append((frame_index, abs(horizontal_translation), scale_change))
        previous_gray = current_gray

    # Magnitudes, not signed values: a pan left and a pan right both move the
    # court under the pipeline by the same amount, and a signed median would
    # report a steady pan and a violent shake as the same near-zero number.
    absolute_translations = np.abs(np.array(horizontal_translations, dtype=float))
    scales = np.array(scale_changes, dtype=float)

    return ClipMotion(
        clip=clip,
        width=metadata['width'],
        height=metadata['height'],
        fps=metadata['fps'],
        frames=frames,
        duration_s=frames / metadata['fps'],
        frame_pairs=frame_pairs,
        pairs_estimated=len(horizontal_translations),
        median_abs_dx_px=float(np.median(absolute_translations)) if len(absolute_translations) else float('nan'),
        p95_abs_dx_px=float(np.percentile(absolute_translations, 95)) if len(absolute_translations) else float('nan'),
        median_abs_scale_change=float(np.median(scales)) if len(scales) else float('nan'),
        max_abs_scale_change=float(scales.max()) if len(scales) else float('nan'),
        frames_abs_dx_over_threshold=int((absolute_translations > LARGE_TRANSLATION_PX).sum()),
        series=series,
    )


def motion_row(motion: ClipMotion) -> list[object]:
    """Return one clip's measured motion as a CSV row."""
    return [
        motion.clip, motion.width, motion.height, round(motion.fps, 4),
        motion.frames, round(motion.duration_s, 4),
        motion.frame_pairs, motion.pairs_estimated,
        round(motion.median_abs_dx_px, 4), round(motion.p95_abs_dx_px, 4),
        round(motion.median_abs_scale_change, 6), round(motion.max_abs_scale_change, 6),
        motion.frames_abs_dx_over_threshold,
    ]


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    """Write a header row and every supplied row to a CSV, creating the output directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print one aligned results table."""
    widths = (
        [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)] if rows
        else [len(header) for header in headers]
    )
    print(f'\n{title}')
    print('  '.join(header.ljust(width) for header, width in zip(headers, widths)))
    for row in rows:
        print('  '.join(str(cell).ljust(width) for cell, width in zip(row, widths)))


def main() -> int:
    """Measure every clip, write the CSV and print both summary tables."""
    cv2.setRNGSeed(RNG_SEED)

    measurements: list[ClipMotion] = []
    for clip in CLIPS:
        print(f'[{clip}] measuring camera motion...')
        measurements.append(measure_clip(clip))

    write_csv(OUTPUT_CSV, HEADERS, [motion_row(motion) for motion in measurements])
    for motion in measurements:
        write_csv(
            Path(SERIES_CSV_TEMPLATE.format(clip=motion.clip)),
            SERIES_HEADERS,
            [[frame, round(abs_dx, 4), round(scale_change, 6)] for frame, abs_dx, scale_change in motion.series],
        )

    print_table(
        'Clip metadata',
        ['clip', 'resolution', 'fps', 'frames', 'duration_s'],
        [
            [m.clip, f'{m.width}x{m.height}', f'{m.fps:.2f}', str(m.frames), f'{m.duration_s:.2f}']
            for m in measurements
        ],
    )
    print_table(
        f'Per-frame camera motion (similarity fit; {LARGE_TRANSLATION_PX:.0f} px threshold)',
        ['clip', 'pairs', 'estimated', 'median_|dx|', 'p95_|dx|', 'median_scale', 'max_scale', f'|dx|>{LARGE_TRANSLATION_PX:.0f}px'],
        [
            [
                m.clip, str(m.frame_pairs), str(m.pairs_estimated),
                f'{m.median_abs_dx_px:.2f}', f'{m.p95_abs_dx_px:.2f}',
                f'{m.median_abs_scale_change:.5f}', f'{m.max_abs_scale_change:.5f}',
                str(m.frames_abs_dx_over_threshold),
            ]
            for m in measurements
        ],
    )

    print()
    print(f'Wrote {OUTPUT_CSV}')
    for motion in measurements:
        print(f'Wrote {SERIES_CSV_TEMPLATE.format(clip=motion.clip)} ({len(motion.series)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
