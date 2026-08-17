"""
ground_truth.py

Ground truth annotation I/O in MOT Challenge CSV format (per line:
frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z; 1-indexed frames,
top-left+width+height boxes). GTAnnotation uses the pipeline's convention
of 0-indexed frames and [x1, y1, x2, y2] boxes; both conversions happen
in load_mot_annotations() and save_mot_annotations().
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

MOT_COLUMNS = 10


@dataclass
class GTAnnotation:
    frame: int             # 0-indexed
    track_id: int
    bbox: list[float]      # [x1, y1, x2, y2]


def load_mot_annotations(path: str) -> list[GTAnnotation]:
    """Load MOT Challenge CSV rows as GTAnnotations, converting to 0-indexed frames and [x1, y1, x2, y2] boxes."""
    if not Path(path).exists():
        raise FileNotFoundError(f'Annotation file does not exist: {path}')

    annotations: list[GTAnnotation] = []
    with open(path, 'r', newline='') as f:
        for line_number, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue

            if len(row) != MOT_COLUMNS:
                raise ValueError(
                    f'{path} line {line_number}: expected {MOT_COLUMNS} MOT columns '
                    f'(frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z), got {len(row)}.'
                )

            try:
                frame = int(row[0])
                track_id = int(row[1])
                left, top, width, height = (float(value) for value in row[2:6])
            except ValueError as error:
                raise ValueError(f'{path} line {line_number}: non-numeric field ({error}).') from error

            if frame < 1:
                raise ValueError(f'{path} line {line_number}: MOT frames are 1-indexed, got frame {frame}.')

            if width <= 0 or height <= 0:
                raise ValueError(
                    f'{path} line {line_number}: box dimensions must be positive, got {width}x{height}.'
                )

            annotations.append(GTAnnotation(
                frame=frame - 1,
                track_id=track_id,
                bbox=[left, top, left + width, top + height],
            ))

    return annotations


def save_mot_annotations(annotations: list[GTAnnotation], path: str) -> None:
    """Save GTAnnotations as MOT Challenge CSV rows, converting to 1-indexed frames and top-left+width+height boxes."""
    rows: list[list[float | int]] = []
    for index, annotation in enumerate(annotations):
        if len(annotation.bbox) != 4:
            raise ValueError(
                f'Annotation {index}: bbox must be [x1, y1, x2, y2], got {annotation.bbox}.'
            )

        if annotation.frame < 0:
            raise ValueError(
                f'Annotation {index}: frames are 0-indexed and cannot be negative, got {annotation.frame}.'
            )

        x1, y1, x2, y2 = annotation.bbox
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            raise ValueError(
                f'Annotation {index}: bbox {annotation.bbox} has non-positive dimensions '
                f'({width}x{height}) and would not survive a load round-trip.'
            )

        # conf=1 marks the entry active and x/y/z are unused in 2D MOT ground
        # truth, conventionally written as -1.
        rows.append([annotation.frame + 1, annotation.track_id, x1, y1, width, height, 1, -1, -1, -1])

    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
