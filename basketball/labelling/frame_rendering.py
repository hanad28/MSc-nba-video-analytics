"""
frame_rendering.py

Thin, pure rendering helpers for the per-frame team-classification labelling
notebook: draw every tracked player's bbox with a distinct outline colour
and track_id label on the full frame, crop a small thumbnail for one player,
and encode a frame to PNG bytes for ipywidgets.Image. Deliberately free of
ipywidgets/IPython imports, unlike the notebook that calls these: plain
numpy arrays and bytes in, so this stays unit-testable without a kernel.
"""
from __future__ import annotations

import cv2
import numpy as np

# A fixed palette, indexed by each track's rank among the players visible in
# one frame (not by raw track_id, which grows across the clip and isn't
# compact: two simultaneously-visible players would otherwise collide on
# the same colour whenever their ids happen to be a multiple of the palette
# size apart, defeating the "distinct colour per box" guarantee). 16 entries
# comfortably exceeds the documented 6-12 on-court boxes per frame
# convention (see scripts/label_ground_truth.py's MIN/MAX_EXPECTED_BOXES).
BOX_COLOURS: list[tuple[int, int, int]] = [
    (0, 200, 0), (0, 0, 255), (255, 0, 0), (0, 200, 200),
    (200, 0, 200), (0, 128, 255), (255, 255, 0), (128, 0, 255),
    (0, 255, 128), (255, 0, 128), (128, 255, 0), (255, 128, 0),
    (0, 64, 128), (128, 64, 0), (64, 0, 128), (64, 128, 64),
]


# Same threshold/offset convention as scripts/label_ground_truth.py's
# BANNER_HEIGHT_PX-based label placement, reused verbatim rather than
# invented fresh: cv2.putText's origin is the text baseline, so clamping it
# to max(0, y1 - 8) renders nothing visible for any box whose y1 is below
# roughly the label's own height -- the label must instead drop BELOW the
# top edge when there isn't room above it.
TOP_EDGE_THRESHOLD_PX = 46
LABEL_OFFSET_BELOW_PX = 16
LABEL_OFFSET_ABOVE_PX = -6


def colour_for_rank(rank: int) -> tuple[int, int, int]:
    """A deterministic BGR colour for a player's rank among one frame's visible tracks, cycling through BOX_COLOURS."""
    return BOX_COLOURS[rank % len(BOX_COLOURS)]


def _label_y(y1: int) -> int:
    """The y-origin for a box's track_id label: below the top edge when there isn't room above it, never clamped to an invisible position."""
    return y1 + LABEL_OFFSET_BELOW_PX if y1 < TOP_EDGE_THRESHOLD_PX else y1 + LABEL_OFFSET_ABOVE_PX


def colours_by_track(tracks: dict[int, object]) -> dict[int, tuple[int, int, int]]:
    """Map every track_id in tracks to its colour, assigned by rank in sorted track_id order within this frame."""
    return {track_id: colour_for_rank(rank) for rank, track_id in enumerate(sorted(tracks))}


def draw_tracked_boxes(frame: np.ndarray, tracks: dict[int, object]) -> np.ndarray:
    """Return a copy of frame with every track's full bbox outlined in a distinct colour and its track_id labelled."""
    annotated = frame.copy()
    colours = colours_by_track(tracks)
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        colour = colours[track_id]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            annotated,
            str(track_id),
            (x1, _label_y(y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
        )
    return annotated


def crop_thumbnail(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
    """A small BGR crop of one player's full bbox (the whole box, not classifier.py's jersey-only crop), clamped to the frame."""
    h, w = frame.shape[:2]
    x1 = min(max(int(bbox[0]), 0), w)
    y1 = min(max(int(bbox[1]), 0), h)
    x2 = min(max(int(bbox[2]), 0), w)
    y2 = min(max(int(bbox[3]), 0), h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((10, 10, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def encode_png(frame: np.ndarray) -> bytes:
    """Encode a BGR frame array to PNG bytes, ready for ipywidgets.Image(value=...)."""
    ok, buffer = cv2.imencode('.png', frame)
    if not ok:
        raise ValueError('cv2.imencode failed to encode the frame as PNG.')
    return buffer.tobytes()
