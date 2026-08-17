"""Unit tests for basketball/labelling/frame_rendering.py's pure drawing/encoding
helpers. These are plain numpy/OpenCV functions (no ipywidgets/IPython import),
so, unlike the notebook that calls them, they are safe and worthwhile to
unit test directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from basketball.labelling.frame_rendering import (
    BOX_COLOURS,
    TOP_EDGE_THRESHOLD_PX,
    _label_y,
    colour_for_rank,
    colours_by_track,
    crop_thumbnail,
    draw_tracked_boxes,
    encode_png,
)


def track(bbox: tuple[float, ...]) -> SimpleNamespace:
    return SimpleNamespace(bbox=list(bbox))


def test_colour_for_rank_is_deterministic_and_cycles():
    assert colour_for_rank(1) == colour_for_rank(1)
    assert colour_for_rank(0) != colour_for_rank(1)


def test_colours_by_track_assigns_by_rank_not_by_raw_track_id():
    # Track ids are not compact and don't start at 0 -- colour must be
    # assigned by each track's rank among the tracks visible in this frame
    # (sorted track_id order), not by track_id itself modulo the palette
    # size, which would collide whenever two ids are a multiple of the
    # palette size apart (e.g. ids 5 and 21 with an 16-entry, or 8-entry,
    # palette).
    colours = colours_by_track({21: track((0, 0, 1, 1)), 5: track((0, 0, 1, 1)), 13: track((0, 0, 1, 1))})

    assert colours[5] == colour_for_rank(0)   # lowest id -> rank 0
    assert colours[13] == colour_for_rank(1)
    assert colours[21] == colour_for_rank(2)


def test_colours_by_track_gives_every_track_a_distinct_colour_beyond_eight_simultaneous_tracks():
    # Regression test: BOX_COLOURS used to have 8 entries indexed by raw
    # track_id % 8, so two of these ids (spaced 8 apart, e.g. 3 and 11)
    # collided even though both are visible in the same frame at once.
    track_ids = [3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91]  # 12 tracks, every pair 8 apart
    assert len(track_ids) > 8

    colours = colours_by_track({tid: track((0, 0, 1, 1)) for tid in track_ids})

    assert len(colours) == len(track_ids)
    assert len(set(colours.values())) == len(track_ids)  # no two tracks share a colour


def test_box_colours_palette_has_at_least_sixteen_distinct_entries():
    # 16 comfortably exceeds the documented 6-12 on-court boxes per frame.
    assert len(BOX_COLOURS) >= 16
    assert len(set(BOX_COLOURS)) == len(BOX_COLOURS)  # every palette entry is itself distinct


# --- Bug: label position must not be clamped to an invisible y=0 ---------

def test_label_y_is_not_clamped_to_an_invisible_position_at_the_top_edge():
    # cv2.putText's origin is the text baseline, so the old
    # max(0, y1 - 8) clamp rendered nothing visible for a box whose y1 is 0
    # (or any y1 below roughly the label's own height) -- the label must
    # drop below the top edge instead, same convention as
    # scripts/label_ground_truth.py's BANNER_HEIGHT_PX-based placement.
    label_y = _label_y(0)

    assert label_y > 0     # not clamped to the invisible y=0 origin
    assert label_y > 8     # comfortably below the frame's top edge, not a sliver


def test_label_y_drops_below_the_top_edge_below_the_threshold():
    assert _label_y(TOP_EDGE_THRESHOLD_PX - 1) > TOP_EDGE_THRESHOLD_PX - 1


def test_label_y_stays_above_the_box_when_there_is_room():
    label_y = _label_y(200)

    assert label_y < 200  # still drawn above the box's top edge, as before


def test_draw_tracked_boxes_does_not_mutate_the_input_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()

    draw_tracked_boxes(frame, {1: track((10, 10, 40, 60))})

    assert np.array_equal(frame, original)


def test_draw_tracked_boxes_draws_something_for_every_track():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = draw_tracked_boxes(frame, {1: track((10, 10, 40, 60)), 2: track((60, 10, 90, 60))})

    assert annotated.shape == frame.shape
    assert annotated.any()  # something was drawn; no longer all-zero


def test_draw_tracked_boxes_handles_no_tracks():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    annotated = draw_tracked_boxes(frame, {})

    assert np.array_equal(annotated, frame)


def test_crop_thumbnail_returns_the_expected_shape():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    crop = crop_thumbnail(frame, [10, 20, 40, 60])

    assert crop.shape == (40, 30, 3)


def test_crop_thumbnail_clamps_a_bbox_extending_past_the_frame_edge():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    crop = crop_thumbnail(frame, [80, 0, 150, 30])

    assert crop.shape == (30, 20, 3)  # width clipped to 100 - 80


def test_crop_thumbnail_returns_a_placeholder_for_a_degenerate_box():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    crop = crop_thumbnail(frame, [10, 10, 10, 10])

    assert crop.shape == (10, 10, 3)
    assert not crop.any()


def test_encode_png_round_trips_through_cv2_imdecode():
    frame = np.full((20, 30, 3), 128, dtype=np.uint8)

    encoded = encode_png(frame)
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert isinstance(encoded, bytes)
    assert decoded.shape == frame.shape
    assert np.array_equal(decoded, frame)
