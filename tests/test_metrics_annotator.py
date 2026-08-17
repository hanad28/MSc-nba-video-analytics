"""Unit tests for Stage 9's MetricsAnnotator (basketball/annotators/metrics_annotator.py)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from basketball.annotators.event_annotator import EventAnnotator
from basketball.annotators.metrics_annotator import MetricsAnnotator
from basketball.annotators.minimap_annotator import MinimapAnnotator
from basketball.annotators.possession_annotator import PossessionAnnotator
from basketball.detection.player_detector import PlayerTrack
from basketball.metrics.player_metrics import TrackSpeed

FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
BACKGROUND = 60


def frames(count: int, height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> list[np.ndarray]:
    """A list of uniform BGR frames, distinguishable from anything the annotator draws."""
    return [np.full((height, width, 3), BACKGROUND, dtype=np.uint8) for _ in range(count)]


def track(track_id: int = 1, x: float = 600.0, y: float = 300.0) -> PlayerTrack:
    """A player box near the centre of a broadcast frame."""
    return PlayerTrack(track_id=track_id, bbox=[x, y, x + 60.0, y + 160.0], confidence=0.9)


# --- frame handling ------------------------------------------------------

@pytest.mark.parametrize('name,index', [('player-track', 1), ('distance', 2), ('speed', 3)])
def test_a_frame_count_mismatch_raises_naming_both_counts(name, index):
    supplied = [[{}, {}], [{}, {}], [{}, {}]]
    supplied[index - 1] = [{}]

    with pytest.raises(ValueError, match=f'1 {name} frames for 2 video frames'):
        MetricsAnnotator().draw(frames(2), *supplied)


def test_the_callers_frames_are_never_mutated():
    originals = frames(2)
    untouched = [frame.copy() for frame in originals]
    tracks = [{1: track()}, {1: track()}]

    MetricsAnnotator().draw(
        originals, tracks, [{1: 0.0}, {1: 5.0}], [{}, {1: TrackSpeed(4.2, 1, 1)}],
    )

    for original, reference in zip(originals, untouched):
        assert np.array_equal(original, reference), 'draw() must copy, never mutate in place'


def test_output_length_and_shape_match_the_input():
    source = frames(3)
    tracks = [{1: track()} for _ in range(3)]

    output = MetricsAnnotator().draw(source, tracks, [{1: 1.0}] * 3, [{}] * 3)

    assert len(output) == 3
    assert all(a.shape == b.shape for a, b in zip(output, source))


# --- what is drawn -------------------------------------------------------

def test_a_track_with_no_speed_draws_no_speed_line():
    # The gap rule suppresses speed where the rate is unknown, and a 0.0 would
    # be indistinguishable from a stationary player.
    annotator = MetricsAnnotator()
    tracks = [{1: track()}]

    without = annotator.draw(frames(1), tracks, [{1: 12.0}], [{}])[0]
    with_speed = annotator.draw(frames(1), tracks, [{1: 12.0}], [{1: TrackSpeed(4.2, 1, 1)}])[0]

    assert not np.array_equal(without, with_speed)
    # The distance line is still drawn on the speedless frame.
    assert np.any(without != BACKGROUND)


def test_a_track_with_neither_metric_draws_nothing():
    output = MetricsAnnotator().draw(frames(1), [{1: track()}], [{}], [{}])[0]

    assert np.array_equal(output, frames(1)[0])


def test_distance_is_drawn_once_a_track_has_a_value():
    output = MetricsAnnotator().draw(frames(1), [{1: track()}], [{1: 7.5}], [{}])[0]

    assert np.any(output != BACKGROUND)


def test_the_text_sits_below_the_players_box_not_in_a_claimed_corner():
    # PossessionAnnotator owns the top-left and EventAnnotator the top-right,
    # both drawn earlier, so attaching the text to the player keeps it clear of
    # them and makes ownership unambiguous. This fixture's player is mid-frame
    # and so is clear of the minimap's bottom-left panel too, but that is a
    # property of where THIS player stands, not of the placement rule: the
    # minimap composites afterwards and overwrites whatever is beneath it, which
    # is what reserved_region exists for. See the reserved-region tests below.
    output = MetricsAnnotator().draw(
        frames(1), [{1: track(x=600.0, y=300.0)}], [{1: 9.9}], [{1: TrackSpeed(4.2, 1, 1)}],
    )[0]

    height, width = output.shape[:2]
    assert np.all(output[:height // 4, :width // 4] == BACKGROUND), 'top-left is the possession caption'
    assert np.all(output[:height // 4, 3 * width // 4:] == BACKGROUND), 'top-right is the event captions'
    assert np.all(output[3 * height // 4:, :width // 4] == BACKGROUND), 'bottom-left is the minimap'
    # Drawn below the box's foot, which sits at y = 460 for this fixture.
    assert np.any(output[460:520, 560:700] != BACKGROUND)


def test_two_tracks_are_labelled_separately():
    tracks = [{1: track(1, x=300.0), 2: track(2, x=900.0)}]

    output = MetricsAnnotator().draw(
        frames(1), tracks, [{1: 5.0, 2: 9.0}], [{1: TrackSpeed(3.0, 1, 1), 2: TrackSpeed(6.0, 1, 1)}],
    )[0]

    left = output[:, :FRAME_WIDTH // 2]
    right = output[:, FRAME_WIDTH // 2:]
    assert np.any(left != BACKGROUND) and np.any(right != BACKGROUND)


def test_different_speeds_render_differently():
    annotator = MetricsAnnotator()
    tracks = [{1: track()}]

    slow = annotator.draw(frames(1), tracks, [{1: 5.0}], [{1: TrackSpeed(2.0, 1, 1)}])[0]
    fast = annotator.draw(frames(1), tracks, [{1: 5.0}], [{1: TrackSpeed(8.0, 1, 1)}])[0]

    assert not np.array_equal(slow, fast)


# --- precision -----------------------------------------------------------

def test_speed_and_distance_are_rendered_to_one_decimal():
    # Stage 7 measured ~5 cm of localisation error, so two decimals on a speed
    # claims precision the measurement does not have.
    assert MetricsAnnotator.SPEED_DECIMALS == 1
    assert MetricsAnnotator.DISTANCE_DECIMALS == 1


def test_speeds_differing_below_the_rendered_precision_look_identical():
    # The consequence of the decimal choice: 5.83 and 5.84 m/s are the same
    # measurement at this error budget and must not be drawn as different.
    annotator = MetricsAnnotator()
    tracks = [{1: track()}]

    first = annotator.draw(frames(1), tracks, [{1: 5.0}], [{1: TrackSpeed(5.83, 1, 1)}])[0]
    second = annotator.draw(frames(1), tracks, [{1: 5.0}], [{1: TrackSpeed(5.84, 1, 1)}])[0]

    assert np.array_equal(first, second)


# --- edge cases ----------------------------------------------------------

def text_bands(output: np.ndarray) -> list[tuple[int, int]]:
    """Return the (first, last) row of each contiguous band of drawn pixels, one band per rendered text line."""
    drawn = sorted({int(row) for row in np.where(output.any(axis=2).any(axis=1))[0]})
    bands: list[list[int]] = []
    for row in drawn:
        if bands and row - bands[-1][-1] <= 1:
            bands[-1].append(row)
        else:
            bands.append([row])
    return [(band[0], band[-1]) for band in bands]


def test_a_box_at_the_frame_edge_keeps_its_text_inside():
    # Clamped rather than drawn off-frame, matching how every other annotator
    # handles a caption near an edge.
    tracks = [{1: track(x=float(FRAME_WIDTH - 20), y=float(FRAME_HEIGHT - 40))}]

    output = MetricsAnnotator().draw(
        frames(1), tracks, [{1: 3.0}], [{1: TrackSpeed(4.0, 1, 1)}],
    )[0]

    assert output.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert np.any(output != BACKGROUND), 'the text must still land inside the frame'


@pytest.mark.parametrize('box_top,box_bottom,description', [
    (300.0, 460.0, 'mid-frame, block below the box'),
    (560.0, float(FRAME_HEIGHT), 'box reaching the bottom edge'),
    (540.0, 700.0, 'foot point at 700, the row the old clamp collapsed at'),
    (559.0, float(FRAME_HEIGHT - 1), 'foot point on the last row'),
])
def test_the_two_lines_never_share_a_row(box_top, box_bottom, description):
    # The defect this pins: each line was clamped independently to
    # height - baseline - 1, so once both would fall past the bottom edge they
    # resolved to the SAME row and the second putText overwrote the first as an
    # illegible smear. Routine on this footage: a near-sideline player's box
    # commonly reaches the bottom of the image.
    #
    # Drawn on a black frame so any non-zero pixel is text, and the bands are
    # counted rather than merely asserting 'something was drawn', which is
    # what let the collapse pass before.
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    tracks = [{1: PlayerTrack(track_id=1, bbox=[600.0, box_top, 660.0, box_bottom], confidence=0.9)}]

    output = MetricsAnnotator().draw(
        [frame], tracks, [{1: 12.3}], [{1: TrackSpeed(4.2, 1, 1)}],
    )[0]

    bands = text_bands(output)
    assert len(bands) == 2, f'expected two separate text lines {description}, got {len(bands)}'
    # And they are genuinely apart, not merely two bands of one smeared row.
    assert bands[1][0] > bands[0][1], 'the second line must start below the first'


def test_the_block_flips_above_the_box_when_there_is_no_room_below():
    # A player at the bottom of the frame always has room above them (their
    # own box occupies it), whereas below them there is none by definition.
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    box_top, box_bottom = 560.0, float(FRAME_HEIGHT)
    tracks = [{1: PlayerTrack(track_id=1, bbox=[600.0, box_top, 660.0, box_bottom], confidence=0.9)}]

    output = MetricsAnnotator().draw(
        [frame], tracks, [{1: 12.3}], [{1: TrackSpeed(4.2, 1, 1)}],
    )[0]

    bands = text_bands(output)
    assert len(bands) == 2
    assert bands[-1][-1] < box_bottom, 'the block must sit above the box, not past its foot'
    # Adjacent to the box, not merely somewhere above it. The fallback for a
    # frame too short for the block anchors at the top of the image, which also
    # satisfies 'above the box', so without this the flip could be deleted
    # entirely and the test would still pass.
    assert bands[0][0] > box_top - 4 * MetricsAnnotator.VERTICAL_OFFSET, (
        'the block must be attached to the box, not parked at the top of the frame'
    )
    assert bands[-1][-1] < box_top, 'the block must clear the box rather than overlap it'


def test_the_line_pitch_leaves_clear_space_above_the_measured_glyph_extent():
    # Derived from cv2.getTextSize rather than hardcoded. The previous
    # LINE_HEIGHT = 12 left zero clear space under OpenCV 5.0.0 and minus one
    # under 4.13.0, where the bands abutted and merged. Asserting against the
    # measured extent rather than an expected pixel count is what makes this
    # hold on any build; a fixed number here would be the same defect.
    annotator = MetricsAnnotator()
    lines = ['4.2 m/s', '12.3 m']

    pitch = annotator._line_pitch(lines)
    extents = []
    for text in lines:
        (_, text_h), baseline = cv2.getTextSize(
            text, MetricsAnnotator.FONT, MetricsAnnotator.FONT_SCALE, MetricsAnnotator.THICKNESS,
        )
        extents.append(text_h + baseline)

    assert pitch - max(extents) == MetricsAnnotator.LINE_GAP
    assert MetricsAnnotator.LINE_GAP >= 2, 'the gap must exceed the band helper\'s merge tolerance'


def test_the_pitch_is_measured_from_the_tallest_line_in_the_block():
    # A digit-only line and one with a descender differ in extent; pitching
    # from the shorter would let the taller abut.
    annotator = MetricsAnnotator()

    assert annotator._line_pitch(['4.2 m/s', '12.3 m']) >= annotator._line_pitch(['12.3 m'])


def test_no_hardcoded_line_height_constant_remains():
    # A constant sitting at or below the measured glyph extent is
    # build-dependent by construction, which is what failed on the pinned
    # environment.
    assert not hasattr(MetricsAnnotator, 'LINE_HEIGHT')


def test_the_line_spacing_is_preserved_wherever_the_block_lands():
    # Stacking from one anchor means the gap between lines is constant; a
    # per-line clamp would compress it near an edge before collapsing it.
    annotator = MetricsAnnotator()
    spacings = []
    for box_top, box_bottom in ((200.0, 360.0), (400.0, 560.0), (560.0, float(FRAME_HEIGHT))):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        tracks = [{1: PlayerTrack(track_id=1, bbox=[600.0, box_top, 660.0, box_bottom], confidence=0.9)}]
        bands = text_bands(annotator.draw([frame], tracks, [{1: 9.9}], [{1: TrackSpeed(4.0, 1, 1)}])[0])
        assert len(bands) == 2
        spacings.append(bands[1][0] - bands[0][0])

    assert len(set(spacings)) == 1, f'line spacing must not vary with position, got {spacings}'


def test_a_single_line_still_draws_beneath_the_box_when_there_is_room():
    # The flip must not fire for a one-line block that fits, or every
    # speedless player's distance would jump above their box.
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    tracks = [{1: PlayerTrack(track_id=1, bbox=[600.0, 300.0, 660.0, 460.0], confidence=0.9)}]

    output = MetricsAnnotator().draw([frame], tracks, [{1: 12.3}], [{}])[0]

    bands = text_bands(output)
    assert len(bands) == 1
    assert bands[0][0] > 460.0, 'a fitting single line belongs below the box'


def test_a_frame_smaller_than_the_text_offset_is_not_fatal():
    tiny = frames(1, height=40, width=60)

    output = MetricsAnnotator().draw(
        tiny, [{1: track(x=5.0, y=5.0)}], [{1: 1.0}], [{1: TrackSpeed(2.0, 1, 1)}],
    )

    assert output[0].shape == (40, 60, 3)


def test_an_empty_frame_of_tracks_is_returned_unchanged():
    output = MetricsAnnotator().draw(frames(1), [{}], [{}], [{}])[0]

    assert np.array_equal(output, frames(1)[0])


# --- the minimap's reserved region ---------------------------------------

def sideline_track() -> PlayerTrack:
    """A player standing on the bottom-left sideline, whose foot point falls inside the minimap's panel."""
    # Measured against the real annotator: on a 1280x720 frame the panel
    # occupies x [12, 221), y [596, 708), and this box places both text rows
    # at y 666 and 682, squarely inside it.
    return PlayerTrack(track_id=7, bbox=[80.0, 560.0, 160.0, 640.0], confidence=0.9)


def panel_region() -> tuple[int, int, int, int]:
    """The rectangle MinimapAnnotator will composite over on a full-size frame."""
    return MinimapAnnotator().occupied_region(FRAME_HEIGHT, FRAME_WIDTH)


def drawn_mask(rendered: np.ndarray) -> np.ndarray:
    """A boolean mask of the pixels the annotator changed from the flat background."""
    return (rendered != BACKGROUND).any(axis=2)


def test_the_minimaps_region_covers_a_bottom_left_players_text_when_not_reserved() -> None:
    # The defect this fix addresses, pinned as a real measurement rather than
    # asserted: without the region, EVERY text pixel lands inside the panel.
    rendered = MetricsAnnotator().draw(
        frames(1), [{7: sideline_track()}], [{7: 42.3}],
        [{7: TrackSpeed(5.8, 5, 1)}],
    )[0]
    x1, y1, x2, y2 = panel_region()
    drawn = drawn_mask(rendered)
    inside = np.zeros(drawn.shape, bool)
    inside[y1:y2, x1:x2] = True

    assert drawn.sum() > 0, 'the fixture must actually draw something'
    assert (drawn & ~inside).sum() == 0, 'without the region every pixel should fall inside the panel'


def test_a_player_inside_the_reserved_region_keeps_their_text_visible() -> None:
    # The fix: with the region supplied, no text pixel falls inside the panel,
    # so none of it is overwritten when the minimap composites afterwards.
    rendered = MetricsAnnotator().draw(
        frames(1), [{7: sideline_track()}], [{7: 42.3}],
        [{7: TrackSpeed(5.8, 5, 1)}], reserved_region=panel_region(),
    )[0]
    x1, y1, x2, y2 = panel_region()
    drawn = drawn_mask(rendered)
    inside = np.zeros(drawn.shape, bool)
    inside[y1:y2, x1:x2] = True

    assert drawn.sum() > 0, 'the text must still be drawn, not dropped'
    assert (drawn & inside).sum() == 0, 'no text pixel may fall inside the reserved panel'


def test_the_text_survives_the_real_minimap_composite() -> None:
    # End to end through the actual MinimapAnnotator rather than a rectangle
    # asserted about: the panel is what overwrites the text, so the check that
    # matters is whether the pixels are still there after it has run.
    minimap = MinimapAnnotator()
    source = frames(1)
    tracks = [{7: sideline_track()}]

    annotated = MetricsAnnotator().draw(
        source, tracks, [{7: 42.3}], [{7: TrackSpeed(5.8, 5, 1)}],
        reserved_region=minimap.occupied_region(FRAME_HEIGHT, FRAME_WIDTH),
    )
    before = drawn_mask(annotated[0]).sum()
    composited = minimap.draw(annotated, [{}], [True])[0]

    x1, y1, x2, y2 = minimap.occupied_region(FRAME_HEIGHT, FRAME_WIDTH)
    outside = np.ones(composited.shape[:2], bool)
    outside[y1:y2, x1:x2] = False
    surviving = (drawn_mask(composited) & outside).sum()

    assert before > 0
    assert surviving == before, 'every text pixel must survive the composite'


def test_a_player_clear_of_the_region_is_not_moved_by_it() -> None:
    # The region must not disturb the ~80% of the frame width the panel never
    # reaches: flipping on rows alone would move text that was never at risk.
    #
    # This fixture is chosen so its block lands SQUARELY IN THE PANEL'S ROW
    # BAND while sitting far to the right of it: feet at y 640 put the two
    # rows at 666 and 682 against the panel's 596 to 708, at x 930 against its
    # 12 to 221. An earlier version used y 520, whose block flipped above the
    # box for the unrelated frame-edge reason and so never entered the row
    # band at all: a row-only check would have passed it, and the mutation
    # confirming that is what exposed the fixture rather than the assertion.
    tracks = [{1: track(x=900.0, y=480.0)}]
    without = MetricsAnnotator().draw(
        frames(1), tracks, [{1: 42.3}], [{1: TrackSpeed(5.8, 5, 1)}],
    )[0]
    with_region = MetricsAnnotator().draw(
        frames(1), tracks, [{1: 42.3}], [{1: TrackSpeed(5.8, 5, 1)}],
        reserved_region=panel_region(),
    )[0]

    assert np.array_equal(without, with_region)


def test_no_reserved_region_leaves_placement_exactly_as_before() -> None:
    # The argument is optional, and a caller drawing no minimap must get the
    # unchanged behaviour rather than a silently different layout.
    tracks = [{7: sideline_track()}]
    default = MetricsAnnotator().draw(
        frames(1), tracks, [{7: 42.3}], [{7: TrackSpeed(5.8, 5, 1)}],
    )[0]
    explicit_none = MetricsAnnotator().draw(
        frames(1), tracks, [{7: 42.3}], [{7: TrackSpeed(5.8, 5, 1)}], reserved_region=None,
    )[0]

    assert np.array_equal(default, explicit_none)


def test_a_degenerate_reserved_region_reserves_nothing() -> None:
    # occupied_region() returns a zero rectangle when the frame is too small to
    # composite into, and that must not be read as reserving the origin.
    tracks = [{7: sideline_track()}]
    unreserved = MetricsAnnotator().draw(
        frames(1), tracks, [{7: 42.3}], [{7: TrackSpeed(5.8, 5, 1)}],
    )[0]
    degenerate = MetricsAnnotator().draw(
        frames(1), tracks, [{7: 42.3}], [{7: TrackSpeed(5.8, 5, 1)}],
        reserved_region=(0, 0, 0, 0),
    )[0]

    assert np.array_equal(unreserved, degenerate)


def test_a_short_box_low_in_the_panel_keeps_its_text_beside_the_player() -> None:
    # The teleport case: feet within ~45 px of the bottom edge, so nothing fits
    # below, while box_top - VERTICAL_OFFSET - block_height still lands inside
    # the panel's rows (596 to 708 here). Both placements are rejected, and the
    # only fallback implemented was the below branch, so execution fell to the
    # top-of-frame anchor and drew the block at row 11 at the player's own
    # centre_x, which for a bottom-left player is where PossessionAnnotator's
    # caption sits. Measured before the fix: a 666 px jump away from the player.
    short_box_low = PlayerTrack(track_id=9, bbox=[90.0, 622.0, 150.0, 677.0], confidence=0.9)
    rendered = MetricsAnnotator().draw(
        frames(1), [{9: short_box_low}], [{9: 42.3}],
        [{9: TrackSpeed(5.8, 5, 1)}], reserved_region=panel_region(),
    )[0]

    bands = text_bands(rendered != BACKGROUND)
    assert bands, 'the text must still be drawn'
    # Adjacent to the player rather than at the top of the frame. The box spans
    # rows 622 to 677, and the top-of-frame anchor would put the first band at
    # about row 11.
    assert bands[0][0] > 400, f'the block teleported to the top of the frame: {bands}'
    assert bands[-1][1] < 700
    # The block is immediately above the box, not merely somewhere in the lower
    # half; a weaker bound would accept a placement drifting away from the
    # player it labels.
    assert 560 <= bands[0][0] <= 622, f'the block must sit just above the box top: {bands}'

    # Partial intersection with the panel is TOLERATED at this placement, and
    # is asserted rather than left incidental. This is the escape hatch: both
    # preferred placements were rejected, so the block is drawn beside the
    # right player and its last row is blended under the minimap. A partly
    # covered line beside the right player beats a fully visible one beside the
    # wrong one, and without this assertion a future change that 'fixed' the
    # overlap by teleporting to the top of the frame would still pass.
    x1, y1, x2, y2 = panel_region()
    drawn = drawn_mask(rendered)
    overlap = int(drawn[y1:y2, x1:x2].sum())
    assert overlap > 0, (
        'this fixture is the escape-hatch case and must actually overlap the panel — '
        'if it no longer does, it has stopped exercising the branch it was built for'
    )

    # Bounded in ROWS, not in pixels. The pixel-count version of this ceiling
    # was fitted to a local measurement of 4 of 549 covered pixels and given
    # '10x headroom', but the covered fraction is itself build-dependent,
    # because glyph WIDTHS differ between OpenCV builds while the placement
    # does not: the same fixture measures 0.7% locally under 5.0.0 and 10.8%
    # under the pinned 4.13.0, so the ceiling was 10x a quantity that varies
    # by an order of magnitude and it failed on the build that matters.
    #
    # Rows are the stable unit here. The block occupies the same rows on both
    # builds to within a pixel (569-597 against 570-596), and the same two
    # bottom rows of the second line fall inside the panel, so a row ratio
    # moves by a factor of ~2 across builds where the pixel ratio moves by
    # ~15. This is the THIRD build-metrics failure in this stage, after
    # Stage 8's caption width and this annotator's own LINE_HEIGHT: the
    # recurring lesson is that anything derived from cv2.getTextSize needs a
    # unit that does not scale with glyph metrics, or headroom sized against
    # the measured variation rather than against one build's value.
    block_rows = {int(row) for row in np.where(drawn.any(axis=1))[0]}
    covered_rows = {row for row in block_rows if y1 <= row < y2}
    assert len(covered_rows) <= 0.25 * len(block_rows), (
        f'{len(covered_rows)} of {len(block_rows)} block rows are under the panel — the '
        f'escape hatch should leave a sliver of the last line covered, not the readout'
    )


def test_an_empty_frame_list_reports_no_region_rather_than_raising() -> None:
    # occupied_region_for() exists so main.py does not reach into
    # frames[0].shape itself, which is an IndexError on an empty or unreadable
    # video. Returning None rather than raising keeps that failure where it
    # belongs (the decode step) instead of surfacing it from the annotator.
    assert MinimapAnnotator().occupied_region_for([]) is None


def test_the_reserved_region_is_read_from_the_annotator_that_owns_it() -> None:
    # The rectangle the metrics text avoids and the rectangle the minimap
    # blends into must be the same one, so occupied_region() is what the
    # compositing step itself uses rather than a parallel calculation.
    #
    # Measured against the RENDERED PIXELS rather than against _composite()'s
    # own arithmetic. An earlier version compared the reported rectangle with
    # the region _composite() derived, which now reads occupied_region()
    # itself, so the two agreed by construction and a mutation shifting the
    # reported rectangle by 30 rows passed unnoticed. Reading the pixels back
    # is the only check that can tell a truthful report from a shifted one.
    minimap = MinimapAnnotator()
    x1, y1, x2, y2 = minimap.occupied_region(FRAME_HEIGHT, FRAME_WIDTH)
    composited = minimap.draw(frames(1), [{}], [True])[0]
    changed = (composited != BACKGROUND).any(axis=2)

    rows = np.where(changed.any(axis=1))[0]
    cols = np.where(changed.any(axis=0))[0]

    assert (int(rows.min()), int(rows.max()) + 1) == (y1, y2), (
        'the reported rectangle must match the rows actually painted'
    )
    assert (int(cols.min()), int(cols.max()) + 1) == (x1, x2), (
        'the reported rectangle must match the columns actually painted'
    )


def test_the_text_colours_do_not_collide_with_the_other_annotators_captions():
    # White text over a black outline; the outline is what must not be
    # confusable with another annotator's marks.
    assert MetricsAnnotator.OUTLINE_COLOUR not in (
        PossessionAnnotator.HOLDER_COLOUR,
        EventAnnotator.PASS_COLOUR,
        EventAnnotator.INTERCEPTION_COLOUR,
        EventAnnotator.UNCLASSIFIED_COLOUR,
        MinimapAnnotator.OFF_TEMPLATE_OUTLINE,
    )
