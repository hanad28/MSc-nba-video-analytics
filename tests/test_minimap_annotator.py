"""Unit tests for Stage 8's MinimapAnnotator (basketball/annotators/minimap_annotator.py)."""

from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from basketball.annotators.event_annotator import EventAnnotator
from basketball.annotators.minimap_annotator import MinimapAnnotator
from basketball.annotators.player_annotator import PlayerAnnotator
from basketball.annotators.possession_annotator import PossessionAnnotator
from basketball.homography.court_mapper import COURT_MARGIN_M
from basketball.keypoints.court_template import COURT_LENGTH_M, COURT_WIDTH_M, to_pixels

FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
BACKGROUND = 60


def frames(count: int, height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> list[np.ndarray]:
    """A list of uniform BGR frames, distinguishable from anything the annotator draws."""
    return [np.full((height, width, 3), BACKGROUND, dtype=np.uint8) for _ in range(count)]


def template_at(directory: Path, width: int = 598, height: int = 321) -> str:
    """Write a synthetic court template at the real asset's dimensions and return its path."""
    path = directory / 'court_template.png'
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return str(path)


def annotator(directory: Path, **kwargs: object) -> MinimapAnnotator:
    """A MinimapAnnotator pointed at a synthetic template rather than the committed asset."""
    return MinimapAnnotator(template_path=template_at(directory), **kwargs)


# --- team colours --------------------------------------------------------

def test_team_colours_are_read_from_player_annotator_not_duplicated():
    # Read by attribute rather than compared against a literal: a duplicated
    # literal would drift the moment either annotator's palette is retuned,
    # and a player would be one colour in the broadcast view and another in
    # the tactical view.
    assert MinimapAnnotator.TEAM_COLOURS[1] is PlayerAnnotator.DEFAULT_TEAM_1_COLOUR
    assert MinimapAnnotator.TEAM_COLOURS[2] is PlayerAnnotator.DEFAULT_TEAM_2_COLOUR
    assert MinimapAnnotator.TEAM_COLOURS[0] is PlayerAnnotator.UNKNOWN_TEAM_COLOUR


def test_a_per_clip_colour_override_reaches_the_minimap(tmp_path):
    # clip_3's red team-2 override must colour both views alike.
    red = (0, 0, 255)
    instance = annotator(tmp_path, team_2_colour=red)

    assert instance.team_colours[2] == red
    assert instance.team_colours[1] == PlayerAnnotator.DEFAULT_TEAM_1_COLOUR


def test_an_override_does_not_mutate_the_class_level_palette(tmp_path):
    annotator(tmp_path, team_2_colour=(0, 0, 255))

    assert MinimapAnnotator.TEAM_COLOURS[2] is PlayerAnnotator.DEFAULT_TEAM_2_COLOUR


# --- frame handling ------------------------------------------------------

def test_a_frame_count_mismatch_raises_naming_both_counts(tmp_path):
    with pytest.raises(ValueError, match='2 court-position frames for 3 video frames'):
        annotator(tmp_path).draw(frames(3), [{}, {}], [True, True])


def test_a_team_assignment_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match='1 team-assignment frames for 2 video frames'):
        annotator(tmp_path).draw(frames(2), [{}, {}], [True, True], [{}])


def test_the_caller_s_frames_are_never_mutated(tmp_path):
    originals = frames(2)
    untouched = [frame.copy() for frame in originals]

    annotator(tmp_path).draw(originals, [{7: (5.0, 5.0)}, {}], [True, True])

    for original, reference in zip(originals, untouched):
        assert np.array_equal(original, reference), 'draw() must copy, never mutate in place'


def test_output_length_and_shape_match_the_input(tmp_path):
    source = frames(3)

    output = annotator(tmp_path).draw(source, [{}, {7: (5.0, 5.0)}, {}], [True, True, True])

    assert len(output) == 3
    assert all(a.shape == b.shape for a, b in zip(output, source))


# --- what is drawn -------------------------------------------------------

def test_the_overlay_lands_in_the_bottom_left_corner(tmp_path):
    # PossessionAnnotator owns the top-left and EventAnnotator the top-right,
    # so the minimap must not occupy either.
    output = annotator(tmp_path).draw(frames(1), [{7: (5.0, 5.0)}], [True])[0]

    height, width = output.shape[:2]
    bottom_left = output[height // 2:, :width // 2]
    top_left = output[:height // 4, :width // 4]
    top_right = output[:height // 4, 3 * width // 4:]

    assert np.any(bottom_left != BACKGROUND), 'the overlay must be drawn bottom-left'
    assert np.all(top_left == BACKGROUND), 'the top-left belongs to the possession caption'
    assert np.all(top_right == BACKGROUND), 'the top-right belongs to the event captions'


def test_an_empty_position_frame_still_draws_the_court(tmp_path):
    # An absent overlay and an empty one look identical, so a viewer could not
    # tell 'no players' from 'no homography', and the second is a measured
    # ~10% of frames.
    output = annotator(tmp_path).draw(frames(1), [{}], [True])[0]

    height, width = output.shape[:2]
    bottom_left = output[height // 2:, :width // 2]

    assert np.any(bottom_left != BACKGROUND), 'the empty court must still be drawn'


def test_an_empty_frame_is_captioned_differently_from_a_populated_one(tmp_path):
    instance = annotator(tmp_path)

    empty = instance.draw(frames(1), [{}], [True])[0]
    populated = instance.draw(frames(1), [{7: (5.0, 5.0)}], [True])[0]

    assert not np.array_equal(empty, populated), 'the two states must be distinguishable'


def test_a_frame_without_a_homography_is_captioned(tmp_path):
    # Comparing against the BARE template is what actually pins the caption: a
    # populated frame draws a dot, so render-vs-render comparisons differ
    # whether or not the caption exists.
    instance = annotator(tmp_path)
    bare = instance.load_template()

    failed = instance._render_court({}, {}, has_homography=False)

    assert not np.array_equal(failed, bare), 'a homography failure must be captioned'


def test_a_homography_with_no_players_is_not_captioned_as_a_failure(tmp_path):
    # The fix this test exists for. map_to_court returns an empty dict for
    # three distinct reasons, and only one is a failure. The minimap is how
    # the transform is verified by eye, so captioning a correctly-mapped frame
    # as a failure would inflate the apparent failure rate above the measured
    # one and send a reader chasing a problem that is not there.
    instance = annotator(tmp_path)
    bare = instance.load_template()

    empty_but_working = instance._render_court({}, {}, has_homography=True)

    assert np.array_equal(empty_but_working, bare), (
        'a frame with a homography and no players must draw a bare court, not a failure caption'
    )


def test_a_frame_whose_positions_were_all_dropped_is_not_captioned_as_a_failure(tmp_path):
    # The third cause: a homography was fitted but every mapped position fell
    # out of bounds. Indistinguishable from the second at the annotator, and
    # equally not a homography failure.
    instance = annotator(tmp_path)

    dropped = instance._render_court({}, {}, has_homography=True)
    failed = instance._render_court({}, {}, has_homography=False)

    assert not np.array_equal(dropped, failed)


def test_the_failure_caption_reaches_the_frame_through_draw(tmp_path):
    # The flag must be wired through draw(), not only reachable on the private
    # render helper.
    instance = annotator(tmp_path)

    captioned = instance.draw(frames(1), [{}], [False])[0]
    uncaptioned = instance.draw(frames(1), [{}], [True])[0]

    assert not np.array_equal(captioned, uncaptioned)


def test_a_homography_flag_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match='1 homography flags for 2 video frames'):
        annotator(tmp_path).draw(frames(2), [{}, {}], [True])


def test_the_homography_flags_cannot_be_omitted(tmp_path):
    # The conflation fixed in 30f6e26 was reachable again through a default
    # that derived the caption from bool(court_positions[index]). Stage 8's
    # wiring is the next task, and an unthreaded call would have restored the
    # bug silently on the very output this stage is verified by. A required
    # argument makes that a TypeError at the call site instead.
    instance = annotator(tmp_path)

    with pytest.raises(TypeError):
        instance.draw(frames(1), [{}])


def test_no_default_reinstates_the_positions_derived_caption(tmp_path):
    # Guards the mechanism, not only the signature: nothing in draw() or
    # _render_court may fall back to inferring the flag from the positions.
    for function in (MinimapAnnotator.draw, MinimapAnnotator._render_court):
        signature = inspect.signature(function)
        parameter = (
            signature.parameters['frame_has_homography']
            if 'frame_has_homography' in signature.parameters
            else signature.parameters['has_homography']
        )
        assert parameter.default is inspect.Parameter.empty, (
            f'{function.__name__} must not default the homography flag'
        )

    source = inspect.getsource(MinimapAnnotator.draw)
    code = '\n'.join(
        line for line in source.splitlines() if not line.strip().startswith('#')
    )
    assert 'bool(court_positions' not in code


# --- the caption fits ----------------------------------------------------

# Fraction of the rendered width the caption must leave free. cv2.getTextSize
# is BUILD-DEPENDENT: this caption measures 76 px under OpenCV 5.0.0 and 96 px
# under 4.13.0, a 26% spread, so an assertion comparing a measured width
# against the exact rendered width is build-dependent by construction and will
# drift on the next upgrade. The margin is what makes this test portable.
CAPTION_WIDTH_HEADROOM = 0.85


@pytest.mark.parametrize('scale', [0.5, 0.35, 0.30, 0.25])
def test_the_caption_fits_within_the_rendered_minimap_at_every_supported_scale(tmp_path, scale):
    # Measured rather than assumed: a caption wider than the render clamps to
    # x=0 and loses its tail, which reads as a different message rather than
    # as an overflow, and every render-vs-render test still passes.
    #
    # 0.25 is the smallest scale parametrised because it is the smallest the
    # class supports; anything below raises in the constructor.
    assert scale >= MinimapAnnotator.MIN_SUPPORTED_SCALE, 'fixture must stay within support'
    instance = annotator(tmp_path, scale=scale)
    rendered_width = instance.load_template().shape[1]

    (text_w, _), _ = cv2.getTextSize(
        MinimapAnnotator.NO_HOMOGRAPHY_CAPTION,
        MinimapAnnotator.FONT,
        MinimapAnnotator.CAPTION_FONT_SCALE,
        MinimapAnnotator.CAPTION_THICKNESS,
    )

    assert text_w <= rendered_width * CAPTION_WIDTH_HEADROOM, (
        f'caption is {text_w}px against a {rendered_width}px minimap at scale {scale}, '
        f'leaving less than the {1 - CAPTION_WIDTH_HEADROOM:.0%} headroom a build difference needs'
    )


def test_a_scale_below_the_supported_minimum_raises(tmp_path):
    # Raised rather than rendered: an overlay too small to read player
    # positions from defeats the minimap's only purpose, and drawing one
    # silently would look like a working tactical view.
    with pytest.raises(ValueError, match='scale must be at least'):
        annotator(tmp_path, scale=MinimapAnnotator.MIN_SUPPORTED_SCALE - 0.01)


def test_the_minimum_supported_scale_leaves_caption_headroom_on_any_build(tmp_path):
    # The floor and the caption must stay consistent: if either moves, the
    # smallest supported render must still hold the caption with headroom.
    instance = annotator(tmp_path, scale=MinimapAnnotator.MIN_SUPPORTED_SCALE)
    rendered_width = instance.load_template().shape[1]

    (text_w, _), _ = cv2.getTextSize(
        MinimapAnnotator.NO_HOMOGRAPHY_CAPTION,
        MinimapAnnotator.FONT,
        MinimapAnnotator.CAPTION_FONT_SCALE,
        MinimapAnnotator.CAPTION_THICKNESS,
    )

    # Against the widest build measured (96 px under OpenCV 4.13.0) rather than
    # this build's own number, so the assertion holds wherever it runs.
    widest_observed = max(text_w, 96)
    assert widest_observed <= rendered_width * CAPTION_WIDTH_HEADROOM


def test_the_caption_is_skipped_rather_than_truncated_when_it_cannot_fit(tmp_path):
    # The guard remains reachable even though no supported scale reaches it:
    # a caller may supply a template narrower than the committed asset, and a
    # future OpenCV build may measure the caption wider still. Exercised with a
    # deliberately narrow template rather than an unsupported scale, since
    # scales below MIN_SUPPORTED_SCALE now raise in the constructor.
    narrow = tmp_path / 'narrow.png'
    cv2.imwrite(str(narrow), np.full((40, 60, 3), 200, dtype=np.uint8))
    instance = MinimapAnnotator(template_path=str(narrow), scale=1.0)
    bare = instance.load_template()

    rendered = instance._render_court({}, {}, has_homography=False)

    assert np.array_equal(rendered, bare)


def test_the_default_scale_leaves_headroom_for_the_caption(tmp_path):
    instance = annotator(tmp_path)
    width = instance.load_template().shape[1]
    (text_w, _), _ = cv2.getTextSize(
        MinimapAnnotator.NO_HOMOGRAPHY_CAPTION, MinimapAnnotator.FONT,
        MinimapAnnotator.CAPTION_FONT_SCALE, MinimapAnnotator.CAPTION_THICKNESS,
    )

    assert text_w < width * 0.9, 'the caption should not span the full minimap width'


def test_players_of_different_teams_are_drawn_in_different_colours(tmp_path):
    instance = annotator(tmp_path)
    positions = [{1: (5.0, 4.0), 2: (20.0, 11.0)}]

    one_team = instance.draw(frames(1), positions, [True], [{1: 1, 2: 1}])[0]
    two_teams = instance.draw(frames(1), positions, [True], [{1: 1, 2: 2}])[0]

    assert not np.array_equal(one_team, two_teams)


def test_an_unassigned_player_is_drawn_in_the_unknown_colour(tmp_path):
    instance = annotator(tmp_path)
    positions = [{1: (5.0, 4.0)}]

    unknown = instance.draw(frames(1), positions, [True], [{}])[0]
    team_one = instance.draw(frames(1), positions, [True], [{1: 1}])[0]

    assert not np.array_equal(unknown, team_one)


def test_positions_at_opposite_ends_of_the_court_render_apart(tmp_path):
    # Pins that the metric-to-pixel mapping is actually applied, rather than
    # every player landing at the same spot.
    instance = annotator(tmp_path)

    left = instance.draw(frames(1), [{1: (1.0, 7.0)}], [True])[0]
    right = instance.draw(frames(1), [{1: (27.0, 7.0)}], [True])[0]

    assert not np.array_equal(left, right)


# --- out-of-bounds positions within the accepted margin -------------------

@pytest.mark.parametrize('position,description', [
    ((14.0, COURT_WIDTH_M + COURT_MARGIN_M), 'the full margin past the sideline'),
    ((14.0, COURT_WIDTH_M), 'exactly on the sideline'),
    ((-COURT_MARGIN_M, 7.0), 'the full margin behind the baseline'),
    ((COURT_LENGTH_M + COURT_MARGIN_M, 7.0), 'the full margin past the far baseline'),
    ((-COURT_MARGIN_M, -COURT_MARGIN_M), 'diagonally outside both'),
])
def test_a_position_within_the_accepted_margin_still_renders(tmp_path, position, description):
    # CourtMapper accepts these and counts them in positions_mapped, but
    # to_pixels() scales linearly with no clamp, so they landed outside the
    # template and cv2.circle dropped the dot. The drawn count then disagreed
    # with the reported count exactly when a player stepped over a line, and
    # the minimap is this stage's verification mechanism.
    instance = annotator(tmp_path)
    bare = instance.load_template()

    rendered = instance._render_court({1: position}, {1: 1}, True)

    assert not np.array_equal(rendered, bare), f'a player {description} must still be drawn'


@pytest.mark.parametrize('position', [
    (14.0, COURT_WIDTH_M + COURT_MARGIN_M),
    (-COURT_MARGIN_M, 7.0),
    (COURT_LENGTH_M + COURT_MARGIN_M, COURT_WIDTH_M + COURT_MARGIN_M),
])
def test_an_off_template_dot_is_placed_fully_inside_the_render(tmp_path, position):
    # PLAYER_RADIUS is included in the clamp so the dot is fully visible rather
    # than half-clipped at the edge, matching how KeypointAnnotator clamps its
    # index labels.
    instance = annotator(tmp_path)
    template = instance.load_template()
    height, width = template.shape[:2]

    (x, y), off_template = instance._render_position(position, width, height)

    assert off_template
    assert MinimapAnnotator.PLAYER_RADIUS <= x <= width - 1 - MinimapAnnotator.PLAYER_RADIUS
    assert MinimapAnnotator.PLAYER_RADIUS <= y <= height - 1 - MinimapAnnotator.PLAYER_RADIUS


def test_an_in_court_position_is_not_flagged(tmp_path):
    # The flag must not fire on positions that already fit, or every dot would
    # be reported as displaced.
    instance = annotator(tmp_path)
    template = instance.load_template()
    height, width = template.shape[:2]

    _, off_template = instance._render_position(
        (COURT_LENGTH_M / 2, COURT_WIDTH_M / 2), width, height,
    )

    assert not off_template


@pytest.mark.parametrize('position,description', [
    ((14.0, COURT_WIDTH_M - 0.5), 'half a metre inside the sideline'),
    ((14.0, COURT_WIDTH_M - 0.2), 'a fifth of a metre inside the sideline'),
    ((0.5, 7.0), 'half a metre inside the baseline'),
    ((COURT_LENGTH_M - 0.5, 7.0), 'half a metre inside the far baseline'),
    ((0.2, 0.2), 'just inside a corner'),
])
def test_a_position_just_inside_a_line_is_not_flagged_as_off_template(tmp_path, position, description):
    # The radius inset spans 0.68 m of court at a 5 px radius, so deciding the
    # flag from the inset marked a band that wide inside every line as
    # displaced, the opposite of what the distinct outline is for. The flag
    # is decided against the render bounds; the inset is placement only.
    #
    # A sub-pixel residual remains and is not a defect: one render row spans
    # 0.136 m, so a position within ~0.14 m of a line rounds past the last row
    # and is flagged. That is the render's own resolution, not the inset's
    # 0.68 m, and it shrinks with the template scale.
    instance = annotator(tmp_path)
    template = instance.load_template()
    height, width = template.shape[:2]

    (x, y), off_template = instance._render_position(position, width, height)

    assert not off_template, f'a player {description} is in court, not off the template'
    # And the dot is still placed whole, which is what the inset is for.
    assert MinimapAnnotator.PLAYER_RADIUS <= x <= width - 1 - MinimapAnnotator.PLAYER_RADIUS
    assert MinimapAnnotator.PLAYER_RADIUS <= y <= height - 1 - MinimapAnnotator.PLAYER_RADIUS


def test_a_position_just_inside_a_line_renders_with_the_ordinary_outline(tmp_path):
    # The end-to-end consequence: such a player must look like every other
    # in-court player, not like a displaced one.
    instance = annotator(tmp_path)

    rendered = instance._render_court({1: (14.0, COURT_WIDTH_M - 0.5)}, {1: 1}, True)

    assert not np.any(np.all(rendered == MinimapAnnotator.OFF_TEMPLATE_OUTLINE, axis=-1))


def test_an_off_template_dot_is_visually_distinguishable_from_an_in_court_one(tmp_path):
    # A dot pinned to the edge is not where the player is, and a reader using
    # the minimap as verification must not mistake it for a measurement.
    instance = annotator(tmp_path)

    in_court = instance._render_court({1: (COURT_LENGTH_M / 2, COURT_WIDTH_M / 2)}, {1: 1}, True)
    off_template = instance._render_court(
        {1: (14.0, COURT_WIDTH_M + COURT_MARGIN_M)}, {1: 1}, True,
    )

    assert not np.array_equal(in_court, off_template)
    # The distinguishing outline colour must actually appear on the clamped
    # render and not on the in-court one.
    assert np.any(np.all(off_template == MinimapAnnotator.OFF_TEMPLATE_OUTLINE, axis=-1))
    assert not np.any(np.all(in_court == MinimapAnnotator.OFF_TEMPLATE_OUTLINE, axis=-1))


def test_the_clamp_lives_at_the_render_site_not_in_to_pixels():
    # to_pixels() is a shared coordinate conversion; clamping there would
    # silently distort every other caller, so it must stay linear.
    far_outside = (COURT_LENGTH_M * 2, COURT_WIDTH_M * 2)

    x, y = to_pixels(far_outside, 209, 112)

    assert x > 209 and y > 112, 'to_pixels() must remain an unclamped linear scaling'


# --- the template --------------------------------------------------------

def test_a_missing_template_raises_naming_the_path(tmp_path):
    instance = MinimapAnnotator(template_path=str(tmp_path / 'absent.png'))

    with pytest.raises(FileNotFoundError, match='absent.png'):
        instance.draw(frames(1), [{}], [True])


def test_an_undecodable_template_raises(tmp_path):
    broken = tmp_path / 'broken.png'
    broken.write_bytes(b'not an image')
    instance = MinimapAnnotator(template_path=str(broken))

    with pytest.raises(OSError, match='could not be decoded'):
        instance.draw(frames(1), [{}], [True])


def test_the_template_is_loaded_once_and_cached(tmp_path, monkeypatch):
    instance = annotator(tmp_path)
    reads: list[str] = []
    real_imread = cv2.imread

    def counting_imread(path: str, *args: object) -> np.ndarray:
        reads.append(path)
        return real_imread(path, *args)

    monkeypatch.setattr(cv2, 'imread', counting_imread)
    instance.draw(frames(4), [{}] * 4, [True] * 4)

    assert len(reads) == 1, 'the template must be decoded once, not per frame'


def test_the_committed_asset_exists_and_matches_the_documented_size():
    # The real asset, not a synthetic one: 598x321 with aspect 1.863, which is
    # why to_pixels() is drawing only and measurement stays in metres.
    asset = Path('assets/court_template.png')

    assert asset.exists()
    image = cv2.imread(str(asset))
    assert image is not None
    height, width = image.shape[:2]
    assert (width, height) == (598, 321)


# --- edge cases ----------------------------------------------------------

def test_a_frame_smaller_than_the_overlay_is_not_fatal(tmp_path):
    # Not reachable at broadcast resolution, but a thumbnail must not kill a run.
    tiny = frames(1, height=40, width=60)

    output = annotator(tmp_path).draw(tiny, [{7: (5.0, 5.0)}], [True])

    assert output[0].shape == (40, 60, 3)


def test_the_minimap_colours_do_not_collide_with_the_caption_colours(tmp_path):
    # The overlay sits under captions drawn by other annotators; its own
    # caption must stay legible against the palette already in use.
    assert MinimapAnnotator.CAPTION_COLOUR not in (
        PossessionAnnotator.HOLDER_COLOUR,
        EventAnnotator.PASS_COLOUR,
        EventAnnotator.INTERCEPTION_COLOUR,
        EventAnnotator.UNCLASSIFIED_COLOUR,
    )
