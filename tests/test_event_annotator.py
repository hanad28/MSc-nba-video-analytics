"""Unit tests for EventAnnotator: transient per-event captions rather than a running per-team tally."""
from __future__ import annotations

import numpy as np
import pytest

from basketball.annotators.event_annotator import EventAnnotator
from basketball.annotators.player_annotator import PlayerAnnotator
from basketball.annotators.possession_annotator import PossessionAnnotator
from basketball.events.event_detector import (
    InterceptionEvent,
    PassEvent,
    UnclassifiedTransition,
)

CLIP_3_RED_OVERRIDE = (0, 0, 200)


@pytest.fixture
def frames() -> list[np.ndarray]:
    return [np.zeros((200, 400, 3), dtype=np.uint8) for _ in range(40)]


def a_pass(frame_idx: int = 5, sender_team: int = 1) -> PassEvent:
    return PassEvent(
        frame_idx=frame_idx,
        sender_track_id=10,
        receiver_track_id=7,
        sender_team=sender_team,
        receiver_team=sender_team,
    )


def an_interception(frame_idx: int = 5, interceptor_team: int = 2) -> InterceptionEvent:
    return InterceptionEvent(
        frame_idx=frame_idx,
        passer_track_id=4,
        interceptor_track_id=13,
        passer_team=1,
        interceptor_team=interceptor_team,
    )


def an_unclassified(frame_idx: int = 5) -> UnclassifiedTransition:
    return UnclassifiedTransition(
        frame_idx=frame_idx, from_track_id=10, to_track_id=7, from_team=0, to_team=1,
    )


def painted(frame: np.ndarray) -> bool:
    return bool(frame.any())


def colours_used(frame: np.ndarray) -> set[tuple[int, int, int]]:
    pixels = frame[frame.any(axis=2)]
    return {tuple(int(c) for c in pixel) for pixel in pixels}


def test_a_pass_renders_on_exactly_hold_frames_starting_at_its_frame_idx(frames):
    output = EventAnnotator().draw(frames, [a_pass(frame_idx=5)], [], [])

    for idx in range(5):
        assert not painted(output[idx]), f'frame {idx} before the event should be untouched'
    for idx in range(5, 5 + EventAnnotator.HOLD_FRAMES):
        assert painted(output[idx]), f'frame {idx} inside the hold window should be captioned'
    for idx in range(5 + EventAnnotator.HOLD_FRAMES, len(frames)):
        assert not painted(output[idx]), f'frame {idx} after the hold window should be untouched'


def test_frames_outside_the_hold_window_are_unmodified_arrays(frames):
    output = EventAnnotator().draw(frames, [a_pass(frame_idx=5)], [], [])

    assert np.array_equal(output[0], frames[0])
    assert np.array_equal(output[-1], frames[-1])


def test_the_three_event_types_render_visibly_different_captions(frames):
    annotator = EventAnnotator()

    pass_frame = annotator.draw(frames, [a_pass()], [], [])[5]
    interception_frame = annotator.draw(frames, [], [an_interception()], [])[5]
    unclassified_frame = annotator.draw(frames, [], [], [an_unclassified()])[5]

    assert EventAnnotator.PASS_COLOUR in colours_used(pass_frame)
    assert EventAnnotator.INTERCEPTION_COLOUR in colours_used(interception_frame)
    assert EventAnnotator.UNCLASSIFIED_COLOUR in colours_used(unclassified_frame)
    # Different text, not merely a different colour.
    assert not np.array_equal(pass_frame, interception_frame)
    assert not np.array_equal(pass_frame, unclassified_frame)


def test_an_event_near_the_clip_end_renders_on_the_remaining_frames_without_raising():
    frames = [np.zeros((200, 400, 3), dtype=np.uint8) for _ in range(10)]

    output = EventAnnotator().draw(frames, [a_pass(frame_idx=8)], [], [])

    assert len(output) == 10
    assert painted(output[8]) and painted(output[9])
    assert not painted(output[7])


def test_two_overlapping_events_render_on_separate_lines_without_overdrawing(frames):
    annotator = EventAnnotator()

    first_only = annotator.draw(frames, [a_pass(frame_idx=5)], [], [])[6]
    both = annotator.draw(frames, [a_pass(frame_idx=5)], [an_interception(frame_idx=6)], [])[6]

    # Both captions are present on the shared frame...
    assert EventAnnotator.PASS_COLOUR in colours_used(both)
    assert EventAnnotator.INTERCEPTION_COLOUR in colours_used(both)
    # ...and the second occupies rows the first did not, rather than covering it.
    first_rows = {int(row) for row in np.nonzero(first_only.any(axis=(1, 2)))[0]}
    both_rows = {int(row) for row in np.nonzero(both.any(axis=(1, 2)))[0]}
    assert first_rows < both_rows


def test_no_more_than_three_lines_render_at_once(frames):
    events = [a_pass(frame_idx=idx) for idx in range(5, 10)]

    output = EventAnnotator().draw(frames, events, [], [])[9]

    rows = np.nonzero(output.any(axis=(1, 2)))[0]
    line_height = rows.max() - rows.min() + 1
    # Three stacked caption boxes at font scale 0.6 stay well under 120px; five
    # would not.
    assert line_height < 120


def test_colours_do_not_collide_with_any_existing_annotator_colour():
    # An event caption must never be readable as a team ellipse or as the
    # possession holder highlight.
    existing = {
        PlayerAnnotator.DEFAULT_TEAM_1_COLOUR,
        PlayerAnnotator.DEFAULT_TEAM_2_COLOUR,
        PlayerAnnotator.UNKNOWN_TEAM_COLOUR,
        CLIP_3_RED_OVERRIDE,
        PossessionAnnotator.HOLDER_COLOUR,
    }
    event_colours = {
        EventAnnotator.PASS_COLOUR,
        EventAnnotator.INTERCEPTION_COLOUR,
        EventAnnotator.UNCLASSIFIED_COLOUR,
    }

    assert len(existing) == 5
    assert event_colours.isdisjoint(existing)
    # The three event colours are also distinct from each other.
    assert len(event_colours) == 3


@pytest.mark.parametrize('height,width', [(200, 400), (200, 60), (30, 400), (20, 20)])
def test_captions_stay_inside_the_frame_for_any_frame_size(height, width):
    frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(3)]

    output = EventAnnotator().draw(frames, [a_pass(frame_idx=0)], [], [])

    assert output[0].shape == (height, width, 3)


def test_empty_event_lists_return_frames_unchanged(frames):
    output = EventAnnotator().draw(frames, [], [], [])

    assert len(output) == len(frames)
    assert all(np.array_equal(after, before) for after, before in zip(output, frames))


def test_a_frame_idx_at_or_beyond_the_frame_count_raises(frames):
    with pytest.raises(ValueError, match='must be aligned frame-for-frame'):
        EventAnnotator().draw(frames, [a_pass(frame_idx=len(frames))], [], [])


def test_a_negative_frame_idx_raises(frames):
    with pytest.raises(ValueError, match='must be aligned frame-for-frame'):
        EventAnnotator().draw(frames, [a_pass(frame_idx=-1)], [], [])


def test_an_interception_caption_shows_the_interceptors_team_not_the_passers(frames):
    # passer_team=1, interceptor_team=2 -- the caption must read team 2, the
    # same credit the scoring notebook gives.
    annotator = EventAnnotator()
    interception = InterceptionEvent(
        frame_idx=5, passer_track_id=4, interceptor_track_id=13, passer_team=1, interceptor_team=2,
    )

    captions = annotator._build_captions(frames, [], [interception], [])

    assert captions[0][1] == 'INTERCEPTION  team 2  4 -> 13'


def test_caption_text_matches_the_specified_wording(frames):
    annotator = EventAnnotator()

    texts = [
        text for _, text, _ in annotator._build_captions(
            frames, [a_pass()], [], [an_unclassified(frame_idx=6)],
        )
    ]

    assert texts == ['PASS  team 1  10 -> 7', 'UNCLASSIFIED  10 -> 7']


def test_does_not_mutate_the_input_frames(frames):
    EventAnnotator().draw(frames, [a_pass(frame_idx=5)], [], [])

    assert all(not frame.any() for frame in frames)
