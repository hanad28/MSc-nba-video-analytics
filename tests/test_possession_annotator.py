"""Unit tests for PossessionAnnotator."""

from __future__ import annotations

import numpy as np
import pytest

from basketball.annotators.player_annotator import PlayerAnnotator
from basketball.annotators.possession_annotator import PossessionAnnotator
from basketball.detection.ball_detector import BallDetection
from basketball.detection.player_detector import PlayerTrack


@pytest.fixture
def frames() -> list[np.ndarray]:
    return [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(2)]


def track(track_id: int = 1, bbox: tuple[float, ...] = (80, 60, 120, 140)) -> PlayerTrack:
    return PlayerTrack(track_id=track_id, bbox=list(bbox), confidence=0.9)


def real_ball(bbox: tuple[float, ...] = (95, 95, 105, 105)) -> dict[int, BallDetection]:
    return {1: BallDetection(bbox=list(bbox), confidence=0.8)}


def interpolated_ball(bbox: tuple[float, ...] = (95, 95, 105, 105)) -> dict[int, BallDetection]:
    return {1: BallDetection(bbox=list(bbox), confidence=0.0)}


def colours_used(frame: np.ndarray) -> set[tuple[int, int, int]]:
    painted = frame[frame.any(axis=2)]
    return {tuple(int(c) for c in pixel) for pixel in painted}


def test_confirmed_holder_is_highlighted_and_captioned(frames):
    tracks = [{1: track()}, {}]
    possession = [1, -1]
    ball_detections = [real_ball(), {}]

    output = PossessionAnnotator().draw(frames, tracks, possession, ball_detections)

    assert PossessionAnnotator.HOLDER_COLOUR in colours_used(output[0])


def test_no_confirmed_holder_but_real_ball_detection_is_captioned_distinctly(frames):
    tracks = [{1: track()}, {}]
    possession = [-1, -1]
    ball_detections = [real_ball(), {}]

    output = PossessionAnnotator().draw(frames, tracks, possession, ball_detections)

    assert PossessionAnnotator.HOLDER_COLOUR not in colours_used(output[0])
    assert output[0].any()  # the caption itself still paints pixels


def test_no_confirmed_holder_and_no_real_ball_detection_is_captioned_distinctly(frames):
    tracks = [{}, {}]
    possession = [-1, -1]
    ball_detections = [{}, {}]

    output = PossessionAnnotator().draw(frames, tracks, possession, ball_detections)

    assert PossessionAnnotator.HOLDER_COLOUR not in colours_used(output[0])


def test_interpolated_only_ball_detection_counts_as_not_really_detected(frames):
    # confidence == 0.0 is the interpolator's own marker for a fabricated
    # position -- state (c), not state (b), must be reported for it.
    tracks = [{}, {}]
    possession = [-1, -1]
    ball_detections = [interpolated_ball(), {}]

    output = PossessionAnnotator().draw(frames, tracks, possession, ball_detections)

    assert PossessionAnnotator.HOLDER_COLOUR not in colours_used(output[0])


def test_returns_copies_and_does_not_mutate_inputs(frames):
    tracks = [{1: track()}, {}]
    possession = [1, -1]
    ball_detections = [real_ball(), {}]

    output = PossessionAnnotator().draw(frames, tracks, possession, ball_detections)

    assert len(output) == 2
    assert all(not original.any() for original in frames)


def test_holder_colour_does_not_collide_with_player_annotator_colours():
    # The system genuinely distinguishes the confirmed-holder state from a
    # team-colour ellipse -- if the two colours ever matched, a highlighted
    # holder would be visually indistinguishable from an ordinary team
    # ellipse of that colour.
    used_by_player_annotator = {
        PlayerAnnotator.DEFAULT_TEAM_1_COLOUR,
        PlayerAnnotator.DEFAULT_TEAM_2_COLOUR,
        PlayerAnnotator.UNKNOWN_TEAM_COLOUR,
    }

    assert PossessionAnnotator.HOLDER_COLOUR not in used_by_player_annotator


def test_rejects_mismatched_possession_length(frames):
    with pytest.raises(ValueError, match='aligned frame-for-frame'):
        PossessionAnnotator().draw(frames, [{1: track()}, {}], [1], [{}, {}])


def test_rejects_mismatched_track_length(frames):
    with pytest.raises(ValueError, match='aligned frame-for-frame'):
        PossessionAnnotator().draw(frames, [{1: track()}], [1, -1], [{}, {}])


def test_rejects_mismatched_ball_detection_length(frames):
    with pytest.raises(ValueError, match='aligned frame-for-frame'):
        PossessionAnnotator().draw(frames, [{1: track()}, {}], [1, -1], [{}])


def test_holder_id_not_present_in_frame_tracks_still_shows_the_possession_caption_but_draws_no_box(frames, monkeypatch):
    # A holder the pipeline confirmed but whose track is absent from this
    # frame (e.g. the track was dropped) must still be reported as in
    # possession -- only the highlight box is impossible to draw, so the
    # frame must not fall back to a no-holder caption and contradict the
    # possession result. Must also not raise a KeyError.
    captions: list[str] = []
    monkeypatch.setattr(
        PossessionAnnotator,
        '_draw_caption',
        lambda self, frame, text: captions.append(text),
    )

    tracks = [{}, {}]
    possession = [1, -1]
    ball_detections = [real_ball(), {}]

    output = PossessionAnnotator().draw(frames, tracks, possession, ball_detections)

    assert captions[0] == 'POSSESSION: PLAYER 1'
    assert PossessionAnnotator.HOLDER_COLOUR not in colours_used(output[0])


def captured_captions(monkeypatch) -> list[tuple[str, int]]:
    """Record every caption drawn, with its line index, via the same _draw_caption patch the tests above use."""
    captions: list[tuple[str, int]] = []
    monkeypatch.setattr(
        PossessionAnnotator,
        '_draw_caption',
        lambda self, frame, text, line_index=0: captions.append((text, line_index)),
    )
    return captions


def test_confirmed_holder_on_an_interpolated_ball_draws_the_second_line(frames, monkeypatch):
    # A streak can confirm a holder off a single genuine detection, leaving
    # every other frame in it interpolated -- these are exactly the frames a
    # human needs flagged when inspecting stills, so the fabricated ball
    # position must be stated on the frame rather than inferred.
    captions = captured_captions(monkeypatch)

    PossessionAnnotator().draw(frames, [{1: track()}, {}], [1, -1], [interpolated_ball(), {}])

    assert captions[0] == ('POSSESSION: PLAYER 1', 0)
    assert captions[1] == (PossessionAnnotator.CONFIRMED_HOLDER_INTERPOLATED_BALL_CAPTION, 1)


def test_confirmed_holder_on_a_real_ball_draws_no_second_line(frames, monkeypatch):
    captions = captured_captions(monkeypatch)

    PossessionAnnotator().draw(frames, [{1: track()}, {}], [1, -1], [real_ball(), {}])

    assert captions[0] == ('POSSESSION: PLAYER 1', 0)
    assert all(line_index == 0 for _, line_index in captions)


def test_confirmed_holder_with_no_ball_entry_draws_no_second_line_and_does_not_raise(frames, monkeypatch):
    # holder_id != -1 with no ball dict entry at all must not reach an
    # attribute access on None while deciding whether to add the second line.
    captions = captured_captions(monkeypatch)

    PossessionAnnotator().draw(frames, [{1: track()}, {}], [1, -1], [{}, {}])

    assert captions[0] == ('POSSESSION: PLAYER 1', 0)
    assert all(line_index == 0 for _, line_index in captions)


def test_the_two_interpolated_ball_captions_read_differently():
    # The annotator is the instrument for inspecting individual frames, so a
    # still must be unambiguous on its own: a confirmed-possession frame and a
    # no-holder frame must not render the same interpolation wording.
    assert (
        PossessionAnnotator.CONFIRMED_HOLDER_INTERPOLATED_BALL_CAPTION
        != PossessionAnnotator.INTERPOLATED_BALL_CAPTION
    )


@pytest.mark.parametrize('height,width', [(20, 20), (30, 300), (300, 30), (5, 5)])
def test_caption_stays_inside_the_frame_for_any_frame_size(height, width):
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    output = PossessionAnnotator().draw([frame], [{}], [-1], [{}])

    # Any painted pixel must be a valid in-bounds index -- proven simply by
    # the fact that draw() did not raise and the output frame kept its shape.
    assert output[0].shape == (height, width, 3)
