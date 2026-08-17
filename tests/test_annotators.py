"""Unit tests for PlayerAnnotator and BallAnnotator."""

from __future__ import annotations

import numpy as np
import pytest

from basketball.annotators.ball_annotator import BallAnnotator
from basketball.annotators.player_annotator import PlayerAnnotator
from basketball.detection.ball_detector import BallDetection
from basketball.detection.player_detector import PlayerTrack


@pytest.fixture
def frames() -> list[np.ndarray]:
    return [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(2)]


def track(track_id: int = 1, bbox: tuple[float, ...] = (80, 60, 120, 140)) -> PlayerTrack:
    return PlayerTrack(track_id=track_id, bbox=list(bbox), confidence=0.9)


def colours_used(frame: np.ndarray) -> set[tuple[int, int, int]]:
    painted = frame[frame.any(axis=2)]
    return {tuple(int(c) for c in pixel) for pixel in painted}


def test_player_annotator_returns_copies(frames):
    tracks = [{1: track()}, {1: track()}]

    output = PlayerAnnotator().draw(frames, tracks)

    assert len(output) == 2
    assert all(not original.any() for original in frames)
    assert all(annotated.any() for annotated in output)


def test_player_annotator_falls_back_to_red_without_team_data(frames):
    output = PlayerAnnotator().draw(frames, [{1: track()}, {}])

    assert (0, 0, 255) in colours_used(output[0])
    assert not output[1].any()


def test_player_annotator_uses_team_colours(frames):
    annotator = PlayerAnnotator()
    tracks = [{1: track(1), 2: track(2, (10, 10, 40, 60))}, {}]
    teams = [{1: 1, 2: 2}, {}]

    output = annotator.draw(frames, tracks, team_assignment=teams)
    used = colours_used(output[0])

    assert annotator.team_colours[1] in used
    assert annotator.team_colours[2] in used


def test_player_annotator_uses_the_unknown_colour_for_unassigned_players(frames):
    annotator = PlayerAnnotator()

    output = annotator.draw(frames, [{7: track(7)}, {}], team_assignment=[{}, {}])

    assert annotator.team_colours[0] in colours_used(output[0])


def test_player_annotator_uses_constructor_colours(frames):
    annotator = PlayerAnnotator(team_1_colour=(1, 2, 3), team_2_colour=(4, 5, 6))
    tracks = [{1: track(1), 2: track(2, (10, 10, 40, 60))}, {}]
    teams = [{1: 1, 2: 2}, {}]

    output = annotator.draw(frames, tracks, team_assignment=teams)
    used = colours_used(output[0])

    assert (1, 2, 3) in used
    assert (4, 5, 6) in used
    assert annotator.team_colours[0] == PlayerAnnotator.UNKNOWN_TEAM_COLOUR  # team 0 stays constant, not configurable


def test_player_annotator_defaults_match_the_documented_constants():
    annotator = PlayerAnnotator()

    assert annotator.team_colours[1] == PlayerAnnotator.DEFAULT_TEAM_1_COLOUR
    assert annotator.team_colours[2] == PlayerAnnotator.DEFAULT_TEAM_2_COLOUR


def test_player_annotator_rejects_mismatched_input_lengths(frames):
    with pytest.raises(ValueError, match='aligned frame-for-frame'):
        PlayerAnnotator().draw(frames, [{1: track()}])


def test_ball_annotator_draws_only_on_frames_with_a_detection(frames):
    detections = [{1: BallDetection(bbox=[98, 100, 106, 108], confidence=0.8)}, {}]

    output = BallAnnotator().draw(frames, detections)

    assert output[0].any()
    assert not output[1].any()
    assert all(not original.any() for original in frames)


def test_ball_annotator_uses_its_green_colour(frames):
    annotator = BallAnnotator()
    detections = [{1: BallDetection(bbox=[98, 100, 106, 108], confidence=0.8)}, {}]

    output = annotator.draw(frames, detections)

    assert annotator.colour == BallAnnotator.COLOUR
    assert annotator.colour in colours_used(output[0])


def test_ball_annotator_annotates_interpolated_detections_too(frames):
    detections = [{1: BallDetection(bbox=[98, 100, 106, 108], confidence=0.0)} for _ in frames]

    output = BallAnnotator().draw(frames, detections)

    assert all(annotated.any() for annotated in output)
