"""Unit tests for the player and ball drawing primitives."""

from __future__ import annotations

import numpy as np
import pytest

from basketball.annotators.annotation_utils import annotate_ball, annotate_player

RED = (0, 0, 255)


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_annotate_player_draws_onto_the_frame_in_place(frame):
    result = annotate_player(frame, [80, 60, 120, 140], track_id=7, colour=RED)

    assert result is frame
    assert frame.any()


def test_annotate_player_uses_the_requested_colour(frame):
    annotate_player(frame, [80, 60, 120, 140], track_id=7, colour=RED)

    painted = frame[frame.any(axis=2)]
    assert (painted == np.array(RED, dtype=np.uint8)).all(axis=1).any()


def test_annotate_player_marks_the_foot_position(frame):
    annotate_player(frame, [80, 60, 120, 140], track_id=7, colour=RED)

    # The label rectangle is centred just below the bbox bottom edge (y2 = 140).
    assert frame[145, 100].tolist() != [0, 0, 0]
    assert not frame[:60].any()


def test_annotate_player_shifts_the_label_for_three_digit_ids(frame):
    single = np.zeros((200, 200, 3), dtype=np.uint8)
    annotate_player(single, [80, 60, 120, 140], track_id=7, colour=RED)

    triple = np.zeros((200, 200, 3), dtype=np.uint8)
    annotate_player(triple, [80, 60, 120, 140], track_id=123, colour=RED)

    assert not np.array_equal(single, triple)


def test_annotate_ball_draws_a_triangle_above_the_ball(frame):
    result = annotate_ball(frame, [98, 100, 106, 108], colour=RED)

    assert result is frame
    ys, xs = np.nonzero(frame.any(axis=2))
    assert ys.max() <= 100          # nothing drawn below the ball's top edge
    assert ys.min() >= 100 - 22     # nor higher than the triangle's 20px reach
    assert 92 <= xs.min() and xs.max() <= 112


def test_annotate_ball_fills_with_the_requested_colour(frame):
    annotate_ball(frame, [98, 100, 106, 108], colour=RED)

    painted = frame[frame.any(axis=2)]
    assert (painted == np.array(RED, dtype=np.uint8)).all(axis=1).any()


def test_annotate_ball_accepts_float_bboxes(frame):
    annotate_ball(frame, [98.4, 100.7, 106.2, 108.9], colour=RED)

    assert frame.any()
