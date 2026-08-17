"""Unit tests for KeypointAnnotator: index labelling, the render threshold and colour non-collision."""
from __future__ import annotations

import numpy as np
import pytest

from basketball.annotators.ball_annotator import BallAnnotator
from basketball.annotators.event_annotator import EventAnnotator
from basketball.annotators.keypoint_annotator import KeypointAnnotator
from basketball.annotators.player_annotator import PlayerAnnotator
from basketball.annotators.possession_annotator import PossessionAnnotator
from basketball.keypoints.court_keypoints import Keypoint
from basketball.keypoints.court_template import NUM_KEYPOINTS

CLIP_3_RED_OVERRIDE = (0, 0, 200)


@pytest.fixture
def frames() -> list[np.ndarray]:
    return [np.zeros((200, 300, 3), dtype=np.uint8) for _ in range(2)]


def keypoints(confidence: float = 0.9, count: int = NUM_KEYPOINTS) -> list[Keypoint]:
    """A frame's worth of keypoints spread across the image at one shared confidence."""
    return [
        Keypoint(index=i, x=20.0 + i * 12, y=30.0 + (i % 5) * 25, confidence=confidence)
        for i in range(count)
    ]


def colours_used(frame: np.ndarray) -> set[tuple[int, int, int]]:
    pixels = frame[frame.any(axis=2)]
    return {tuple(int(channel) for channel in pixel) for pixel in pixels}


def test_confident_keypoints_are_rendered(frames):
    output = KeypointAnnotator().draw(frames, [keypoints(), []])

    assert KeypointAnnotator.KEYPOINT_COLOUR in colours_used(output[0])
    assert not output[1].any()


def test_the_index_label_is_drawn_beside_each_point():
    # The verdicted audit asks whether index k is on the correct landmark,
    # which is unanswerable from unlabelled dots -- so a labelled point must
    # paint materially more than the dot alone.
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    one_point = [Keypoint(index=17, x=150.0, y=100.0, confidence=0.9)]

    annotated = KeypointAnnotator().draw([frame], [one_point])[0]
    dot_only_pixels = np.pi * KeypointAnnotator.RADIUS ** 2

    assert int(np.count_nonzero(annotated.any(axis=2))) > dot_only_pixels * 1.5


def test_two_different_indices_render_differently():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    annotator = KeypointAnnotator()

    first = annotator.draw([frame], [[Keypoint(index=1, x=150.0, y=100.0, confidence=0.9)]])[0]
    second = annotator.draw([frame], [[Keypoint(index=16, x=150.0, y=100.0, confidence=0.9)]])[0]

    assert not np.array_equal(first, second)


def test_the_render_threshold_suppresses_low_confidence_points(frames):
    low = [Keypoint(index=i, x=20.0 + i * 12, y=50.0, confidence=0.2) for i in range(NUM_KEYPOINTS)]

    output = KeypointAnnotator(render_threshold=0.5).draw(frames, [low, []])

    assert not output[0].any()


def test_the_render_threshold_is_configurable(frames):
    modest = [Keypoint(index=i, x=20.0 + i * 12, y=50.0, confidence=0.3) for i in range(NUM_KEYPOINTS)]

    permissive = KeypointAnnotator(render_threshold=0.25).draw(frames, [modest, []])
    strict = KeypointAnnotator(render_threshold=0.9).draw(frames, [modest, []])

    assert permissive[0].any()
    assert not strict[0].any()


def test_the_default_render_threshold_is_the_class_constant():
    assert KeypointAnnotator().render_threshold == KeypointAnnotator.DEFAULT_RENDER_THRESHOLD


def test_a_point_exactly_at_the_threshold_is_rendered(frames):
    at_threshold = [Keypoint(index=0, x=100.0, y=100.0, confidence=0.5)]

    output = KeypointAnnotator(render_threshold=0.5).draw(frames, [at_threshold, []])

    assert output[0].any()


def test_colour_does_not_collide_with_any_existing_annotator_colour():
    # Referenced by attribute rather than by literal so a future recolour of
    # any existing annotator is caught here rather than silently colliding.
    existing = {
        PlayerAnnotator.DEFAULT_TEAM_1_COLOUR,
        PlayerAnnotator.DEFAULT_TEAM_2_COLOUR,
        PlayerAnnotator.UNKNOWN_TEAM_COLOUR,
        PossessionAnnotator.HOLDER_COLOUR,
        PossessionAnnotator.CAPTION_COLOUR,
        BallAnnotator.COLOUR,
        EventAnnotator.PASS_COLOUR,
        EventAnnotator.INTERCEPTION_COLOUR,
        EventAnnotator.UNCLASSIFIED_COLOUR,
        CLIP_3_RED_OVERRIDE,
    }

    assert KeypointAnnotator.KEYPOINT_COLOUR not in existing
    assert KeypointAnnotator.LABEL_COLOUR not in existing


def test_mismatched_frame_count_raises_naming_both_counts(frames):
    with pytest.raises(ValueError, match='1 keypoint frames for 2 video frames'):
        KeypointAnnotator().draw(frames, [keypoints()])


def test_frames_are_not_mutated_in_place(frames):
    KeypointAnnotator().draw(frames, [keypoints(), keypoints()])

    assert all(not frame.any() for frame in frames)


def test_empty_keypoint_lists_return_frames_unchanged(frames):
    output = KeypointAnnotator().draw(frames, [[], []])

    assert all(np.array_equal(after, before) for after, before in zip(output, frames))


def test_a_point_near_the_frame_edge_does_not_raise_or_grow_the_frame():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    edge = [
        Keypoint(index=0, x=79.0, y=59.0, confidence=0.9),
        Keypoint(index=17, x=0.0, y=0.0, confidence=0.9),
    ]

    output = KeypointAnnotator().draw([frame], [edge])

    assert output[0].shape == (60, 80, 3)


def test_a_confident_keypoint_at_the_origin_is_still_rendered():
    # Mirrors the detector's (0, 0) non-sentinel contract: the origin is
    # keypoint 0's legitimate template corner, not an absence marker.
    frame = np.zeros((200, 300, 3), dtype=np.uint8)

    output = KeypointAnnotator().draw([frame], [[Keypoint(index=0, x=0.0, y=0.0, confidence=0.95)]])

    assert output[0].any()
