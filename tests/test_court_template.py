"""Unit tests for the metric court template: coverage, the mirror involution and the drawing-only pixel scaling."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from basketball.keypoints.court_template import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    FLIP_INDEX,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    TEMPLATE_POINTS_M,
    template_array,
    template_point,
    to_pixels,
)

ALL_INDICES = set(range(NUM_KEYPOINTS))


def test_template_points_cover_exactly_the_eighteen_indices():
    assert set(TEMPLATE_POINTS_M) == ALL_INDICES


def test_keypoint_names_cover_exactly_the_eighteen_indices():
    assert set(KEYPOINT_NAMES) == ALL_INDICES


def test_flip_index_covers_exactly_the_eighteen_indices():
    # A permutation, not merely the right length -- ultralytics' own flip_idx
    # check is a length check alone, which the identity mapping also passes.
    assert len(FLIP_INDEX) == NUM_KEYPOINTS
    assert sorted(FLIP_INDEX) == sorted(ALL_INDICES)


def test_flip_index_is_an_involution():
    # Applying the mirror twice must return the original index; anything else
    # is not a reflection and would mislabel points under horizontal flip.
    for index in range(NUM_KEYPOINTS):
        assert FLIP_INDEX[FLIP_INDEX[index]] == index


def test_every_pair_mirrors_about_the_halfway_line():
    # The geometric content of FLIP_INDEX: a point and its mirror share a y and
    # their x values sum to the court length.
    for index in range(NUM_KEYPOINTS):
        x_m, y_m = TEMPLATE_POINTS_M[index]
        mirror_x_m, mirror_y_m = TEMPLATE_POINTS_M[FLIP_INDEX[index]]

        assert mirror_x_m == pytest.approx(COURT_LENGTH_M - x_m)
        assert mirror_y_m == pytest.approx(y_m)


def test_flip_index_matches_the_list_used_for_training():
    # The training script owns the copy the model was actually trained with;
    # a silent divergence would mean the template describes a different layout
    # from the one the detector learned.
    source = Path('training/split_court_keypoints.py').read_text(encoding='utf-8')
    committed = source.split('FLIP_IDX = ')[1].split(']')[0] + ']'

    assert FLIP_INDEX == eval(committed)


def test_dimension_constants_match_the_config_metrics_section():
    config = yaml.safe_load(Path('config/default.yaml').read_text(encoding='utf-8'))

    assert COURT_LENGTH_M == config['metrics']['court_length_m']
    assert COURT_WIDTH_M == config['metrics']['court_width_m']


def test_every_point_lies_within_the_court_bounds():
    for index, (x_m, y_m) in TEMPLATE_POINTS_M.items():
        assert 0.0 <= x_m <= COURT_LENGTH_M, f'keypoint {index} x is off court'
        assert 0.0 <= y_m <= COURT_WIDTH_M, f'keypoint {index} y is off court'


def test_no_two_keypoints_share_a_position():
    # Two landmarks at one position would give the homography a degenerate
    # correspondence pair.
    assert len(set(TEMPLATE_POINTS_M.values())) == NUM_KEYPOINTS


def test_template_point_returns_the_tabulated_position():
    assert template_point(0) == (0.0, 0.0)
    assert template_point(15) == (COURT_LENGTH_M, 0.0)


def test_template_point_raises_on_an_index_past_the_end():
    with pytest.raises(KeyError, match='valid indices are 0 to 17'):
        template_point(NUM_KEYPOINTS)


def test_template_point_raises_on_a_negative_index():
    # -1 must not silently wrap to the last entry the way list indexing would.
    with pytest.raises(KeyError, match='valid indices are 0 to 17'):
        template_point(-1)


def test_template_array_preserves_the_order_given():
    requested = [7, 0, 15, 3]

    assert template_array(requested) == [TEMPLATE_POINTS_M[index] for index in requested]


def test_template_array_handles_an_empty_request():
    assert template_array([]) == []


def test_template_array_propagates_the_index_guard():
    with pytest.raises(KeyError, match='valid indices are 0 to 17'):
        template_array([0, NUM_KEYPOINTS])


def test_to_pixels_maps_the_origin_to_the_origin():
    assert to_pixels((0.0, 0.0), 1000, 500) == (0.0, 0.0)


def test_to_pixels_maps_the_far_corner_to_the_image_extent():
    corner = to_pixels((COURT_LENGTH_M, COURT_WIDTH_M), 1000, 500)

    assert corner == pytest.approx((1000.0, 500.0))


def test_to_pixels_scales_the_centre_to_the_image_centre():
    centre = to_pixels((COURT_LENGTH_M / 2, COURT_WIDTH_M / 2), 1000, 500)

    assert centre == pytest.approx((500.0, 250.0))
