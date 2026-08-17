"""Unit tests for geometry and box utility functions."""
from basketball.utils.geometry import euclidean_distance, foot_point
from basketball.utils.box_utils import bbox_center, bbox_width


def test_bbox_center():
    cx, cy = bbox_center(0, 0, 4, 6)
    assert cx == 2.0
    assert cy == 3.0


def test_bbox_width():
    assert bbox_width(0, 4) == 4.0


def test_euclidean_distance_zero():
    assert euclidean_distance((1.0, 1.0), (1.0, 1.0)) == 0.0


def test_euclidean_distance_known():
    assert abs(euclidean_distance((0.0, 0.0), (3.0, 4.0)) - 5.0) < 1e-6


def test_foot_point():
    fx, fy = foot_point(0, 0, 4, 6)
    assert fx == 2.0
    assert fy == 6.0
