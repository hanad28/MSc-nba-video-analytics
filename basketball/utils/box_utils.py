"""
box_utils.py

Bounding box helper functions: centroid and width calculations.
"""
from typing import Tuple


def bbox_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Return the (cx, cy) centre point of a bounding box."""
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_width(x1: float, x2: float) -> float:
    """Return the width of a bounding box."""
    return x2 - x1
