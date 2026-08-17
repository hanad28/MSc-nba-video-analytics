"""
geometry.py

Generic geometric helper functions: point distance and foot-point calculations.
"""
from typing import Tuple

import numpy as np


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Return the Euclidean distance between two 2D points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def foot_point(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Return the bottom-centre point of a bounding box, approximating a player's foot position."""
    return (x1 + x2) / 2.0, y2
