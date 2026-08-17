"""
court_template.py

Metric positions of the 18 court keypoints on a regulation NBA half-court plane,
providing the target correspondences that Homography maps detected keypoints onto.
"""
from __future__ import annotations

# NBA regulation dimensions. These match config/default.yaml's metrics section;
# FIBA differs (28.0 x 15.0, uniform 6.75 m three-point line), so a FIBA clip
# would need its own table rather than a rescaling of this one.
COURT_LENGTH_M = 28.65
COURT_WIDTH_M = 15.24

LANE_WIDTH_M = 4.88
FREE_THROW_DISTANCE_M = 5.79      # 19 ft, baseline to free-throw line
CORNER_THREE_INSET_M = 0.914      # 3 ft, sideline to the corner three-point line

# Derived rather than tabulated, so a dimension change propagates correctly.
_LANE_A = (COURT_WIDTH_M - LANE_WIDTH_M) / 2
_LANE_B = COURT_WIDTH_M - _LANE_A
_THREE_A = CORNER_THREE_INSET_M
_THREE_B = COURT_WIDTH_M - CORNER_THREE_INSET_M
_CENTRE_X = COURT_LENGTH_M / 2
_RIGHT_FREE_THROW_X = COURT_LENGTH_M - FREE_THROW_DISTANCE_M

# Sideline A is y = 0 and sideline B is y = COURT_WIDTH_M. They are NOT 'near'
# and 'far': which one faces the camera varies by broadcast, and index 0 is not
# reliably the near corner across the three evaluation clips.
TEMPLATE_POINTS_M: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (0.0, _THREE_A),
    2: (0.0, _LANE_A),
    3: (0.0, _LANE_B),
    4: (0.0, _THREE_B),
    5: (0.0, COURT_WIDTH_M),
    6: (_CENTRE_X, COURT_WIDTH_M),
    7: (_CENTRE_X, 0.0),
    8: (FREE_THROW_DISTANCE_M, _LANE_A),
    9: (FREE_THROW_DISTANCE_M, _LANE_B),
    10: (COURT_LENGTH_M, COURT_WIDTH_M),
    11: (COURT_LENGTH_M, _THREE_B),
    12: (COURT_LENGTH_M, _LANE_B),
    13: (COURT_LENGTH_M, _LANE_A),
    14: (COURT_LENGTH_M, _THREE_A),
    15: (COURT_LENGTH_M, 0.0),
    16: (_RIGHT_FREE_THROW_X, _LANE_A),
    17: (_RIGHT_FREE_THROW_X, _LANE_B),
}

KEYPOINT_NAMES: dict[int, str] = {
    0: 'left baseline at sideline A',
    1: 'left corner three-point, side A',
    2: 'left lane corner at baseline, side A',
    3: 'left lane corner at baseline, side B',
    4: 'left corner three-point, side B',
    5: 'left baseline at sideline B',
    6: 'centre line at sideline B',
    7: 'centre line at sideline A',
    8: 'left free-throw line, side A',
    9: 'left free-throw line, side B',
    10: 'right baseline at sideline B',
    11: 'right corner three-point, side B',
    12: 'right lane corner at baseline, side B',
    13: 'right lane corner at baseline, side A',
    14: 'right corner three-point, side A',
    15: 'right baseline at sideline A',
    16: 'right free-throw line, side A',
    17: 'right free-throw line, side B',
}

# The mirror pairing used as flip_idx when training. Kept here rather than only
# in training/split_court_keypoints.py because it is a property of the layout,
# not of the training run, and a future comparator needs the same mapping.
FLIP_INDEX: list[int] = [15, 14, 13, 12, 11, 10, 6, 7, 16, 17, 5, 4, 3, 2, 1, 0, 8, 9]

NUM_KEYPOINTS = 18


def template_point(index: int) -> tuple[float, float]:
    """Return one keypoint's metric court position, raising on an index outside the 18-point layout."""
    if index not in TEMPLATE_POINTS_M:
        raise KeyError(
            f'No template point for keypoint index {index}; '
            f'valid indices are 0 to {NUM_KEYPOINTS - 1}.'
        )
    return TEMPLATE_POINTS_M[index]


def template_array(indices: list[int]) -> list[tuple[float, float]]:
    """Return metric court positions for a list of keypoint indices, in the order given."""
    return [template_point(index) for index in indices]


def to_pixels(point_m: tuple[float, float], width_px: int, height_px: int) -> tuple[float, float]:
    """Scale a metric court position into a top-down image of the given pixel size, for drawing only."""
    # Drawing only, never measurement: the court render in assets/ has aspect
    # 1.863 against this table's 1.880, so scaled marks sit up to ~1% off its
    # painted lines. Measurement works in metres and never round-trips pixels.
    x_m, y_m = point_m
    return (x_m / COURT_LENGTH_M * width_px, y_m / COURT_WIDTH_M * height_px)
