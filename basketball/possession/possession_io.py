"""
possession_io.py

The single source of truth for possession's ball input: the adopted gate
followed by interpolation, in main.py's order.
"""
from __future__ import annotations

from typing import NamedTuple

from basketball.detection.ball_detector import BallDetection
from basketball.detection.ball_interpolation import BallInterpolator

BALL_DETECTIONS_CACHE_TEMPLATE = 'data/processed/{clip}/ball_detections.pkl'


class BallInput(NamedTuple):
    """The two ball tracks possession work needs: gated real detections, and the interpolated track the tracker consumes."""

    gated: list[dict[int, BallDetection]]
    filled: list[dict[int, BallDetection]]


def possession_ball_input(
    raw_ball_detections: list[dict[int, BallDetection]],
) -> BallInput:
    """Gate raw ball detections with the adopted global-trajectory gate and then interpolate, exactly as main.py does, returning both stages."""
    # Gate before interpolating: fill_missing(raw) on its own would
    # interpolate THROUGH the false detections the gate exists to remove,
    # producing possession numbers from a ball track main.py never
    # generates. This function exists so that ordering lives in exactly one
    # place; call it rather than reproducing the two steps.
    interpolator = BallInterpolator()
    gated = interpolator.filter_detections_global_trajectory(raw_ball_detections)
    filled = interpolator.fill_missing(gated)
    # gated is deliberately returned as well as filled: a labelling tool must
    # show only real detections (an interpolated bbox would invite a
    # fabricated answer), while the tracker needs a position on every frame.
    return BallInput(gated=gated, filled=filled)
