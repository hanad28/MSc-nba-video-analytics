"""
player_metrics.py converts CourtMapper's per-frame metric court positions into
per-track distance covered and windowed speed, suppressing any speed measured
across a gap too long to mean an instantaneous rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SPEED_WINDOW = 5


@dataclass
class TrackSpeed:
    """One track's speed on one frame, carrying both the span it was averaged over and the gap since its previous sighting."""

    # Two different spans, named so neither can be read as the other. The
    # figure a consumer needs to judge how instantaneous a speed is, is
    # window_frames: the speed is a mean over that whole span, not over the
    # last hop. last_gap_frames describes only the most recent displacement,
    # which is the smaller number whenever a track was seen densely and then
    # sparsely; reporting it alone understates the span the speed covers.
    #
    # Both are kept rather than one: window_frames answers 'over how long is
    # this an average', last_gap_frames answers 'how stale is the latest
    # observation', and neither substitutes for the other.
    speed_ms: float
    window_frames: int
    last_gap_frames: int


@dataclass
class MetricsReport:
    """Per-clip totals and availability counts for the distance and speed measurement."""

    n_frames: int = 0
    total_distance_m: dict[int, float] = field(default_factory=dict)
    displacements_measured: int = 0
    speeds_computed: int = 0
    speeds_suppressed_by_gap: int = 0

    def summary(self) -> str:
        """One-line summary of distance covered and how many speeds the gap rule suppressed."""
        attempted = self.speeds_computed + self.speeds_suppressed_by_gap
        rate = self.speeds_computed / attempted if attempted else float('nan')
        return (
            f'[metrics] {len(self.total_distance_m)} tracks over {self.n_frames} frames; '
            f'{self.displacements_measured} displacements measured, '
            f'{self.speeds_computed} speeds computed ({rate:.1%} of attempts), '
            f'{self.speeds_suppressed_by_gap} suppressed by the gap rule.'
        )


class PlayerMetrics:
    """Computes per-track distance covered and windowed speed in metres from mapped court positions."""

    # A displacement spanning more than this many frames contributes distance
    # but NO speed. Chosen from the measured gap distribution across all three
    # clips (3,841 displacements): gaps of 1 to 5 frames cover 99.5%, and above
    # 5 the counts fall to single digits and the values scatter: 6, 7, 8, 9,
    # 10, 11, 16, 17, 19, then 46 with nothing in between. The 46-frame gaps
    # are clip_3's contiguous unmapped run, and a speed averaged over 1.53
    # seconds of a fast break could conceal almost anything; it is a different
    # quantity from an instantaneous speed wearing the same name.
    #
    # A class constant rather than a config key precisely because it is a
    # reasoned choice from a measured distribution rather than a swept
    # parameter; a config key would imply it had been tuned.
    MAX_SPEED_GAP = 5

    def __init__(self, fps: float, speed_window: int = DEFAULT_SPEED_WINDOW) -> None:
        if fps <= 0.0:
            raise ValueError(f'fps must be positive, got {fps}.')
        if speed_window < 1:
            raise ValueError(f'speed_window must be at least 1 frame, got {speed_window}.')

        self.fps = fps
        self.speed_window = speed_window

    def _windowed_speed(
        self,
        history: list[tuple[int, int, float]],
        frame_idx: int,
    ) -> tuple[float, int] | None:
        """Return the speed over the trailing window and the number of frames it spans, or None when the window holds nothing."""
        window_start = frame_idx - self.speed_window
        in_window = [
            (from_frame, to_frame, distance)
            for from_frame, to_frame, distance in history
            if to_frame > window_start
        ]
        if not in_window:
            return None

        # The denominator is ELAPSED time across the window, never the number
        # of frames the track was seen in it. Dividing by
        # frames_present / fps reads plausibly and is the central arithmetic
        # error to avoid: a player seen in 5 of 15 frames would have their
        # speed computed as though those 5 sightings were consecutive,
        # inflating it roughly threefold.
        #
        # The span is measured from the FROM-frame of the earliest displacement
        # in the window, not from the frame it landed on. A track seen only at
        # frames 0 and 5 covered its distance over 5 frames of elapsed time,
        # not the 1 frame its single sighting occupies; measuring from the
        # to-frame would reintroduce the same inflation the window exists to
        # avoid, just one level down.
        earliest_from = min(from_frame for from_frame, _, _ in in_window)
        elapsed_frames = max(1, frame_idx - earliest_from)
        elapsed_seconds = elapsed_frames / self.fps

        # No correction factor is applied anywhere in this class. A multiplier
        # chosen because distances 'look overestimated' would have no
        # measurement behind it, and would most likely be compensating for a
        # wrong denominator above, two errors partly cancelling. If a distance
        # is wrong, the cause is upstream and belongs there.
        speed = sum(distance for _, _, distance in in_window) / elapsed_seconds
        return speed, elapsed_frames

    def compute(
        self,
        court_positions: list[dict[int, tuple[float, float]]],
    ) -> tuple[list[dict[int, float]], list[dict[int, TrackSpeed]], MetricsReport]:
        """Return per-frame cumulative distance and speed per track, plus a report of what was measurable."""
        # Three parallel per-frame lists, matching how player tracks,
        # possession, team assignment and court positions are already carried:
        # list[dict[track_id, ...]], one entry per frame, length always equal
        # to the frame count. Distance and speed are separate lists rather than
        # one dict of pairs because a track can have a distance and no speed
        # (the gap rule suppresses speed alone), and a combined shape would need
        # a sentinel for the missing half, which is exactly the absence-versus-
        # zero confusion this stage exists to avoid.
        report = MetricsReport(n_frames=len(court_positions))
        distances: list[dict[int, float]] = []
        speeds: list[dict[int, TrackSpeed]] = []

        last_seen: dict[int, tuple[int, tuple[float, float]]] = {}
        cumulative: dict[int, float] = {}
        history: dict[int, list[tuple[int, int, float]]] = {}

        for frame_idx, frame_positions in enumerate(court_positions):
            frame_distances: dict[int, float] = {}
            frame_speeds: dict[int, TrackSpeed] = {}

            for track_id, position in frame_positions.items():
                previous = last_seen.get(track_id)
                last_seen[track_id] = (frame_idx, position)

                if previous is None:
                    # First sighting: the track exists but has covered nothing
                    # yet, so its cumulative distance is a real 0.0 rather than
                    # an absent measurement.
                    cumulative.setdefault(track_id, 0.0)
                    frame_distances[track_id] = cumulative[track_id]
                    continue

                previous_frame, previous_position = previous
                gap = frame_idx - previous_frame
                displacement = float(np.hypot(
                    position[0] - previous_position[0],
                    position[1] - previous_position[1],
                ))

                # Cumulative distance sums EVERY displacement, including those
                # spanning gaps: the player did travel that distance, and only
                # the rate at which they covered it is uncertain. Excluding
                # gap-spanning displacements would understate distance covered
                # for no good reason.
                cumulative[track_id] = cumulative.get(track_id, 0.0) + displacement
                frame_distances[track_id] = cumulative[track_id]
                report.displacements_measured += 1

                if gap > self.MAX_SPEED_GAP:
                    # No speed entry at all, rather than 0.0. A zero is
                    # indistinguishable in the output from a stationary player,
                    # and absence and zero are different measurements.
                    report.speeds_suppressed_by_gap += 1
                    # The track's accumulated history is DISCARDED, not merely
                    # skipped. A suppressed gap means the intervening motion is
                    # unknown, so nothing recorded before it can legitimately
                    # contribute to a window spanning it: retaining those
                    # displacements averages pre-gap distance over a span that
                    # includes the gap itself, diluting the speed by whatever
                    # fraction of the window the gap occupies.
                    #
                    # Structural rather than arithmetic, so it holds at any
                    # speed_window. At the default 5 a suppressed gap of 6+
                    # frames already pushes the older entries out and the bug
                    # is unreachable, but speed_window is a config key, and at
                    # 10 a track seen at frames 0-3, absent 4-9, then seen at
                    # 10 and 11 would report 9.0 m/s against a true 30.0.
                    history.pop(track_id, None)
                    continue

                history.setdefault(track_id, []).append((previous_frame, frame_idx, displacement))
                windowed = self._windowed_speed(history[track_id], frame_idx)
                if windowed is None:
                    continue

                speed, window_frames = windowed
                frame_speeds[track_id] = TrackSpeed(
                    speed_ms=speed, window_frames=window_frames, last_gap_frames=gap,
                )
                report.speeds_computed += 1

            distances.append(frame_distances)
            speeds.append(frame_speeds)

        report.total_distance_m = dict(cumulative)
        return distances, speeds, report
