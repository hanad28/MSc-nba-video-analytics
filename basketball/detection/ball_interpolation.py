"""
ball_interpolation.py

Fills gaps in ball detection sequences caused by occlusion or missed
detections, using linear interpolation over the ball's observed trajectory.
Sits immediately after BallDetector in the pipeline, cleaning raw detection
output before possession logic ever sees it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from basketball.detection.ball_detector import BallDetection
from basketball.utils.box_utils import bbox_center
from basketball.utils.geometry import euclidean_distance

MAX_BALL_TRAVEL = 25  # maximum plausible ball displacement in pixels per frame

# Defaults for predicted-position anchoring: unswept starting values, held
# fixed.
BASE_TOLERANCE = 40.0  # pixels of slack allowed around the predicted position
GROWTH_RATE = 15.0  # extra pixels of slack per frame of gap since the last anchor
ANCHOR_HISTORY_SIZE = 4  # number of recent accepted anchors kept for the velocity fit
MAX_ANCHOR_SPAN = 8  # anchors spanning more frames than this are too sparse for a linear fit

# Guard 1 (filter_detections_staleness_aware()) -- targets the stale-but-
# genuine-anchor failure mode: past this many frames since the last accepted
# anchor, its implied trajectory is no longer trusted for gating. Unswept
# starting value, held fixed (chosen from the 6-10 frame range).
STALE_ANCHOR_THRESHOLD = 8

# Guard 2 (is_motion_consistent() / filter_detections_motion_consistency())
# -- targets the single-gate false-positive-acceptance failure mode. All
# unswept starting values, held fixed.
MOTION_CONSISTENCY_HISTORY = 3  # number of most-recent accepted anchors used to estimate recent velocity
MAX_ANGLE_DEVIATION_DEG = 90.0  # candidate's implied direction may deviate from the recent velocity direction by at most this many degrees
MIN_REFERENCE_SPEED_PX = 5.0  # below this recent speed (px/frame), direction is not well-defined enough to gate on angle
MAX_SPEED_RATIO = 3.0  # candidate's implied speed may exceed the recent speed by at most this multiple
MIN_PLAUSIBLE_SPEED_CAP_PX = 60.0  # ...or this floor (px/frame), whichever is larger -- avoids punishing legitimate acceleration when recent speed was small

# Combined gate (filter_detections_combined()) -- freshness thresholds
# gating the motion-consistency screening layer. Both start at the same
# numeric value as STALE_ANCHOR_THRESHOLD / MAX_ANCHOR_SPAN (8), since they
# answer the same underlying "is this recent-history reference still
# trustworthy" question those constants already answer for their own gates.
# Kept as separate, independently-tunable constants rather than silently
# reused from either, because these gate a different layer -- the screen
# inside the combined gate, not a whole gate's override/fit behaviour -- and
# need not share a value.
COMBINED_FRESHNESS_GAP_THRESHOLD = 8  # frame_gap from the last accepted anchor to the candidate must be at most this for the screen to be considered trustworthy
COMBINED_REFERENCE_SPAN_THRESHOLD = 8  # the reference window's own anchors (newest - oldest, within the last MOTION_CONSISTENCY_HISTORY accepted) must span at most this many frames for the screen to be considered trustworthy

# Option B (filter_detections_global_trajectory()) -- global, non-causal
# trajectory selection, swept by scripts/evaluate_gate_candidates.py;
# outputs in results/ball_detection/gate_candidate_evaluation/.
GLOBAL_MAX_SPEED_PX = 100.0  # S_MAX: hard straight-line speed cap (px/frame) an edge may imply; exists to forbid teleports, not to model typical motion
GLOBAL_SKIP_PENALTY = 1000.0  # LAMBDA: cost, in the same px^2/frame units as the motion term, of leaving one detection off the chosen path. Adopted at 1000, located by the gate-candidate evaluation (a 500 penalty misses the W4 flight on real data).
GLOBAL_MAX_EDGE_GAP = 45  # K_MAX: edges spanning more frames than this are long-silence resets -- allowed at zero motion cost, since the continuity assumption is void after ~1.5s of detector silence at 30fps


class BallInterpolator:
    """
    Filters implausible ball detections and interpolates missing positions
    across detection gaps using Pandas linear interpolation.
    """

    def filter_detections(
        self,
        ball_detections: list[dict[int, BallDetection]],
        max_travel: float = MAX_BALL_TRAVEL,
    ) -> list[dict[int, BallDetection]]:
        """Remove detections that jump further than max_travel pixels per frame (default MAX_BALL_TRAVEL) from the previous valid detection."""
        result = [dict(frame) for frame in ball_detections]
        last_good_idx = -1

        for i, frame in enumerate(result):
            if 1 not in frame:
                continue

            if last_good_idx == -1:
                last_good_idx = i
                continue

            current_bbox = frame[1].bbox
            last_good_bbox = result[last_good_idx][1].bbox

            frame_gap = i - last_good_idx
            adjusted_max = max_travel * frame_gap

            cx_cur, cy_cur = bbox_center(*current_bbox)
            cx_last, cy_last = bbox_center(*last_good_bbox)
            distance = euclidean_distance((cx_cur, cy_cur), (cx_last, cy_last))

            if distance > adjusted_max:
                result[i] = {}
            else:
                last_good_idx = i

        return result

    def filter_detections_predicted(
        self,
        ball_detections: list[dict[int, BallDetection]],
        base_tolerance: float = BASE_TOLERANCE,
        growth_rate: float = GROWTH_RATE,
        history_size: int = ANCHOR_HISTORY_SIZE,
        max_anchor_span: int = MAX_ANCHOR_SPAN,
        fallback_travel: float = MAX_BALL_TRAVEL,
    ) -> list[dict[int, BallDetection]]:
        """
        Alternative gate to filter_detections(): rejects detections that fall too
        far from a position predicted by extrapolating the ball's recent velocity,
        rather than from the frozen last-accepted position. Falls back to the
        frozen-point check when the anchor history is too short or too sparse to
        support a linear fit.
        """
        result = [dict(frame) for frame in ball_detections]
        anchors: list[tuple[int, float, float]] = []

        for i, frame in enumerate(result):
            if 1 not in frame:
                continue

            cx_cur, cy_cur = bbox_center(*frame[1].bbox)

            if not anchors:
                anchors.append((i, cx_cur, cy_cur))
                continue

            last_idx, last_cx, last_cy = anchors[-1]
            frame_gap = i - last_idx

            predicted = self._predict_position(anchors, frame_gap, max_anchor_span)

            if predicted is None:
                # Not enough usable history -- reproduce the original frozen-point gate.
                # Note this bootstrap check is stricter than the predicted-position
                # tolerance below, so a ball that is already moving faster than
                # fallback_travel px/frame can never accumulate the second anchor the
                # velocity fit needs. fallback_travel is exposed so a sweep
                # can test loosening it rather than having the value baked in here.
                accepted = euclidean_distance(
                    (cx_cur, cy_cur), (last_cx, last_cy)
                ) <= fallback_travel * frame_gap
            else:
                tolerance = base_tolerance + growth_rate * frame_gap
                accepted = euclidean_distance((cx_cur, cy_cur), predicted) <= tolerance

            if accepted:
                anchors.append((i, cx_cur, cy_cur))
                anchors = anchors[-history_size:]
            else:
                result[i] = {}

        return result

    def trace_predicted_decisions(
        self,
        ball_detections: list[dict[int, BallDetection]],
        base_tolerance: float = BASE_TOLERANCE,
        growth_rate: float = GROWTH_RATE,
        history_size: int = ANCHOR_HISTORY_SIZE,
        max_anchor_span: int = MAX_ANCHOR_SPAN,
        fallback_travel: float = MAX_BALL_TRAVEL,
    ) -> dict[int, dict]:
        """
        Replays filter_detections_predicted()'s gate and records the decision made
        at each frame carrying a raw detection: which path was taken and why, the
        predicted position, the measured distance and the tolerance applied. Purely
        diagnostic -- returns per-frame records rather than filtered detections.
        """
        anchors: list[tuple[int, float, float]] = []
        records: dict[int, dict] = {}

        for i, frame in enumerate(ball_detections):
            if 1 not in frame:
                continue

            cx_cur, cy_cur = bbox_center(*frame[1].bbox)

            if not anchors:
                anchors.append((i, cx_cur, cy_cur))
                records[i] = {
                    'path': 'seed: first anchor',
                    'predicted': None,
                    'reference': None,
                    'distance': None,
                    'tolerance': None,
                    'accepted': True,
                    'frame_gap': None,
                    'anchors_before': 0,
                    'anchor_span': None,
                }
                continue

            last_idx, last_cx, last_cy = anchors[-1]
            frame_gap = i - last_idx
            anchors_before = len(anchors)
            anchor_span = anchors[-1][0] - anchors[0][0]

            predicted, path = self._predict_position_explained(anchors, frame_gap, max_anchor_span)

            if predicted is None:
                reference = (last_cx, last_cy)
                tolerance = fallback_travel * frame_gap
            else:
                reference = predicted
                tolerance = base_tolerance + growth_rate * frame_gap

            distance = euclidean_distance((cx_cur, cy_cur), reference)
            accepted = distance <= tolerance

            records[i] = {
                'path': path,
                'predicted': predicted,
                'reference': reference,
                'distance': distance,
                'tolerance': tolerance,
                'accepted': accepted,
                'frame_gap': frame_gap,
                'anchors_before': anchors_before,
                'anchor_span': anchor_span,
            }

            if accepted:
                anchors.append((i, cx_cur, cy_cur))
                anchors = anchors[-history_size:]

        return records

    def _predict_position(
        self,
        anchors: list[tuple[int, float, float]],
        frame_gap: int,
        max_anchor_span: int,
    ) -> tuple[float, float] | None:
        """Extrapolate the expected ball centre from recent anchor velocity, or None when the history is too short or too sparse to fit."""
        predicted, _ = self._predict_position_explained(anchors, frame_gap, max_anchor_span)

        return predicted

    def _predict_position_explained(
        self,
        anchors: list[tuple[int, float, float]],
        frame_gap: int,
        max_anchor_span: int,
    ) -> tuple[tuple[float, float] | None, str]:
        """Same as _predict_position() but also returns which decision path was taken, for diagnostic instrumentation."""
        if len(anchors) < 2:
            return None, 'fallback: fewer than 2 anchors'

        if anchors[-1][0] - anchors[0][0] > max_anchor_span:
            return None, 'fallback: anchor span exceeded'

        frames = np.array([a[0] for a in anchors], dtype=float)
        xs = np.array([a[1] for a in anchors], dtype=float)
        ys = np.array([a[2] for a in anchors], dtype=float)

        # Least-squares slope of position against frame index, fitted per axis.
        vx = np.polyfit(frames, xs, 1)[0]
        vy = np.polyfit(frames, ys, 1)[0]

        _, last_cx, last_cy = anchors[-1]

        return (last_cx + vx * frame_gap, last_cy + vy * frame_gap), 'predicted'

    def filter_detections_staleness_aware(
        self,
        ball_detections: list[dict[int, BallDetection]],
        stale_anchor_threshold: int = STALE_ANCHOR_THRESHOLD,
        fallback_travel: float = MAX_BALL_TRAVEL,
    ) -> list[dict[int, BallDetection]]:
        """
        Independent alternative gate to filter_detections(): same frozen-
        last-accepted-point distance check, except once the gap since the
        last accepted anchor exceeds stale_anchor_threshold, the anchor's
        implied trajectory is no longer trusted for gating -- the next
        candidate is accepted outright and becomes the new anchor, rather
        than being held to a distance tolerance that keeps growing off an
        increasingly stale reference point. Targets the stale-but-genuine-
        anchor failure mode (confirmed clip_3 172-190, clip_2 163-170): the
        anchor's own motion assumption can go wrong (a catch redirecting the
        ball, a made shot) long before MAX_BALL_TRAVEL * frame_gap grows
        large enough to admit the ball's real return. Does not call or share
        state with filter_detections() or filter_detections_predicted().
        """
        result = [dict(frame) for frame in ball_detections]
        last_good_idx = -1

        for i, frame in enumerate(result):
            if 1 not in frame:
                continue

            if last_good_idx == -1:
                last_good_idx = i
                continue

            frame_gap = i - last_good_idx

            if frame_gap > stale_anchor_threshold:
                last_good_idx = i
                continue

            current_bbox = frame[1].bbox
            last_good_bbox = result[last_good_idx][1].bbox

            cx_cur, cy_cur = bbox_center(*current_bbox)
            cx_last, cy_last = bbox_center(*last_good_bbox)
            distance = euclidean_distance((cx_cur, cy_cur), (cx_last, cy_last))
            adjusted_max = fallback_travel * frame_gap

            if distance > adjusted_max:
                result[i] = {}
            else:
                last_good_idx = i

        return result

    def trace_staleness_aware_decisions(
        self,
        ball_detections: list[dict[int, BallDetection]],
        stale_anchor_threshold: int = STALE_ANCHOR_THRESHOLD,
        fallback_travel: float = MAX_BALL_TRAVEL,
    ) -> dict[int, dict]:
        """
        Replays filter_detections_staleness_aware()'s gate and records the
        decision made at each frame carrying a raw detection: which path was
        taken (seed, stale-reset accept, or frozen-point check), the
        distance, the tolerance, and the anchor frame it was measured
        against. Purely diagnostic -- returns per-frame records rather than
        filtered detections.
        """
        records: dict[int, dict] = {}
        last_good_idx = -1

        for i, frame in enumerate(ball_detections):
            if 1 not in frame:
                continue

            if last_good_idx == -1:
                last_good_idx = i
                records[i] = {
                    'path': 'seed: first accepted detection',
                    'distance': None,
                    'tolerance': None,
                    'anchor_frame': None,
                    'frame_gap': None,
                    'accepted': True,
                }
                continue

            frame_gap = i - last_good_idx

            if frame_gap > stale_anchor_threshold:
                records[i] = {
                    'path': f'stale-reset accept (gap {frame_gap} > {stale_anchor_threshold})',
                    'distance': None,
                    'tolerance': None,
                    'anchor_frame': last_good_idx,
                    'frame_gap': frame_gap,
                    'accepted': True,
                }
                last_good_idx = i
                continue

            cx_cur, cy_cur = bbox_center(*ball_detections[i][1].bbox)
            cx_last, cy_last = bbox_center(*ball_detections[last_good_idx][1].bbox)
            distance = euclidean_distance((cx_cur, cy_cur), (cx_last, cy_last))
            tolerance = fallback_travel * frame_gap
            accepted = distance <= tolerance

            records[i] = {
                'path': 'frozen-point check',
                'distance': distance,
                'tolerance': tolerance,
                'anchor_frame': last_good_idx,
                'frame_gap': frame_gap,
                'accepted': accepted,
            }

            if accepted:
                last_good_idx = i

        return records

    def is_motion_consistent(
        self,
        recent_accepted: list[tuple[int, float, float]],
        candidate: tuple[int, float, float],
        max_angle_deviation_deg: float = MAX_ANGLE_DEVIATION_DEG,
        min_reference_speed: float = MIN_REFERENCE_SPEED_PX,
        max_speed_ratio: float = MAX_SPEED_RATIO,
        min_plausible_speed_cap: float = MIN_PLAUSIBLE_SPEED_CAP_PX,
    ) -> bool:
        """
        Checks whether a candidate detection is broadly consistent with the
        direction and speed implied by the most recent accepted detections,
        rather than only its distance from the single last accepted point.
        Deliberately lighter-weight than filter_detections_predicted()'s
        linear fit: estimates one net velocity vector across the oldest and
        newest of the last few accepted anchors rather than extrapolating a
        predicted position, so this stays a plausibility check rather than a
        second trajectory-fit gate with its own staleness failure mode. With
        fewer than two prior accepted detections there is no recent velocity
        to compare against, so the check is inconclusive and returns True
        rather than forcing a reject.
        """
        consistent, _ = self._motion_consistency_explained(
            recent_accepted, candidate, max_angle_deviation_deg,
            min_reference_speed, max_speed_ratio, min_plausible_speed_cap,
        )

        return consistent

    def _motion_consistency_explained(
        self,
        recent_accepted: list[tuple[int, float, float]],
        candidate: tuple[int, float, float],
        max_angle_deviation_deg: float,
        min_reference_speed: float,
        max_speed_ratio: float,
        min_plausible_speed_cap: float,
    ) -> tuple[bool, str]:
        """Same as is_motion_consistent() but also returns the reason, for diagnostic instrumentation."""
        if len(recent_accepted) < 2:
            return True, 'inconclusive: fewer than 2 prior accepted detections'

        oldest_idx, oldest_x, oldest_y = recent_accepted[0]
        newest_idx, newest_x, newest_y = recent_accepted[-1]
        candidate_idx, candidate_x, candidate_y = candidate

        reference_gap = newest_idx - oldest_idx
        candidate_gap = candidate_idx - newest_idx

        if reference_gap <= 0 or candidate_gap <= 0:
            return True, 'inconclusive: degenerate frame gap in history or candidate'

        v_recent = np.array([(newest_x - oldest_x) / reference_gap, (newest_y - oldest_y) / reference_gap])
        v_candidate = np.array([(candidate_x - newest_x) / candidate_gap, (candidate_y - newest_y) / candidate_gap])

        speed_recent = float(np.linalg.norm(v_recent))
        speed_candidate = float(np.linalg.norm(v_candidate))

        allowed_speed = max(speed_recent * max_speed_ratio, min_plausible_speed_cap)
        if speed_candidate > allowed_speed:
            return False, f'inconsistent: implied speed {speed_candidate:.2f} px/frame > allowed {allowed_speed:.2f} px/frame'

        if speed_recent >= min_reference_speed and speed_candidate >= min_reference_speed:
            cos_angle = np.dot(v_recent, v_candidate) / (speed_recent * speed_candidate)
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_angle)))

            if angle_deg > max_angle_deviation_deg:
                return False, f'inconsistent: direction deviates {angle_deg:.1f} deg > {max_angle_deviation_deg} deg from recent velocity'

        return True, 'consistent'

    def filter_detections_motion_consistency(
        self,
        ball_detections: list[dict[int, BallDetection]],
        history_size: int = MOTION_CONSISTENCY_HISTORY,
        max_angle_deviation_deg: float = MAX_ANGLE_DEVIATION_DEG,
        min_reference_speed: float = MIN_REFERENCE_SPEED_PX,
        max_speed_ratio: float = MAX_SPEED_RATIO,
        min_plausible_speed_cap: float = MIN_PLAUSIBLE_SPEED_CAP_PX,
    ) -> list[dict[int, BallDetection]]:
        """
        Independent gate built on is_motion_consistent(): accepts a
        candidate only if it is broadly consistent with the direction and
        speed implied by the last history_size accepted detections. Targets
        the single-gate false-positive-acceptance failure mode (confirmed
        clip_3 frame 90, clip_2 frame 50): a false positive can fall within a
        pure distance tolerance while still moving in an implausible
        direction or at an implausible speed relative to the ball's recent
        motion. Does not call or share state with filter_detections() or
        filter_detections_predicted().
        """
        result = [dict(frame) for frame in ball_detections]
        accepted_anchors: list[tuple[int, float, float]] = []

        for i, frame in enumerate(result):
            if 1 not in frame:
                continue

            cx_cur, cy_cur = bbox_center(*frame[1].bbox)
            candidate = (i, cx_cur, cy_cur)

            if self.is_motion_consistent(
                accepted_anchors, candidate, max_angle_deviation_deg,
                min_reference_speed, max_speed_ratio, min_plausible_speed_cap,
            ):
                accepted_anchors.append(candidate)
                accepted_anchors = accepted_anchors[-history_size:]
            else:
                result[i] = {}

        return result

    def trace_motion_consistency_decisions(
        self,
        ball_detections: list[dict[int, BallDetection]],
        history_size: int = MOTION_CONSISTENCY_HISTORY,
        max_angle_deviation_deg: float = MAX_ANGLE_DEVIATION_DEG,
        min_reference_speed: float = MIN_REFERENCE_SPEED_PX,
        max_speed_ratio: float = MAX_SPEED_RATIO,
        min_plausible_speed_cap: float = MIN_PLAUSIBLE_SPEED_CAP_PX,
    ) -> dict[int, dict]:
        """
        Replays filter_detections_motion_consistency()'s gate and records
        the reason for each frame's decision via
        _motion_consistency_explained(). Purely diagnostic -- returns
        per-frame records rather than filtered detections.
        """
        records: dict[int, dict] = {}
        accepted_anchors: list[tuple[int, float, float]] = []

        for i, frame in enumerate(ball_detections):
            if 1 not in frame:
                continue

            cx_cur, cy_cur = bbox_center(*frame[1].bbox)
            candidate = (i, cx_cur, cy_cur)

            accepted, reason = self._motion_consistency_explained(
                accepted_anchors, candidate, max_angle_deviation_deg,
                min_reference_speed, max_speed_ratio, min_plausible_speed_cap,
            )

            records[i] = {
                'reason': reason,
                'accepted': accepted,
                'anchors_before': len(accepted_anchors),
            }

            if accepted:
                accepted_anchors.append(candidate)
                accepted_anchors = accepted_anchors[-history_size:]

        return records

    def filter_detections_combined(
        self,
        ball_detections: list[dict[int, BallDetection]],
        fallback_travel: float = MAX_BALL_TRAVEL,
        freshness_gap_threshold: int = COMBINED_FRESHNESS_GAP_THRESHOLD,
        reference_span_threshold: int = COMBINED_REFERENCE_SPAN_THRESHOLD,
        history_size: int = MOTION_CONSISTENCY_HISTORY,
        max_angle_deviation_deg: float = MAX_ANGLE_DEVIATION_DEG,
        min_reference_speed: float = MIN_REFERENCE_SPEED_PX,
        max_speed_ratio: float = MAX_SPEED_RATIO,
        min_plausible_speed_cap: float = MIN_PLAUSIBLE_SPEED_CAP_PX,
    ) -> list[dict[int, BallDetection]]:
        """
        Combined gate: the last of the causal alternatives. Base
        layer: identical in spirit to filter_detections() -- frozen
        last-accepted-anchor distance check against MAX_BALL_TRAVEL *
        frame_gap, a tolerance that grows unboundedly with the gap. A
        candidate failing this is REJECTed outright, no further checks. This
        base layer alone already provides staleness self-correction: a
        genuinely-returning ball eventually falls back within an
        ever-growing tolerance, the same mechanism already observed
        readmitting the real ball at clip_3 frame 188 under the unmodified
        original gate -- so Guard 1's unconditional-accept-on-reset is not
        needed here and is not used.

        Screening layer: a candidate that passes the base check is
        additionally screened by is_motion_consistent() (reused directly,
        not reimplemented) -- but ONLY when the reference window that check
        would use is itself fresh. Freshness requires BOTH the gap from the
        last accepted anchor to the candidate being small
        (freshness_gap_threshold) AND the reference window's own anchors (up
        to the last history_size accepted) spanning few enough frames
        (reference_span_threshold). Guard 2 on its own only checked
        something like the first condition; the second is the specific fix
        for Guard 2's confirmed bug (clip_1 frames 33-49, 51-63, one real
        fast-break drive) where a history-size-3 reference window built from
        sparse acceptances could still be internally stale -- spanning many
        frames -- even when the gap to the immediate candidate was small.
        When the window is not fresh by either measure, there is no
        trustworthy motion basis to screen against, so the candidate is
        accepted as-is (the base distance check already passed) rather than
        screened by a reference that cannot be trusted -- same
        "inconclusive means accept" principle used elsewhere in this module,
        applied to freshness rather than just anchor count.

        Does not call or share state with filter_detections(),
        filter_detections_predicted(), filter_detections_staleness_aware(),
        or filter_detections_motion_consistency() -- all four are left
        unmodified for comparison.
        """
        result = [dict(frame) for frame in ball_detections]
        accepted_anchors: list[tuple[int, float, float]] = []

        for i, frame in enumerate(result):
            if 1 not in frame:
                continue

            cx_cur, cy_cur = bbox_center(*frame[1].bbox)
            candidate = (i, cx_cur, cy_cur)

            if not accepted_anchors:
                accepted_anchors.append(candidate)
                continue

            last_idx, last_cx, last_cy = accepted_anchors[-1]
            frame_gap = i - last_idx
            distance = euclidean_distance((cx_cur, cy_cur), (last_cx, last_cy))
            tolerance = fallback_travel * frame_gap

            if distance > tolerance:
                result[i] = {}
                continue

            reference_window = accepted_anchors[-history_size:]
            span = reference_window[-1][0] - reference_window[0][0]
            fresh = (
                frame_gap <= freshness_gap_threshold
                and len(reference_window) >= 2
                and span <= reference_span_threshold
            )

            if not fresh:
                accepted_anchors.append(candidate)
                accepted_anchors = accepted_anchors[-history_size:]
                continue

            if self.is_motion_consistent(
                reference_window, candidate, max_angle_deviation_deg,
                min_reference_speed, max_speed_ratio, min_plausible_speed_cap,
            ):
                accepted_anchors.append(candidate)
                accepted_anchors = accepted_anchors[-history_size:]
            else:
                result[i] = {}

        return result

    def trace_combined_decisions(
        self,
        ball_detections: list[dict[int, BallDetection]],
        fallback_travel: float = MAX_BALL_TRAVEL,
        freshness_gap_threshold: int = COMBINED_FRESHNESS_GAP_THRESHOLD,
        reference_span_threshold: int = COMBINED_REFERENCE_SPAN_THRESHOLD,
        history_size: int = MOTION_CONSISTENCY_HISTORY,
        max_angle_deviation_deg: float = MAX_ANGLE_DEVIATION_DEG,
        min_reference_speed: float = MIN_REFERENCE_SPEED_PX,
        max_speed_ratio: float = MAX_SPEED_RATIO,
        min_plausible_speed_cap: float = MIN_PLAUSIBLE_SPEED_CAP_PX,
    ) -> dict[int, dict]:
        """
        Replays filter_detections_combined()'s gate and records, per frame:
        the base distance/tolerance result, whether the motion-consistency
        screen's reference window was judged fresh (and which of the two
        freshness conditions failed, if either), the screen's own
        angle/speed reason if it was applied, and the final decision --
        detailed enough that a still image can show exactly which layer
        made the decision and why. Purely diagnostic -- returns per-frame
        records rather than filtered detections.
        """
        records: dict[int, dict] = {}
        accepted_anchors: list[tuple[int, float, float]] = []

        for i, frame in enumerate(ball_detections):
            if 1 not in frame:
                continue

            cx_cur, cy_cur = bbox_center(*frame[1].bbox)
            candidate = (i, cx_cur, cy_cur)

            if not accepted_anchors:
                accepted_anchors.append(candidate)
                records[i] = {
                    'path': 'seed: first accepted detection',
                    'base_distance': None,
                    'base_tolerance': None,
                    'base_accepted': True,
                    'anchor_frame': None,
                    'frame_gap': None,
                    'freshness_gap_ok': None,
                    'freshness_span_ok': None,
                    'reference_span': None,
                    'reference_anchors_count': None,
                    'fresh': None,
                    'motion_check_applied': False,
                    'motion_reason': None,
                    'accepted': True,
                }
                continue

            last_idx, last_cx, last_cy = accepted_anchors[-1]
            frame_gap = i - last_idx
            distance = euclidean_distance((cx_cur, cy_cur), (last_cx, last_cy))
            tolerance = fallback_travel * frame_gap
            base_accepted = distance <= tolerance

            if not base_accepted:
                records[i] = {
                    'path': f'base-reject (distance {distance:.2f} > tolerance {tolerance:.2f})',
                    'base_distance': distance,
                    'base_tolerance': tolerance,
                    'base_accepted': False,
                    'anchor_frame': last_idx,
                    'frame_gap': frame_gap,
                    'freshness_gap_ok': None,
                    'freshness_span_ok': None,
                    'reference_span': None,
                    'reference_anchors_count': None,
                    'fresh': None,
                    'motion_check_applied': False,
                    'motion_reason': None,
                    'accepted': False,
                }
                continue

            reference_window = accepted_anchors[-history_size:]
            span = reference_window[-1][0] - reference_window[0][0]
            freshness_gap_ok = frame_gap <= freshness_gap_threshold
            freshness_span_ok = len(reference_window) >= 2 and span <= reference_span_threshold
            fresh = freshness_gap_ok and freshness_span_ok

            if not fresh:
                reasons = []
                if not freshness_gap_ok:
                    reasons.append(f'frame_gap {frame_gap} > {freshness_gap_threshold}')
                if not freshness_span_ok:
                    if len(reference_window) < 2:
                        reasons.append('fewer than 2 accepted anchors')
                    else:
                        reasons.append(f'reference span {span} > {reference_span_threshold}')

                accepted_anchors.append(candidate)
                accepted_anchors = accepted_anchors[-history_size:]

                records[i] = {
                    'path': f'screen-skipped (not fresh: {"; ".join(reasons)}) -- accepted on base check alone',
                    'base_distance': distance,
                    'base_tolerance': tolerance,
                    'base_accepted': True,
                    'anchor_frame': last_idx,
                    'frame_gap': frame_gap,
                    'freshness_gap_ok': freshness_gap_ok,
                    'freshness_span_ok': freshness_span_ok,
                    'reference_span': span,
                    'reference_anchors_count': len(reference_window),
                    'fresh': False,
                    'motion_check_applied': False,
                    'motion_reason': None,
                    'accepted': True,
                }
                continue

            motion_accepted, motion_reason = self._motion_consistency_explained(
                reference_window, candidate, max_angle_deviation_deg,
                min_reference_speed, max_speed_ratio, min_plausible_speed_cap,
            )

            if motion_accepted:
                accepted_anchors.append(candidate)
                accepted_anchors = accepted_anchors[-history_size:]

            records[i] = {
                'path': f'screen-{"consistent" if motion_accepted else "inconsistent"}',
                'base_distance': distance,
                'base_tolerance': tolerance,
                'base_accepted': True,
                'anchor_frame': last_idx,
                'frame_gap': frame_gap,
                'freshness_gap_ok': freshness_gap_ok,
                'freshness_span_ok': freshness_span_ok,
                'reference_span': span,
                'reference_anchors_count': len(reference_window),
                'fresh': True,
                'motion_check_applied': True,
                'motion_reason': motion_reason,
                'accepted': motion_accepted,
            }

        return records

    def _trajectory_nodes(
        self,
        ball_detections: list[dict[int, BallDetection]],
    ) -> list[tuple[int, float, float]]:
        """Collect every frame carrying a raw detection as a (frame_index, cx, cy) node, in ascending frame order, for the global trajectory solver."""
        nodes: list[tuple[int, float, float]] = []

        for i, frame in enumerate(ball_detections):
            if 1 in frame:
                cx, cy = bbox_center(*frame[1].bbox)
                nodes.append((i, cx, cy))

        return nodes

    def _solve_global_trajectory(
        self,
        nodes: list[tuple[int, float, float]],
        max_speed: float,
        skip_penalty: float,
        max_edge_gap: int,
    ) -> tuple[list[int], list[float], int]:
        """
        Dynamic program finding the minimum-cost path through the detection
        nodes, where an edge costs squared displacement per frame of gap plus
        a fixed penalty for every detection it skips over. Returns the parent
        pointers, the best cost to reach each node, and the index of the
        node the optimal path ends on. Shared by
        filter_detections_global_trajectory() and
        trace_global_trajectory_decisions() so the two cannot diverge.
        """
        n = len(nodes)

        # best[j] starts as the cost of beginning the path at j, which skips
        # every one of the j detections before it.
        best = [skip_penalty * j for j in range(n)]
        parent = [-1] * n

        for j in range(n):
            frame_j, x_j, y_j = nodes[j]

            for i in range(j):
                frame_i, x_i, y_i = nodes[i]
                gap = frame_j - frame_i
                n_skipped = j - i - 1

                if gap > max_edge_gap:
                    # Long-silence reset: after this much detector silence the
                    # continuity assumption is void, so no motion cost is charged.
                    edge_cost = skip_penalty * n_skipped
                else:
                    distance = euclidean_distance((x_i, y_i), (x_j, y_j))
                    if distance > max_speed * gap:
                        continue

                    edge_cost = (distance * distance) / gap + skip_penalty * n_skipped

                candidate = best[i] + edge_cost

                # Strict < with i ascending makes the lowest-index predecessor
                # win ties, keeping the reconstructed path deterministic.
                if candidate < best[j]:
                    best[j] = candidate
                    parent[j] = i

        end_index = -1
        best_total: float | None = None
        for j in range(n):
            total = best[j] + skip_penalty * (n - 1 - j)
            if best_total is None or total < best_total:
                best_total = total
                end_index = j

        return parent, best, end_index

    def _trajectory_path_indices(self, parent: list[int], end_index: int) -> list[int]:
        """Walk the solver's parent pointers back from the end node, returning the chosen node indices in ascending order."""
        path: list[int] = []
        k = end_index

        while k != -1:
            path.append(k)
            k = parent[k]

        path.reverse()

        return path

    def filter_detections_global_trajectory(
        self,
        ball_detections: list[dict[int, BallDetection]],
        max_speed: float = GLOBAL_MAX_SPEED_PX,
        skip_penalty: float = GLOBAL_SKIP_PENALTY,
        max_edge_gap: int = GLOBAL_MAX_EDGE_GAP,
    ) -> list[dict[int, BallDetection]]:
        """
        Offline, non-causal alternative gate: instead of judging each
        detection against the last accepted anchor in a single forward pass,
        lays out every detection in the clip and selects the single
        least-implausible route through them by dynamic programming.
        Detections on that route are accepted; all others are rejected.

        This is the first non-causal variant in this module -- the other four
        gates are all single-forward-pass and causal. Non-causality is
        deliberate and legitimate here: the pipeline is offline batch
        processing, and every downstream consumer reads the cleaned track
        only after the whole clip has been processed.

        An edge from one detection to a later one costs squared displacement
        divided by the frame gap, plus skip_penalty for each detection it
        passes over. That cost is path-invariant under constant velocity --
        including collinear in-flight points costs the same in the motion
        term as skipping them -- so the skip penalty then strictly favours
        including a real flight, while an off-path cluster (a static false
        positive beside the ball's route) must pay two large detour hops to
        enter and leave, which the skip penalty undercuts. Edges implying a
        straight-line speed above max_speed are forbidden outright; edges
        spanning more than max_edge_gap frames are treated as long-silence
        resets and charged no motion cost at all.

        Unlike the three guard gates in this module, no part of this
        decision rests on trusted recent accepted history, so it has no
        stale-reference failure mode: the evidence for any single detection
        is the whole clip. Does not call or share state with
        filter_detections(), filter_detections_predicted(),
        filter_detections_staleness_aware(),
        filter_detections_motion_consistency(), or
        filter_detections_combined().
        """
        result = [dict(frame) for frame in ball_detections]
        nodes = self._trajectory_nodes(ball_detections)

        # A clip with no detections, or exactly one, has no route to choose
        # between and passes through unchanged.
        if len(nodes) <= 1:
            return result

        parent, _, end_index = self._solve_global_trajectory(
            nodes, max_speed, skip_penalty, max_edge_gap,
        )
        on_path = set(self._trajectory_path_indices(parent, end_index))

        for node_index, (frame_index, _, _) in enumerate(nodes):
            if node_index not in on_path:
                result[frame_index] = {}

        return result

    def trace_global_trajectory_decisions(
        self,
        ball_detections: list[dict[int, BallDetection]],
        max_speed: float = GLOBAL_MAX_SPEED_PX,
        skip_penalty: float = GLOBAL_SKIP_PENALTY,
        max_edge_gap: int = GLOBAL_MAX_EDGE_GAP,
    ) -> dict[int, dict]:
        """
        Replays filter_detections_global_trajectory()'s selection and records,
        per frame carrying a raw detection: whether it landed on the chosen
        path, which earlier accepted frame preceded it there, and that edge's
        gap, displacement, implied speed and motion cost. Runs the same
        solver the gate itself uses rather than a reimplementation, so its
        verdicts cannot drift from the gate's. Purely diagnostic -- returns
        per-frame records rather than filtered detections.
        """
        nodes = self._trajectory_nodes(ball_detections)
        records: dict[int, dict] = {}

        if not nodes:
            return records

        if len(nodes) == 1:
            frame_index = nodes[0][0]
            records[frame_index] = {
                'accepted': True,
                'node_index': 0,
                'reason': 'only detection in the clip -- passed through unchanged',
                'edge_kind': 'path start',
                'previous_accepted_frame': None,
                'edge_gap': None,
                'edge_distance': None,
                'edge_speed': None,
                'edge_motion_cost': None,
            }
            return records

        parent, _, end_index = self._solve_global_trajectory(
            nodes, max_speed, skip_penalty, max_edge_gap,
        )
        path = self._trajectory_path_indices(parent, end_index)
        on_path = set(path)
        predecessor = {later: earlier for earlier, later in zip(path, path[1:])}

        for node_index, (frame_index, cx, cy) in enumerate(nodes):
            if node_index not in on_path:
                records[frame_index] = {
                    'accepted': False,
                    'node_index': node_index,
                    'reason': 'skipped -- cheaper to leave off the chosen path than to detour through it',
                    'edge_kind': None,
                    'previous_accepted_frame': None,
                    'edge_gap': None,
                    'edge_distance': None,
                    'edge_speed': None,
                    'edge_motion_cost': None,
                }
                continue

            previous_index = predecessor.get(node_index)

            if previous_index is None:
                records[frame_index] = {
                    'accepted': True,
                    'node_index': node_index,
                    'reason': 'first detection on the chosen path',
                    'edge_kind': 'path start',
                    'previous_accepted_frame': None,
                    'edge_gap': None,
                    'edge_distance': None,
                    'edge_speed': None,
                    'edge_motion_cost': None,
                }
                continue

            frame_prev, x_prev, y_prev = nodes[previous_index]
            gap = frame_index - frame_prev
            distance = euclidean_distance((x_prev, y_prev), (cx, cy))

            if gap > max_edge_gap:
                edge_kind = 'long-silence reset (no motion cost charged)'
                motion_cost = 0.0
            else:
                edge_kind = 'motion edge'
                motion_cost = (distance * distance) / gap

            records[frame_index] = {
                'accepted': True,
                'node_index': node_index,
                'reason': 'on the chosen path',
                'edge_kind': edge_kind,
                'previous_accepted_frame': frame_prev,
                'edge_gap': gap,
                'edge_distance': distance,
                'edge_speed': distance / gap,
                'edge_motion_cost': motion_cost,
            }

        return records

    def fill_missing(
        self,
        ball_detections: list[dict[int, BallDetection]],
    ) -> list[dict[int, BallDetection]]:
        """Fill frames with no ball detection via Pandas linear interpolation over bbox coordinates."""
        if not any(1 in frame for frame in ball_detections):
            # Without a single surviving detection there is nothing to interpolate
            # between, so every frame stays empty -- indistinguishable downstream
            # from a ball that was simply never in shot.
            print(
                f'[BallInterpolator] No ball detections survive across {len(ball_detections)} '
                f'frames — nothing to interpolate, every frame will be empty.'
            )

        rows = []
        for frame in ball_detections:
            if 1 in frame:
                rows.append(frame[1].bbox)
            else:
                rows.append([np.nan, np.nan, np.nan, np.nan])

        df = pd.DataFrame(rows, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        df = df.ffill()

        result: list[dict[int, BallDetection]] = []
        for i, row in enumerate(df.to_numpy().tolist()):
            if any(np.isnan(v) for v in row):
                result.append({})
            elif 1 in ball_detections[i]:
                result.append(ball_detections[i])
            else:
                result.append({
                    1: BallDetection(
                        bbox=row,
                        confidence=0.0,
                    )
                })

        return result
