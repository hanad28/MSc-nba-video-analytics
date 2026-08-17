from __future__ import annotations

from basketball.cache.cache_utils import data_digest, load_valid_cache, save_cache_with_meta
from basketball.detection.ball_detector import BallDetection
from basketball.detection.player_detector import PlayerTrack
from basketball.utils.box_utils import bbox_center
from basketball.utils.geometry import euclidean_distance


class PossessionTracker:
    """Tracks which player is in possession of the ball across frames using bounding box containment and proximity, with temporal confirmation."""

    # Increment whenever find_holder(), the streak-confirmation machine, the
    # real-anchor rule or the gap-tolerance behaviour changes: those are code
    # rather than parameters, so the cache fingerprint cannot see them by
    # itself and would otherwise serve a stale cache. 1 -> 2: overlap_ratio()
    # gained a zero-area guard (was ZeroDivisionError on a zero-width or
    # zero-height ball box, now 0.0 with a proximity fallback), changing
    # find_holder()'s behaviour on exactly that input.
    LOGIC_REVISION = 2

    def __init__(self, config: dict) -> None:
        if 'possession' not in config:
            raise KeyError('Config has no \'possession\' section — PossessionTracker cannot be configured.')

        possession_config = config['possession']
        missing = [
            key for key in ('max_ball_distance', 'hold_threshold', 'bbox_overlap_min')
            if key not in possession_config
        ]
        if missing:
            raise KeyError(f'Config section \'possession\' is missing required keys: {missing}')

        self.max_ball_distance: float = possession_config['max_ball_distance']
        self.hold_threshold: int = possession_config['hold_threshold']
        self.bbox_overlap_min: float = possession_config['bbox_overlap_min']

        if self.hold_threshold < 1:
            raise ValueError(f'hold_threshold must be at least 1, got {self.hold_threshold}.')

    def get_proximity_points(self, player_bbox: list[float], ball_center: tuple[float, float]) -> list[tuple[float, float]]:
        """Returns candidate points on a player's bounding box for measuring distance to the ball."""
        ball_center_x, ball_center_y = ball_center
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1

        points: list[tuple[float, float]] = []

        if y1 < ball_center_y < y2:
            points.append((x1, ball_center_y))
            points.append((x2, ball_center_y))

        if x1 < ball_center_x < x2:
            points.append((ball_center_x, y1))
            points.append((ball_center_x, y2))

        points += [
            (x1 + width / 2, y1),
            (x2, y1),
            (x1, y1),
            (x2, y1 + height / 2),
            (x1, y1 + height / 2),
            (x1 + width / 2, y1 + height / 2),
            (x2, y2),
            (x1, y2),
            (x1 + width / 2, y2),
            (x1 + width / 2, y1 + height / 3),
        ]
        return points

    def closest_distance(self, ball_center: tuple[float, float], player_bbox: list[float]) -> float:
        """Returns the minimum distance from the ball centre to any proximity point on a player's bounding box."""
        points = self.get_proximity_points(player_bbox, ball_center)
        return min(euclidean_distance(ball_center, point) for point in points)

    def overlap_ratio(self, player_bbox: list[float], ball_bbox: list[float]) -> float:
        """Returns the fraction of the ball's bounding box area contained within the player's bounding box."""
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        ix1 = max(px1, bx1)
        iy1 = max(py1, by1)
        ix2 = min(px2, bx2)
        iy2 = min(py2, by2)

        if ix2 < ix1 or iy2 < iy1:
            return 0.0

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        ball_area = (bx2 - bx1) * (by2 - by1)
        if ball_area <= 0:
            # A zero-width or zero-height ball box has no area to be
            # contained, so containment is 0.0 rather than a ZeroDivisionError
            # that would take down a whole clip's run. find_holder() then
            # falls through to its proximity test, which is the correct
            # treatment for a box carrying no area evidence. In practice only
            # area == 0 reaches here: an inverted box makes the intersection
            # empty, so the check above returns first.
            return 0.0
        return intersection_area / ball_area

    def find_holder(self, ball_center: tuple[float, float], player_tracks_frame: dict[int, PlayerTrack], ball_bbox: list[float]) -> int:
        """Determines which player in a frame is most likely holding the ball, prioritising containment over proximity."""
        high_containment_candidates: list[tuple[int, float]] = []
        regular_distance_candidates: list[tuple[int, float]] = []

        for player_id, player_track in player_tracks_frame.items():
            player_bbox = player_track.bbox
            if not player_bbox:
                continue

            containment = self.overlap_ratio(player_bbox, ball_bbox)
            distance = self.closest_distance(ball_center, player_bbox)

            if containment > self.bbox_overlap_min:
                high_containment_candidates.append((player_id, containment))
            else:
                regular_distance_candidates.append((player_id, distance))

        if high_containment_candidates:
            best_candidate = max(high_containment_candidates, key=lambda candidate: candidate[1])
            return best_candidate[0]

        if regular_distance_candidates:
            best_candidate = min(regular_distance_candidates, key=lambda candidate: candidate[1])
            if best_candidate[1] < self.max_ball_distance:
                return best_candidate[0]

        return -1

    def assign_possession(
        self,
        player_tracks: list[dict[int, PlayerTrack]],
        ball_detections: list[dict[int, BallDetection]],
        cache_path: str | None = None,
    ) -> list[int]:
        """Assigns ball possession to a player for each frame, requiring a confirmed candidate streak backed by at least one genuine ball detection."""
        if len(player_tracks) != len(ball_detections):
            raise ValueError(
                f'Got {len(player_tracks)} track frames for {len(ball_detections)} ball detection '
                f'frames — the two must be aligned frame-for-frame.'
            )

        fingerprint = None
        if cache_path is not None:
            fingerprint = {
                'player_tracks_digest': data_digest(player_tracks),
                'ball_detections_digest': data_digest(ball_detections),
                'hold_threshold': self.hold_threshold,
                'bbox_overlap_min': self.bbox_overlap_min,
                'max_ball_distance': self.max_ball_distance,
                'n_frames': len(ball_detections),
                'logic_revision': self.LOGIC_REVISION,
            }
            cached = load_valid_cache(cache_path, fingerprint)
            if cached is not None:
                return cached

        num_frames = len(ball_detections)
        possession_list: list[int] = [-1] * num_frames

        streak_player_id: int | None = None
        streak_start_frame: int | None = None
        streak_length = 0
        streak_has_real_anchor = False
        streak_confirmed = False

        for frame_num in range(num_frames):
            ball_info = ball_detections[frame_num].get(1)

            if ball_info is None or not ball_info.bbox:
                continue

            ball_center = bbox_center(*ball_info.bbox)
            best_player_id = self.find_holder(ball_center, player_tracks[frame_num], ball_info.bbox)

            if best_player_id == -1:
                streak_player_id = None
                streak_length = 0
                streak_has_real_anchor = False
                streak_confirmed = False
                continue

            if best_player_id == streak_player_id:
                streak_length += 1
            else:
                streak_player_id = best_player_id
                streak_start_frame = frame_num
                streak_length = 1
                streak_has_real_anchor = False
                streak_confirmed = False

            if ball_info.confidence > 0.0:
                streak_has_real_anchor = True

            if not streak_confirmed and streak_length >= self.hold_threshold and streak_has_real_anchor:
                streak_confirmed = True
                for backfill_frame in range(streak_start_frame, frame_num + 1):
                    possession_list[backfill_frame] = streak_player_id
            elif streak_confirmed:
                possession_list[frame_num] = streak_player_id

        if cache_path is not None:
            save_cache_with_meta(possession_list, cache_path, fingerprint)

        return possession_list
