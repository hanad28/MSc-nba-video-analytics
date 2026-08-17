"""
court_mapper.py maps tracked players onto the metric court plane for a whole
clip, fitting one homography per frame from that frame's confident keypoints
and reporting how many frames could not produce one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from basketball.detection.player_detector import PlayerTrack
from basketball.homography.transform import (
    DegenerateCorrespondencesError,
    Homography,
    HomographyError,
    InsufficientCorrespondencesError,
    MalformedCorrespondencesError,
)
from basketball.keypoints.court_keypoints import Keypoint
from basketball.keypoints.court_template import COURT_LENGTH_M, COURT_WIDTH_M, TEMPLATE_POINTS_M
from basketball.utils.geometry import foot_point

# Players legitimately step past the sideline and baseline, and the measured
# ~5 cm localisation error moves a mapped position by a little more, so the
# accepted region is the court plus a margin rather than the court exactly.
COURT_MARGIN_M = 2.0

DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class MappingReport:
    """Per-clip counts of how many frames produced a homography and why the rest did not."""

    n_frames: int = 0
    mapped_frames: int = 0
    insufficient_keypoints: int = 0
    degenerate_keypoints: int = 0
    malformed_input: int = 0
    positions_mapped: int = 0
    positions_dropped_out_of_bounds: int = 0
    positions_dropped_at_horizon: int = 0
    # Only the insufficient-keypoint cause records frame indices; the
    # degenerate and malformed branches increment counters alone. The asymmetry
    # is deliberate, not an oversight. Stage 7's measurement over all 534 frames
    # recorded zero numerically degenerate fits across every clip, so a
    # diagnostic list for a cause never observed would be speculative
    # machinery. The counters exist, so if degenerate frames do appear the
    # availability report will show them and the list can be extended then,
    # against evidence. This field is named for exactly what it holds; if that
    # becomes misleading once another cause appears, rename it at that point.
    frames_below_threshold: list[int] = field(default_factory=list)
    # Per frame, whether a homography was fitted at all, recorded during the
    # single mapping pass rather than recomputed, so there is one source of
    # truth. An empty position dict has three distinct causes: no homography,
    # a homography with no tracked players, or a homography whose every mapped
    # position was dropped. Only the first is a failure, and MinimapAnnotator
    # needs this to caption them apart: the minimap is how the transform is
    # visually verified, so captioning a correctly-mapped frame as a failure
    # would inflate the apparent failure rate above the measured one and send
    # a reader chasing a problem that is not there.
    frame_has_homography: list[bool] = field(default_factory=list)

    @property
    def unmapped_frames(self) -> int:
        """The number of frames that produced no homography, for any reason."""
        return self.insufficient_keypoints + self.degenerate_keypoints + self.malformed_input

    def reconciles(self) -> bool:
        """Whether the mapped and unmapped frame counts account for every frame."""
        return self.mapped_frames + self.unmapped_frames == self.n_frames

    def summary(self) -> str:
        """One-line summary of homography availability and dropped positions across the clip."""
        rate = self.mapped_frames / self.n_frames if self.n_frames else float('nan')
        return (
            f'[court] {self.mapped_frames}/{self.n_frames} frames mapped ({rate:.1%}); '
            f'unmapped: {self.insufficient_keypoints} too few keypoints, '
            f'{self.degenerate_keypoints} degenerate, {self.malformed_input} malformed. '
            f'Positions: {self.positions_mapped} mapped, '
            f'{self.positions_dropped_out_of_bounds} dropped out of bounds, '
            f'{self.positions_dropped_at_horizon} dropped at the horizon.'
        )


class CourtMapper:
    """Maps tracked player positions from image space onto the metric court plane, one independent homography per frame."""

    # No caching. The stage runs no inference: it is a 4x4 solve and a handful
    # of matrix multiplies per frame, microseconds against the minutes every
    # cached stage costs. A cache would need a fingerprint and a revision
    # constant to stay honest, which is more machinery and more staleness risk
    # than the work it would save.

    def __init__(
        self,
        keypoint_confidence_threshold: float = DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
        court_margin_m: float = COURT_MARGIN_M,
    ) -> None:
        # The PER-KEYPOINT threshold, selecting which landmarks are trusted
        # as correspondences, the same quantity as
        # measure_court_keypoints.KEYPOINT_CONFIDENCE_THRESHOLD. It is NOT
        # config's keypoints.confidence_threshold, which main.py already
        # reads and passes to CourtKeypoints as its INSTANCE threshold,
        # gating whether a court is detected at all. The two are deliberately
        # different quantities and both
        # default to 0.5, so a conflation would be invisible, hence the
        # longer name rather than the shorter one the config key suggests.
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        self.court_margin_m = court_margin_m

    def _correspondences(
        self,
        frame_keypoints: list[Keypoint],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Pair each confident keypoint's image position with its template position, by index."""
        # The index IS the correspondence: keypoint k pairs with
        # TEMPLATE_POINTS_M[k]. An index with no template entry is skipped
        # rather than raising, so a checkpoint predicting more than 18 points
        # degrades instead of killing the clip.
        image_points: list[tuple[float, float]] = []
        template_points: list[tuple[float, float]] = []
        for keypoint in frame_keypoints:
            if keypoint.confidence < self.keypoint_confidence_threshold:
                continue
            template = TEMPLATE_POINTS_M.get(keypoint.index)
            if template is None:
                continue
            image_points.append((float(keypoint.x), float(keypoint.y)))
            template_points.append(template)
        return image_points, template_points

    def _in_bounds(self, position: tuple[float, float]) -> bool:
        """Whether a mapped court position falls within the court plus the accepted margin."""
        x, y = position
        return (
            -self.court_margin_m <= x <= COURT_LENGTH_M + self.court_margin_m
            and -self.court_margin_m <= y <= COURT_WIDTH_M + self.court_margin_m
        )

    def map_to_court(
        self,
        player_tracks: list[dict[int, PlayerTrack]],
        keypoints_per_frame: list[list[Keypoint]],
    ) -> tuple[list[dict[int, tuple[float, float]]], MappingReport]:
        """Map every frame's tracked players onto the court plane, returning one dict per frame and a report of homography availability."""
        if len(keypoints_per_frame) != len(player_tracks):
            raise ValueError(
                f'Got {len(keypoints_per_frame)} keypoint frames for {len(player_tracks)} '
                f'track frames — the two must be aligned frame-for-frame.'
            )

        report = MappingReport(n_frames=len(player_tracks))
        positions: list[dict[int, tuple[float, float]]] = []

        for frame_idx, (tracks, frame_keypoints) in enumerate(zip(player_tracks, keypoints_per_frame)):
            image_points, template_points = self._correspondences(frame_keypoints)

            # A fresh Homography per frame, never a carried one. Carrying the
            # last valid matrix looks like the helpful choice and is the worst
            # available option here: the keypoints vanish precisely BECAUSE the
            # camera is panning, so a stale matrix maps the same image position
            # to a steadily wronger court position, and every player jumps when
            # a real homography returns. Stage 9 computes speed from
            # frame-to-frame displacement, so that jump becomes a large
            # fabricated velocity indistinguishable from a real sprint. A gap
            # produces no speed at all, which is honest and recoverable; stale
            # geometry produces a wrong speed that looks exactly like a
            # measurement. On clip_3 the shortfall is a single contiguous
            # 45-frame window, so this is 1.5 seconds of fabrication, not one
            # stray frame.
            homography = Homography()
            try:
                homography.transform_points(image_points, template_points)
            except InsufficientCorrespondencesError:
                report.insufficient_keypoints += 1
                report.frames_below_threshold.append(frame_idx)
                report.frame_has_homography.append(False)
                positions.append({})
                continue
            except DegenerateCorrespondencesError:
                report.degenerate_keypoints += 1
                report.frame_has_homography.append(False)
                positions.append({})
                continue
            except MalformedCorrespondencesError:
                report.malformed_input += 1
                report.frame_has_homography.append(False)
                positions.append({})
                continue

            frame_positions: dict[int, tuple[float, float]] = {}
            for track_id, track in tracks.items():
                x1, y1, x2, y2 = track.bbox
                # The foot point, not the box centre: a homography maps the
                # ground plane, and a player's feet are the only part of them
                # on it. A centre or head position sits a metre or more above
                # the plane and maps to a systematically wrong court location,
                # further from the camera the taller the player appears.
                try:
                    court_position = homography.apply_homography(foot_point(x1, y1, x2, y2))
                except HomographyError:
                    report.positions_dropped_at_horizon += 1
                    continue

                if not self._in_bounds(court_position):
                    # Counted rather than silently discarded: a high drop rate
                    # is evidence the homography is wrong, not that players
                    # left the court.
                    report.positions_dropped_out_of_bounds += 1
                    continue

                frame_positions[track_id] = court_position
                report.positions_mapped += 1

            report.mapped_frames += 1
            report.frame_has_homography.append(True)
            positions.append(frame_positions)

        return positions, report
