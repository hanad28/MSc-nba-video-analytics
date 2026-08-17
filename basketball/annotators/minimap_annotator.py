"""
minimap_annotator.py composites a top-down court view into a corner of each
frame, drawing CourtMapper's metric positions onto the court template so the
tactical view can be read against the broadcast view it came from.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from basketball.annotators.player_annotator import PlayerAnnotator
from basketball.keypoints.court_template import to_pixels

DEFAULT_TEMPLATE_PATH = 'assets/court_template.png'


class MinimapAnnotator:
    """Draws a semi-transparent top-down court overlay with players placed by their mapped court positions and coloured by team."""

    # Bottom-left. PossessionAnnotator owns the top-left and can occupy two
    # lines; EventAnnotator owns the top-right and can stack three. The bottom
    # edge is unused by both, and left over right keeps the minimap clear of
    # the event captions, which are the longest text drawn.
    MARGIN = 12
    OVERLAY_ALPHA = 0.75

    # Read from PlayerAnnotator rather than redeclared, so a player is the same
    # colour in the broadcast view and the tactical view by construction. A
    # duplicated literal would drift the moment either is retuned.
    TEAM_COLOURS = {
        1: PlayerAnnotator.DEFAULT_TEAM_1_COLOUR,
        2: PlayerAnnotator.DEFAULT_TEAM_2_COLOUR,
        0: PlayerAnnotator.UNKNOWN_TEAM_COLOUR,
    }

    PLAYER_RADIUS = 5
    PLAYER_OUTLINE = (0, 0, 0)

    # A dot whose position falls OFF the template is pinned to the edge rather
    # than drawn where the player is, so it is outlined distinctly. The minimap
    # is this stage's verification mechanism, and 'this position is not where
    # the mark sits' is exactly what a reader must not mistake for a
    # measurement. White against the black in-court outline, and thicker, so
    # the two are distinguishable at a 5 px radius.
    OFF_TEMPLATE_OUTLINE = (255, 255, 255)
    OFF_TEMPLATE_OUTLINE_THICKNESS = 2

    # Captioned ONLY when no homography could be fitted. An empty position
    # dict has three causes (no homography, a homography with no tracked
    # players, or a homography whose every position was dropped) and only the
    # first is a failure. The other two draw a bare court and are left
    # uncaptioned: an empty court under a working transform is a normal
    # condition, not a fault, and labelling it would make the on-screen
    # failure rate look worse than the measured one. The minimap is how the
    # transform is verified by eye, so that would send a reader chasing a
    # problem that is not there.
    NO_HOMOGRAPHY_CAPTION = 'NO HOMOGRAPHY'
    CAPTION_COLOUR = (255, 255, 255)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    CAPTION_FONT_SCALE = 0.35
    CAPTION_THICKNESS = 1

    # The smallest scale this annotator supports. 0.25 renders a 149x80 court,
    # the point below which player positions stop being readable, which is the
    # minimap's entire purpose, so claiming support for a smaller size would be
    # claiming something the overlay cannot do.
    #
    # The floor also clears the no-homography caption on every supported
    # build: cv2.getTextSize is build-dependent (the caption measures 96 px
    # under OpenCV 4.13.0 against 76 px under 5.0.0), and a caption tuned to
    # a few pixels of headroom would simply break on the next build.
    MIN_SUPPORTED_SCALE = 0.25

    def __init__(
        self,
        template_path: str = DEFAULT_TEMPLATE_PATH,
        scale: float = 0.35,
        team_1_colour: tuple[int, int, int] | None = None,
        team_2_colour: tuple[int, int, int] | None = None,
    ) -> None:
        # Raised rather than rendered: a smaller scale produces an overlay too
        # small to read player positions from, which is the only thing the
        # minimap is for, and silently drawing an illegible one would look like
        # a working tactical view.
        if scale < self.MIN_SUPPORTED_SCALE:
            raise ValueError(
                f'scale must be at least {self.MIN_SUPPORTED_SCALE}, got {scale} — '
                f'a smaller minimap cannot render player positions legibly.'
            )

        self.template_path = template_path
        self.scale = scale
        # Per-clip overrides flow through the same way PlayerAnnotator takes
        # them, so clip_3's red team-2 override colours both views alike.
        self.team_colours = dict(self.TEAM_COLOURS)
        if team_1_colour is not None:
            self.team_colours[1] = tuple(team_1_colour)
        if team_2_colour is not None:
            self.team_colours[2] = tuple(team_2_colour)
        self._template: np.ndarray | None = None

    def load_template(self) -> np.ndarray:
        """Load and cache the court template image, raising a clear error naming the path if it is absent or unreadable."""
        if self._template is not None:
            return self._template

        if not os.path.exists(self.template_path):
            raise FileNotFoundError(
                f'Court template image does not exist: {self.template_path}. '
                f'It is required to render the Stage 8 tactical view.'
            )
        template = cv2.imread(self.template_path)
        if template is None:
            raise OSError(f'Court template image could not be decoded: {self.template_path}.')

        height, width = template.shape[:2]
        self._template = cv2.resize(
            template, (max(1, int(width * self.scale)), max(1, int(height * self.scale))),
            interpolation=cv2.INTER_AREA,
        )
        return self._template

    def occupied_region_for(self, frames: list[np.ndarray]) -> tuple[int, int, int, int] | None:
        """Return the rectangle this overlay will occupy on the given frames, or None when there are no frames to measure."""
        # Takes the frame list rather than a height and width so a caller does
        # not have to reach into frames[0].shape itself, which is both a second
        # place the frame geometry is read and an AttributeError waiting for
        # any caller whose frames are not ndarrays.
        if not frames:
            return None

        height, width = frames[0].shape[:2]
        return self.occupied_region(height, width)

    def occupied_region(self, frame_height: int, frame_width: int) -> tuple[int, int, int, int]:
        """Return the bottom-left rectangle this overlay will composite into, as (x1, y1, x2, y2) in frame pixels."""
        # Published rather than left implicit because this annotator draws LAST,
        # over everything else: any annotator that ran earlier and put marks
        # here has already been overwritten by the time the frame is written.
        # MetricsAnnotator attaches its text to each player's box, which is
        # collision-free against the two top-corner captions but not against a
        # panel composited afterwards: a player near the bottom-left sideline
        # had their speed drawn and then covered. Deriving the rectangle here,
        # from the same MARGIN and template the compositing step uses, is what
        # stops a consumer's copy of the geometry drifting from the real one.
        template = self.load_template()
        court_h, court_w = template.shape[:2]
        court_h = min(court_h, max(0, frame_height - 2 * self.MARGIN))
        court_w = min(court_w, max(0, frame_width - 2 * self.MARGIN))
        if court_h <= 0 or court_w <= 0:
            # Nothing is composited at this frame size, so nothing is occupied.
            return (0, 0, 0, 0)

        y1 = frame_height - self.MARGIN - court_h
        return (self.MARGIN, y1, self.MARGIN + court_w, y1 + court_h)

    def _render_court(
        self,
        court_positions: dict[int, tuple[float, float]],
        team_assignment: dict[int, int],
        has_homography: bool,
    ) -> np.ndarray:
        """Return a copy of the scaled template with each mapped player drawn as a team-coloured dot, captioned only when no homography was fitted."""
        court = self.load_template().copy()
        height, width = court.shape[:2]

        for track_id, position in court_positions.items():
            # to_pixels() is drawing only and is never inverted: the template
            # image's aspect is 1.863 against the metric table's 1.880, so a
            # mark can sit ~1% off its painted lines. Measurement stays in
            # metres throughout CourtMapper and Stage 9.
            centre, off_template = self._render_position(position, width, height)
            colour = self.team_colours.get(team_assignment.get(track_id, 0), self.team_colours[0])
            cv2.circle(court, centre, self.PLAYER_RADIUS, colour, thickness=-1)
            # A thin outline so an off-white team-1 dot stays visible against
            # the template's own pale floor. A dot whose position is off the
            # template gets a distinct outline instead: it is pinned to the edge
            # rather than drawn where the player actually is, and a viewer
            # reading the minimap as verification must tell the two apart.
            outline = self.OFF_TEMPLATE_OUTLINE if off_template else self.PLAYER_OUTLINE
            thickness = self.OFF_TEMPLATE_OUTLINE_THICKNESS if off_template else 1
            cv2.circle(court, centre, self.PLAYER_RADIUS, outline, thickness=thickness)

        if not has_homography:
            # The court is always drawn: an absent overlay and an empty one
            # are visually identical, so a viewer could not otherwise tell
            # 'nothing to draw' from 'the minimap stopped working'. Only the
            # genuine failure is captioned.
            self._draw_no_homography_caption(court)
        return court

    def _render_position(
        self,
        position: tuple[float, float],
        width: int,
        height: int,
    ) -> tuple[tuple[int, int], bool]:
        """Return one court position's pixel centre placed fully inside the render, and whether the position itself falls off the template."""
        # Clamped HERE rather than in to_pixels(), which is a shared coordinate
        # conversion: clamping there would silently distort every other caller.
        # The constraint belongs to this render, not to the conversion.
        #
        # CourtMapper accepts positions within COURT_MARGIN_M of the court and
        # counts them in positions_mapped, but to_pixels() scales linearly with
        # no clamp, so those land outside the template and cv2.circle drops the
        # dot entirely. The drawn count would then disagree with the reported
        # count exactly when a player steps over a line. Note this bites on the
        # line too, not only past it: y = COURT_WIDTH_M maps to y = height, one
        # row past the last.
        #
        # PLAYER_RADIUS is included so the dot is fully visible rather than
        # half-clipped at the edge, matching how KeypointAnnotator clamps its
        # index labels.
        x, y = to_pixels(position, width, height)

        # Two separate questions, deliberately not one. Whether the position is
        # OFF THE TEMPLATE is decided against the render bounds, and that is
        # what the distinct outline reports. Whether the dot needed NUDGING to
        # be drawn whole is a placement detail of the radius inset, and says
        # nothing about where the player is: at a 5 px radius the inset spans
        # 0.68 m of court, so folding it into the flag marked a band that wide
        # inside every line as displaced when those positions are genuinely in
        # court, the opposite of what the outline exists to say.
        is_off_template = not (0 <= x <= width - 1 and 0 <= y <= height - 1)

        low_x, high_x = self.PLAYER_RADIUS, max(self.PLAYER_RADIUS, width - 1 - self.PLAYER_RADIUS)
        low_y, high_y = self.PLAYER_RADIUS, max(self.PLAYER_RADIUS, height - 1 - self.PLAYER_RADIUS)
        placed_x = min(max(x, low_x), high_x)
        placed_y = min(max(y, low_y), high_y)

        return (int(round(placed_x)), int(round(placed_y))), is_off_template

    def _draw_no_homography_caption(self, court: np.ndarray) -> None:
        """Draw the no-homography caption centred on the court render, skipping it when the render is too narrow to hold it."""
        height, width = court.shape[:2]
        (text_w, text_h), _ = cv2.getTextSize(
            self.NO_HOMOGRAPHY_CAPTION, self.FONT, self.CAPTION_FONT_SCALE, self.CAPTION_THICKNESS,
        )

        # Measured against the render rather than assumed to fit: clamping a
        # too-wide caption to x=0 truncates its tail, which reads as a
        # different message rather than as an overflow, and every render-vs-
        # render test would still pass.
        if text_w > width:
            return

        x = max(0, (width - text_w) // 2)
        y = max(text_h, height // 2)
        cv2.putText(
            court, self.NO_HOMOGRAPHY_CAPTION, (x, y), self.FONT, self.CAPTION_FONT_SCALE,
            self.CAPTION_COLOUR, self.CAPTION_THICKNESS,
        )

    def _composite(self, frame: np.ndarray, court: np.ndarray) -> None:
        """Blend the rendered court into the frame's bottom-left corner, in place on the already-copied frame."""
        frame_h, frame_w = frame.shape[:2]

        # The destination rectangle comes from occupied_region() rather than
        # being recomputed here, so the region this blends into and the region
        # MetricsAnnotator avoids are the same rectangle by construction. A
        # frame smaller than the overlay is not reachable at broadcast
        # resolution, but cropping rather than raising keeps a small test or
        # thumbnail frame renderable instead of killing the run.
        x1, y1, x2, y2 = self.occupied_region(frame_h, frame_w)
        court_h, court_w = y2 - y1, x2 - x1
        if court_h <= 0 or court_w <= 0:
            return

        region = frame[y1:y1 + court_h, x1:x1 + court_w]
        cv2.addWeighted(
            court[:court_h, :court_w], self.OVERLAY_ALPHA, region, 1.0 - self.OVERLAY_ALPHA, 0.0,
            dst=region,
        )

    def draw(
        self,
        frames: list[np.ndarray],
        court_positions: list[dict[int, tuple[float, float]]],
        frame_has_homography: list[bool],
        team_assignment: list[dict[int, int]] | None = None,
    ) -> list[np.ndarray]:
        """Return annotated copies of all frames with the tactical court view composited into the bottom-left corner."""
        # frame_has_homography is REQUIRED, and deliberately sits before the
        # optional team_assignment despite arriving later in the pipeline. An
        # empty position dict has three causes and only one is a failure, so
        # deriving the caption from bool(court_positions[index]) would caption
        # a frame that mapped correctly as NO HOMOGRAPHY; a default here would
        # let the wiring do exactly that silently, on the very output this
        # stage is verified by. Callers
        # pass MappingReport.frame_has_homography.
        if len(court_positions) != len(frames):
            raise ValueError(
                f'Got {len(court_positions)} court-position frames for {len(frames)} video '
                f'frames — the two must be aligned frame-for-frame.'
            )
        if team_assignment is not None and len(team_assignment) != len(frames):
            raise ValueError(
                f'Got {len(team_assignment)} team-assignment frames for {len(frames)} video '
                f'frames — the two must be aligned frame-for-frame.'
            )
        if len(frame_has_homography) != len(frames):
            raise ValueError(
                f'Got {len(frame_has_homography)} homography flags for {len(frames)} video '
                f'frames — the two must be aligned frame-for-frame.'
            )

        output: list[np.ndarray] = []
        for index, frame in enumerate(frames):
            frame = frame.copy()
            teams = team_assignment[index] if team_assignment is not None else {}
            self._composite(
                frame,
                self._render_court(court_positions[index], teams, frame_has_homography[index]),
            )
            output.append(frame)
        return output
