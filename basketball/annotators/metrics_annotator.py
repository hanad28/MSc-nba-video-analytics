"""
metrics_annotator.py draws each tracked player's current speed and cumulative
distance beneath their bounding box, omitting the speed line entirely on frames
where no speed could be measured rather than rendering a misleading zero.
"""
from __future__ import annotations

import cv2
import numpy as np

from basketball.detection.player_detector import PlayerTrack
from basketball.metrics.player_metrics import TrackSpeed
from basketball.utils.geometry import foot_point


class MetricsAnnotator:
    """Annotates video frames with per-player speed in metres per second and cumulative distance in metres."""

    # Drawn beneath each player's own box rather than in a corner: the
    # possession caption owns the top-left and the event captions the
    # top-right, so attaching the text to the player keeps it clear of both,
    # and makes it unambiguous which figure belongs to whom.
    #
    # That reasoning does NOT extend to the minimap, which is composited
    # afterwards over the bottom-left and simply overwrites whatever is
    # already there. Avoiding a region is only possible if this annotator
    # knows where it is, so draw() takes the rectangle; see reserved_region.
    TEXT_COLOUR: tuple[int, int, int] = (255, 255, 255)
    OUTLINE_COLOUR: tuple[int, int, int] = (0, 0, 0)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.4
    THICKNESS = 1
    OUTLINE_THICKNESS = 3

    # Clear space between stacked lines, ON TOP of the measured glyph extent.
    # Line spacing is derived from cv2.getTextSize rather than hardcoded: a
    # fixed LINE_HEIGHT of 12 leaves ZERO clear space under OpenCV 5.0.0 and
    # minus one under 4.13.0, where the rendered bands abut and merge. Two
    # lines with no gap between them render poorly whatever a test asserts,
    # and a constant sitting at or below the measured extent is
    # build-dependent by construction.
    LINE_GAP = 4

    # Clear of annotate_player()'s foot ellipse and its track-id label, which
    # sit at the bottom-centre of the box.
    VERTICAL_OFFSET = 26

    # One decimal on both. Stage 7 measured ~5 cm of localisation error, so a
    # speed rendered as 5.83 m/s claims a precision the measurement does not
    # have; 5.8 is honest at this error budget.
    SPEED_DECIMALS = 1
    DISTANCE_DECIMALS = 1

    def draw(
        self,
        frames: list[np.ndarray],
        player_tracks: list[dict[int, PlayerTrack]],
        distances: list[dict[int, float]],
        speeds: list[dict[int, TrackSpeed]],
        reserved_region: tuple[int, int, int, int] | None = None,
    ) -> list[np.ndarray]:
        """Return annotated copies of all frames with each tracked player's speed and cumulative distance drawn beneath their box, avoiding any reserved rectangle."""
        # reserved_region is an (x1, y1, x2, y2) rectangle a LATER annotator
        # will composite over: in production, MinimapAnnotator's bottom-left
        # panel, obtained from its own occupied_region(). Optional because a
        # caller drawing no minimap has nothing to avoid, and defaulting to
        # None keeps that caller's text placed exactly as before.
        for name, supplied in (
            ('player-track', player_tracks), ('distance', distances), ('speed', speeds),
        ):
            if len(supplied) != len(frames):
                raise ValueError(
                    f'Got {len(supplied)} {name} frames for {len(frames)} video frames — '
                    f'the two must be aligned frame-for-frame.'
                )

        output: list[np.ndarray] = []
        for index, frame in enumerate(frames):
            frame = frame.copy()
            for track_id, track in player_tracks[index].items():
                self._draw_track(
                    frame, track,
                    distances[index].get(track_id),
                    speeds[index].get(track_id),
                    reserved_region,
                )
            output.append(frame)
        return output

    def _draw_track(
        self,
        frame: np.ndarray,
        track: PlayerTrack,
        distance_m: float | None,
        speed: TrackSpeed | None,
        reserved_region: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Draw one player's metric lines beneath their box, in place on the already-copied frame."""
        lines: list[str] = []
        if speed is not None:
            # Omitted entirely when absent rather than drawn as 0.0: a zero is
            # indistinguishable from a stationary player, and the gap rule
            # suppresses speed precisely where the rate is unknown rather than
            # known to be nothing.
            lines.append(f'{speed.speed_ms:.{self.SPEED_DECIMALS}f} m/s')
        if distance_m is not None:
            # Distance and speed vanish together on the ~10% of frames Stage 8
            # could not map, and across clip_3's contiguous 45-frame run, so
            # this readout blinks. The last cumulative distance is
            # deliberately NOT carried across those frames. Distance is a known
            # quantity, so carrying it is possible in a way carrying a speed is
            # not, but a carried figure is sourced from a frame the pipeline
            # could not map, and would read as current while being stale: the
            # same shape as the last-valid-homography fallback this stage's
            # design rejects. A readout that disappears when measurement is
            # unavailable is honest about a real limitation, and the blink is
            # that limitation being visible, which on the clip Stage 8 is
            # verified against is the point rather than a defect.
            lines.append(f'{distance_m:.{self.DISTANCE_DECIMALS}f} m')
        if not lines:
            return

        x1, y1, x2, y2 = track.bbox
        centre_x, foot_y = foot_point(x1, y1, x2, y2)
        pitch = self._line_pitch(lines)
        first_row = self._block_origin(frame, lines, y1, foot_y, centre_x, reserved_region)
        for line_index, text in enumerate(lines):
            self._draw_line(frame, text, centre_x, first_row + line_index * pitch)

    def _line_pitch(self, lines: list[str]) -> int:
        """Return the vertical distance between stacked baselines: the tallest line's measured extent plus a clear gap."""
        # Measured across every line in the block, not just the first:
        # a digit-only line and one with a descender differ in extent,
        # and pitching from the shorter would let the taller abut.
        extents = []
        for text in lines:
            (_, text_h), baseline = cv2.getTextSize(
                text, self.FONT, self.FONT_SCALE, self.THICKNESS,
            )
            extents.append(text_h + baseline)
        return max(extents) + self.LINE_GAP

    def _block_origin(
        self,
        frame: np.ndarray,
        lines: list[str],
        box_top: float,
        foot_y: float,
        centre_x: float,
        reserved_region: tuple[int, int, int, int] | None = None,
    ) -> int:
        """Return the baseline row of the block's first line, flipping it above the box when there is no room below or when the space below is reserved."""
        # The whole stack is positioned ONCE and every line placed relative to
        # it, following EventAnnotator's caption stack. Clamping each line
        # independently collapses them onto the same row whenever both would
        # fall past the bottom edge, and the second putText then overwrites the
        # first as an illegible smear. That is routine rather than exotic here:
        # a near-sideline player's box commonly reaches the bottom of the
        # image, and any foot point below y ~ 700 on a 720-row frame collapsed.
        height = frame.shape[0]
        (_, text_h), baseline = cv2.getTextSize(
            lines[0], self.FONT, self.FONT_SCALE, self.THICKNESS,
        )
        block_height = (len(lines) - 1) * self._line_pitch(lines)
        widest = max(
            cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.THICKNESS)[0][0]
            for text in lines
        )

        below = int(round(foot_y)) + self.VERTICAL_OFFSET
        fits_below = below + block_height + baseline < height
        # A reserved rectangle is checked in BOTH axes. Checking rows alone
        # would flip every player standing low in the frame, including the ~80%
        # of the width the panel does not reach, which would move text that was
        # never at risk and make the output harder to read for a defect
        # affecting one corner.
        blocked_below = self._intersects(
            reserved_region, centre_x, widest, below - text_h, below + block_height + baseline,
        )
        if fits_below and not blocked_below:
            return below

        # Flipped ABOVE the box rather than squeezed against the bottom edge.
        # A player at the bottom of the frame always has room above them
        # (their own box occupies it), whereas below them there is none by
        # definition, so this is the only placement that keeps both lines
        # readable for the case that actually occurs. The same flip serves the
        # reserved-region case: a player standing on the bottom-left sideline
        # is exactly a player whose box extends up out of the panel.
        above = int(round(box_top)) - self.VERTICAL_OFFSET - block_height
        blocked_above = self._intersects(
            reserved_region, centre_x, widest, above - text_h, above + block_height + baseline,
        )
        if above - text_h >= 0 and not blocked_above:
            return above

        # Both placements are blocked or off-frame. Preferring whichever is
        # merely covered over one that is off-frame entirely: a covered line is
        # invisible, but an off-frame row is written outside the array bounds
        # and clamped onto a row it shares with the other line.
        #
        # Both branches are needed. Without this one, a short box low in the
        # bottom-left (feet within ~45 px of the bottom edge, so nothing fits
        # below, with box_top - VERTICAL_OFFSET - block_height still inside
        # the panel's rows) falls past both returns to the top-of-frame
        # anchor. That draws the readout at the player's own centre_x on row
        # text_h, which for a bottom-left player is precisely where
        # PossessionAnnotator's caption sits: a covered readout becomes a
        # readout in the wrong corner, attached to nobody and overlapping
        # another annotator's text. Staying next to the player is what makes a
        # per-player figure legible at all, so a covered placement beside the
        # right player beats a visible one beside the wrong one.
        if fits_below:
            return below
        if above - text_h >= 0:
            return above

        # Neither fits: a frame shorter than the block itself, unreachable at
        # broadcast resolution. Anchored to the top so the lines still stack
        # rather than piling onto one row.
        return text_h

    def _intersects(
        self,
        reserved_region: tuple[int, int, int, int] | None,
        centre_x: float,
        text_width: int,
        top: int,
        bottom: int,
    ) -> bool:
        """Report whether a text block of the given width, centred horizontally, would overlap the reserved rectangle."""
        if reserved_region is None:
            return False

        rx1, ry1, rx2, ry2 = reserved_region
        if rx2 <= rx1 or ry2 <= ry1:
            return False

        # The x-span is computed the way _draw_line places the text (centred
        # on the foot point) rather than from the player's box. The two differ
        # whenever a wide caption hangs past a narrow box, and it is the drawn
        # text, not the box, that the panel would cover.
        left = centre_x - text_width / 2
        return not (left + text_width <= rx1 or left >= rx2 or bottom <= ry1 or top >= ry2)

    def _draw_line(
        self,
        frame: np.ndarray,
        text: str,
        centre_x: float,
        row: int,
    ) -> None:
        """Draw one centred, outlined text line at the given baseline row, clamped horizontally inside the frame."""
        height, width = frame.shape[:2]
        (text_w, text_h), baseline = cv2.getTextSize(
            text, self.FONT, self.FONT_SCALE, self.THICKNESS,
        )

        x = int(round(centre_x - text_w / 2))
        x = min(max(0, x), max(0, width - text_w))
        # Vertical position comes from the block origin, not from a per-line
        # clamp: the clamp is what collapsed the stack. This bound only stops
        # a pathologically short frame from writing outside the array.
        y = min(max(text_h, row), max(text_h, height - baseline - 1))

        # A dark outline under the white text so both stay legible against the
        # pale court floor and the dark crowd alike.
        cv2.putText(
            frame, text, (x, y), self.FONT, self.FONT_SCALE,
            self.OUTLINE_COLOUR, self.OUTLINE_THICKNESS, cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, (x, y), self.FONT, self.FONT_SCALE,
            self.TEXT_COLOUR, self.THICKNESS, cv2.LINE_AA,
        )
