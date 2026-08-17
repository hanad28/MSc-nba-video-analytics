"""
player_annotator.py

Renders player tracking results onto video frames as foot-position ellipses
with persistent track ID labels, coloured by team assignment.
"""
import numpy as np

from basketball.annotators.annotation_utils import annotate_player
from basketball.detection.player_detector import PlayerTrack


class PlayerAnnotator:
    """Annotates video frames with player tracking ellipses and track ID labels."""

    # Defaults for team 1 / team 2, overridable per instance (and, via
    # main.py, per clip) since a fixed pair misrenders any clip whose actual
    # jersey colours don't match: clip_3's red Rockets render dark
    # blue under these defaults. Team 0 (unknown/unresolved) is not
    # configurable: it is never a "this clip's team colour", just a status.
    DEFAULT_TEAM_1_COLOUR: tuple[int, int, int] = (255, 245, 238)  # BGR: off-white
    DEFAULT_TEAM_2_COLOUR: tuple[int, int, int] = (128, 0, 0)      # BGR: dark blue
    UNKNOWN_TEAM_COLOUR: tuple[int, int, int] = (128, 128, 128)    # BGR: grey (constant)

    def __init__(
        self,
        team_1_colour: tuple[int, int, int] | None = None,
        team_2_colour: tuple[int, int, int] | None = None,
    ) -> None:
        self.team_colours: dict[int, tuple[int, int, int]] = {
            1: tuple(team_1_colour) if team_1_colour is not None else self.DEFAULT_TEAM_1_COLOUR,
            2: tuple(team_2_colour) if team_2_colour is not None else self.DEFAULT_TEAM_2_COLOUR,
            0: self.UNKNOWN_TEAM_COLOUR,
        }

    def draw(
        self,
        frames: list[np.ndarray],
        tracks: list[dict[int, PlayerTrack]],
        team_assignment: list[dict[int, int]] | None = None,
    ) -> list[np.ndarray]:
        """Return annotated copies of all frames with player ellipses and track IDs drawn."""
        if len(tracks) != len(frames):
            raise ValueError(
                f'Got {len(tracks)} track frames for {len(frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )

        if team_assignment is not None and len(team_assignment) != len(frames):
            raise ValueError(
                f'Got {len(team_assignment)} team assignment frames for {len(frames)} video '
                f'frames — the two must be aligned frame-for-frame.'
            )

        output: list[np.ndarray] = []
        for frame_idx, (frame, frame_tracks) in enumerate(zip(frames, tracks)):
            frame = frame.copy()
            for track_id, player_track in frame_tracks.items():
                if team_assignment is not None:
                    frame_teams = team_assignment[frame_idx]
                    team_id = frame_teams.get(track_id, 0)
                    if team_id not in self.team_colours:
                        raise ValueError(
                            f'Frame {frame_idx} assigns player {track_id} unknown team id '
                            f'{team_id}; expected one of {sorted(self.team_colours)}.'
                        )
                    colour = self.team_colours[team_id]
                else:
                    colour = (0, 0, 255)  # red fallback: no team data

                frame = annotate_player(frame, player_track.bbox, track_id, colour)
            output.append(frame)
        return output
