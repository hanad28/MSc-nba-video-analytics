"""
generate_possession.py

Regenerates each clip's possession cache from scratch and prints a per-clip
summary of confirmed frames, distinct holders and real ball anchors.
"""
from __future__ import annotations

import os

import yaml

from basketball.cache.cache_utils import load_cache
from basketball.possession.possession_io import possession_ball_input
from basketball.possession.possession_tracker import PossessionTracker

CLIPS = ['clip_1', 'clip_2', 'clip_3']


def main() -> None:
    """Delete and regenerate every clip's possession cache, printing a summary line per clip."""
    # Under a main() guard, not at import: this deletes possession.pkl and
    # re-runs the tracker, which must never happen as a side effect of
    # something merely importing the module.
    with open('config/default.yaml') as config_file:
        config = yaml.safe_load(config_file)

    tracker = PossessionTracker(config)

    for clip in CLIPS:
        player_tracks = load_cache(f'data/processed/{clip}/player_detections.pkl')
        ball_detections_raw = load_cache(f'data/processed/{clip}/ball_detections.pkl')

        # Gate THEN interpolate, via the shared helper: calling
        # fill_missing(raw) directly would interpolate through exactly the
        # false detections the adopted gate exists to remove, producing a
        # ball track main.py never generates.
        ball_input = possession_ball_input(ball_detections_raw)
        ball_detections = ball_input.filled

        cache_path = f'data/processed/{clip}/possession.pkl'
        if os.path.exists(cache_path):
            os.remove(cache_path)

        possession = tracker.assign_possession(
            player_tracks,
            ball_detections,
            cache_path=cache_path,
        )

        confirmed_frames = sum(1 for p in possession if p != -1)
        distinct_holders = len({p for p in possession if p != -1})
        real_anchors = sum(
            1 for frame in ball_input.gated
            for v in frame.values()
            if v.confidence > 0.0
        )

        print(
            f'{clip}: {confirmed_frames}/{len(possession)} frames confirmed possession, '
            f'{distinct_holders} distinct players | '
            f'{real_anchors} real ball anchors out of {len(possession)} frames'
        )


if __name__ == '__main__':
    main()
