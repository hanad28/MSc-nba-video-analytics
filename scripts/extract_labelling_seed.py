"""
extract_labelling_seed.py

Read-only JupyterHub-side extractor for the ground-truth labelling workflow.
For each clip, samples every 10th frame of
the cached player detections (data/processed/{clip}/player_detections.pkl,
loaded via load_cache; no inference, no video access, no cache writes) and
writes the player boxes for those frames to data/annotations/{clip}_seed.json,
keyed by frame index. The seeds pre-populate scripts/label_ground_truth.py,
which runs on the local Windows machine instead: labelling needs an
interactive OpenCV window and JupyterHub has no display, while the local
machine has the clips but no torch/ultralytics, so detections cross over as
these small JSON files. Seed files are a few KB and committed to git;
data/annotations/ is tracked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Running this file directly (`python scripts/extract_labelling_seed.py`) puts
# scripts/ on sys.path[0], not the repo root, so `basketball` would not be importable.
# Insert the repo root explicitly rather than relying on the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basketball.cache.cache_utils import load_cache

CLIPS = ('clip_1', 'clip_2', 'clip_3')

# Ground truth is labelled on every 10th frame.
# Must match SAMPLE_STRIDE in scripts/label_ground_truth.py, pinned by a test.
SAMPLE_STRIDE = 10

CACHE_PATH_TEMPLATE = 'data/processed/{clip}/player_detections.pkl'
SEED_PATH_TEMPLATE = 'data/annotations/{clip}_seed.json'


def sample_seed_boxes(cached_tracks: list[dict], stride: int = SAMPLE_STRIDE) -> dict[int, list[list[float]]]:
    """Player boxes for every stride-th cached frame, keyed by real (0-indexed) frame index."""
    return {
        frame: [list(track.bbox) for track in cached_tracks[frame].values()]
        for frame in range(0, len(cached_tracks), stride)
    }


def write_seed(seed: dict[int, list[list[float]]], path: str) -> None:
    """Write a seed mapping as JSON; frame keys become strings because JSON objects cannot key on ints."""
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump({str(frame): boxes for frame, boxes in seed.items()}, f, indent=2)


def main() -> None:
    for clip in CLIPS:
        cache_path = CACHE_PATH_TEMPLATE.format(clip=clip)
        if not Path(cache_path).exists():
            raise FileNotFoundError(
                f'{clip}: player detection cache not found at {cache_path} — '
                f'run the pipeline on JupyterHub to generate it first.'
            )

        cached_tracks = load_cache(cache_path)
        seed = sample_seed_boxes(cached_tracks)
        write_seed(seed, SEED_PATH_TEMPLATE.format(clip=clip))

        total_boxes = sum(len(boxes) for boxes in seed.values())
        print(
            f'{clip}: {len(seed)} sampled frames ({len(cached_tracks)} cached), '
            f'{total_boxes} boxes -> {SEED_PATH_TEMPLATE.format(clip=clip)}'
        )


if __name__ == '__main__':
    main()
