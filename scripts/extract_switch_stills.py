"""
extract_switch_stills.py

Read-only frame-extraction script. Pulls still frames out of the already
fully annotated videos data/outputs/clip_1_annotated.avi and
data/outputs/clip_3_annotated.avi -- both already carry player ellipses and
track IDs rendered by the production pipeline, so this script performs no
detection, tracking or annotation of its own.

Frames are read with a single sequential pass per clip (cap.read() from
frame 0, checking each frame index against a required set) rather than
cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, ...) seeking. Seeking is unreliable
on compressed formats like this project's XVID .avi outputs -- decoding
resumes from the nearest keyframe, so a seek can silently return a
neighbouring frame instead of the exact one requested, and exact frame
identity is the entire point of this script. Both clips are well under 300
frames, so a full sequential pass is effectively free.

This is a lightweight one-off diagnostic script, not a pipeline component --
it deliberately does not import anything from basketball/.

For each switch frame in SWITCH_FRAMES (clip_3), saves the frame itself plus
the frames 10 before and 10 after, to visually inspect what the tracker was
doing around each ID switch. Also saves frame 110 and its +/-10 neighbours
from both clip_1 and clip_3, to check whether the frame-110 switch appearing
in both clips is coincidence or shares a common cause (e.g. a simultaneous
camera cut or occlusion pattern).

Output goes to data/outputs/switch_stills/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

CLIP_1_PATH = Path('data/outputs/clip_1_annotated.avi')
CLIP_3_PATH = Path('data/outputs/clip_3_annotated.avi')
OUTPUT_DIR = Path('data/outputs/switch_stills')

SWITCH_FRAMES = [40, 90, 150, 210, 230, 240]
SHARED_FRAME = 110
NEIGHBOUR_OFFSET = 10


def _collect_required_frames(clip_name: str, switch_frames: list[int], include_shared: bool) -> dict[int, list[str]]:
    """Build a map of frame index -> output filenames needed from one clip."""
    required: dict[int, list[str]] = {}

    def _add(frame_idx: int, filename: str) -> None:
        required.setdefault(frame_idx, []).append(filename)

    for switch_frame in switch_frames:
        before_idx = switch_frame - NEIGHBOUR_OFFSET
        after_idx = switch_frame + NEIGHBOUR_OFFSET
        _add(before_idx, f'{clip_name}_f{before_idx}_before{switch_frame}.png')
        _add(switch_frame, f'{clip_name}_f{switch_frame}_switch.png')
        _add(after_idx, f'{clip_name}_f{after_idx}_after{switch_frame}.png')

    if include_shared:
        before_idx = SHARED_FRAME - NEIGHBOUR_OFFSET
        after_idx = SHARED_FRAME + NEIGHBOUR_OFFSET
        _add(before_idx, f'{clip_name}_f{before_idx}_shared110_before.png')
        _add(SHARED_FRAME, f'{clip_name}_f{SHARED_FRAME}_shared110.png')
        _add(after_idx, f'{clip_name}_f{after_idx}_shared110_after.png')

    return required


def _extract_clip(path: Path, clip_name: str, required_frames: dict[int, list[str]]) -> None:
    """Sequentially decode path from frame 0, writing out every frame present in required_frames."""
    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f'{clip_name}: {path} (last modified {mtime})')

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f'could not open {path}')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'  reported frame count: {frame_count}')

    written: set[int] = set()
    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_idx in required_frames:
            for filename in required_frames[frame_idx]:
                out_path = OUTPUT_DIR / filename
                cv2.imwrite(str(out_path), frame)
                print(f'  saved {out_path}')
            written.add(frame_idx)

        frame_idx += 1

    cap.release()

    for missing_idx in sorted(set(required_frames) - written):
        if missing_idx < 0:
            reason = 'negative index -- offset underflows the start of the clip'
        elif missing_idx >= frame_count:
            reason = f'beyond the reported frame count ({frame_count})'
        else:
            reason = 'frame could not be decoded'
        for filename in required_frames[missing_idx]:
            print(f'  WARNING: could not extract frame {missing_idx} for {filename} ({reason})')


def main() -> None:
    """Extract all switch-frame and shared-frame stills for clip_1 and clip_3."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CLIP_1_PATH.exists():
        raise FileNotFoundError(f'{CLIP_1_PATH} not found')
    if not CLIP_3_PATH.exists():
        raise FileNotFoundError(f'{CLIP_3_PATH} not found')

    clip_3_required = _collect_required_frames('clip_3', SWITCH_FRAMES, include_shared=True)
    clip_1_required = _collect_required_frames('clip_1', [], include_shared=True)

    print(f'Extracting clip_3 switch-frame windows {SWITCH_FRAMES} and shared frame {SHARED_FRAME}')
    _extract_clip(CLIP_3_PATH, 'clip_3', clip_3_required)

    print(f'Extracting clip_1 shared frame {SHARED_FRAME}')
    _extract_clip(CLIP_1_PATH, 'clip_1', clip_1_required)

    print('Done.')


if __name__ == '__main__':
    main()
