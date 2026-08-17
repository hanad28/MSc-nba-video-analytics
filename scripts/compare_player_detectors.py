"""
compare_player_detectors.py

Read-only rendering script producing a like-for-like visual comparison of the
two player-capable detectors on clip_1, clip_2 and clip_3: the production
player checkpoint (models/players.pt, player = class 1) and the fy4c2
controlled-replication checkpoint (models/ball.pt, whose
Player class index is resolved from the checkpoint's own model.names at
runtime, never hardcoded). Both detectors run the same frame-by-frame
inference call: model(frame, verbose=False), no batching, no explicit
imgsz, ultralytics' default 0.25 NMS confidence floor. That bare call was
PlayerDetector.run_detection()'s exact production form when this comparison
was run (5 August 2026); production has since changed to pass
conf=self.conf_threshold explicitly (since 6 August 2026). This script
deliberately keeps the default-floor call unchanged: its outputs are the
recorded evidence for the two-checkpoint comparison and must stay
reproducible as run. Both detectors then pass through the same filtering
(player class only, conf >= 0.5, matching
PlayerDetector.filter_detections()) and the same drawing code (plain bounding
boxes in one shared colour with a confidence label above each box; no
ellipses, no track IDs, no team colours, no ball marker), so any visible
difference between the paired output videos is attributable to the detectors
alone. ByteTrack is deliberately excluded: this is a raw detection
comparison, and tracking would mask or smooth per-frame detector differences.

Per clip, writes three videos to data/outputs/player_detector_comparison/:
the players.pt / fy4c2 player-class comparison pair, plus an all-classes
fy4c2 render drawing every class (Ball, Clock, Hoop, Overlay, Player, Ref,
Scoreboard) in class-distinct colours with the class name in each label,
showing directly whether referees are classed as Ref rather than Player.
Prints a per-clip summary table (total detections, mean detections per
frame, zero-detection frames) extended with fy4c2 per-class detection
counts. Does not import or modify PlayerDetector, any annotator, or any
production cache/output.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Running this file directly (`python scripts/compare_player_detectors.py`) puts
# scripts/ on sys.path[0], not the repo root, so `basketball` would not be importable.
# Insert the repo root explicitly rather than relying on the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basketball.utils.io_utils import get_video_metadata, load_video, write_video

PLAYERS_MODEL_PATH = 'models/players.pt'
# The fy4c2 replication checkpoint was adopted as production under this name.
FY4C2_MODEL_PATH = 'models/ball.pt'

# models/players.pt is the original 2-class checkpoint: its model.names is
# {0: 'ball', 1: 'player'}: player is class 1, not 0. This matches
# PlayerDetector.CLASS_ID. The fy4c2 checkpoint deliberately gets no such
# constant: its Player index is resolved from its own model.names at runtime.
PLAYERS_PT_CLASS_ID = 1

CLIP_PATHS = {
    'clip_1': 'data/raw/clip_1.mp4',
    'clip_2': 'data/raw/clip_2.mp4',
    'clip_3': 'data/raw/clip_3.mp4',
}

OUTPUT_DIR = Path('data/outputs/player_detector_comparison')

# Matches PlayerDetector.filter_detections()'s hardcoded confidence threshold:
# the production operating point, applied identically to both detectors and to
# the all-classes fy4c2 render, so its per-class counts stay comparable.
CONF_THRESHOLD = 0.5

BOX_COLOUR = (0, 165, 255)  # BGR: orange, one shared colour for both detectors, distinct from PlayerAnnotator's team colours and BallAnnotator's green
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.5

# One distinct colour per class index for the all-classes fy4c2 video; cycles if a
# checkpoint ever carries more classes than the palette.
ALL_CLASS_PALETTE = [
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (0, 165, 255),  # orange
    (0, 255, 0),    # green
    (255, 255, 0),  # cyan
    (0, 0, 255),    # red
]

# write_video() (basketball/utils/io_utils.py) hardcodes the XVID fourcc, which pairs
# with an .avi container, not .mp4, matching main.py's own output naming convention.
OUTPUT_EXTENSION = '.avi'


def resolve_player_class_index(model: YOLO) -> int:
    """Returns the Player class index read from the checkpoint's own names mapping, not an assumption."""
    for class_id, name in model.names.items():
        if name.lower() == 'player':
            return class_id
    raise ValueError(f'No class named "player" found in model.names: {model.names}')


def confirm_players_pt_class_index(model: YOLO) -> None:
    """Raises if the players.pt checkpoint's expected player class index no longer holds; a wrong index renders zero boxes silently."""
    name = model.names.get(PLAYERS_PT_CLASS_ID)
    if name is None or name.lower() != 'player':
        raise ValueError(
            f'{PLAYERS_MODEL_PATH} class {PLAYERS_PT_CLASS_ID} is {name!r}, expected "player" — '
            f'checkpoint may have been swapped (model.names={model.names}).'
        )


def confirm_clip_paths(clip_paths: dict[str, str]) -> None:
    """Raises FileNotFoundError if any expected clip file is missing."""
    for name, path in clip_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f'{name}: expected file not found at {path}.')


def confirm_model_paths() -> None:
    """Raises FileNotFoundError if either checkpoint is missing, naming where the weights are distributed."""
    for path in (PLAYERS_MODEL_PATH, FY4C2_MODEL_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(
                f'{path} not found. Model weights are distributed via the GitHub Release '
                f'for this repository; download them into models/.'
            )


def run_inference(model: YOLO, frames: list[np.ndarray]) -> list[list[tuple[list[float], float, int]]]:
    """Runs the comparison's frozen inference call per frame; returns one (bbox, conf, class_id) list per frame at conf >= CONF_THRESHOLD."""
    # model(frame, verbose=False)[0]: frame-by-frame, no batching, no explicit
    # imgsz, ultralytics' default 0.25 NMS confidence floor. This was
    # PlayerDetector.run_detection()'s exact call when the comparison was run
    # (5 August 2026); production now passes conf=self.conf_threshold
    # explicitly (since 6 August 2026). Deliberately unchanged here: the
    # comparison's outputs are recorded dissertation evidence and must stay
    # reproducible as run. The >= 0.5 gate mirrors filter_detections()'
    # confidence post-filter; class filtering is applied separately by each
    # consumer.
    per_frame_dets: list[list[tuple[list[float], float, int]]] = []
    for frame in frames:
        result = model(frame, verbose=False)[0]
        per_frame_dets.append([
            (box.xyxy[0].tolist(), float(box.conf[0]), int(box.cls[0]))
            for box in result.boxes
            if float(box.conf[0]) >= CONF_THRESHOLD
        ])
    return per_frame_dets


def filter_to_class(
    per_frame_dets: list[list[tuple[list[float], float, int]]],
    class_index: int,
) -> list[list[tuple[list[float], float]]]:
    """Keeps one class's (bbox, conf) detections per frame, the identical player-class gate for both detectors."""
    return [
        [(bbox, conf) for bbox, conf, class_id in frame_dets if class_id == class_index]
        for frame_dets in per_frame_dets
    ]


def draw_player_boxes(
    frames: list[np.ndarray],
    per_frame_dets: list[list[tuple[list[float], float]]],
) -> list[np.ndarray]:
    """Draws a plain rectangle and confidence label per detection, the identical treatment for both detectors."""
    output: list[np.ndarray] = []
    for frame, frame_dets in zip(frames, per_frame_dets):
        frame = frame.copy()
        for bbox, conf in frame_dets:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOUR, BOX_THICKNESS)
            cv2.putText(
                frame,
                f'{conf:.2f}',
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                BOX_COLOUR,
                BOX_THICKNESS,
            )
        output.append(frame)
    return output


def draw_all_class_boxes(
    frames: list[np.ndarray],
    per_frame_dets: list[list[tuple[list[float], float, int]]],
    class_names: dict[int, str],
) -> list[np.ndarray]:
    """Draws every detection in a class-distinct colour with the class name in each label."""
    output: list[np.ndarray] = []
    for frame, frame_dets in zip(frames, per_frame_dets):
        frame = frame.copy()
        for bbox, conf, class_id in frame_dets:
            colour = ALL_CLASS_PALETTE[class_id % len(ALL_CLASS_PALETTE)]
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, BOX_THICKNESS)
            cv2.putText(
                frame,
                f'{class_names[class_id]} {conf:.2f}',
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                colour,
                BOX_THICKNESS,
            )
        output.append(frame)
    return output


def print_summary_table(
    clip_name: str,
    n_frames: int,
    results: dict[str, list[list[tuple[list[float], float]]]],
) -> None:
    """Prints total detections, mean detections per frame, and zero-detection frame count for each detector."""
    print(f'\n  {clip_name} summary ({n_frames} frames):')
    print(f'    {"detector":<28}{"total":>8}{"mean/frame":>14}{"zero-det frames":>18}')
    for label, per_frame_dets in results.items():
        counts = [len(frame_dets) for frame_dets in per_frame_dets]
        total = sum(counts)
        n_zero = sum(1 for count in counts if count == 0)
        print(f'    {label:<28}{total:>8}{total / n_frames:>14.2f}{n_zero:>18}')


def print_fy4c2_class_counts(
    per_frame_dets: list[list[tuple[list[float], float, int]]],
    class_names: dict[int, str],
) -> None:
    """Prints total fy4c2 detections per class, directly showing whether referees land in Ref rather than Player."""
    counts = Counter(class_id for frame_dets in per_frame_dets for _, _, class_id in frame_dets)
    print(f'\n    fy4c2 per-class detections (conf >= {CONF_THRESHOLD}):')
    for class_id in sorted(class_names):
        print(f'      {class_names[class_id]:<12}{counts[class_id]:>8}')


def write_annotated_video(annotated_frames: list[np.ndarray], clip_name: str, suffix: str, fps: float) -> None:
    """Writes one annotated video into the comparison output directory."""
    output_path = OUTPUT_DIR / f'{clip_name}_{suffix}{OUTPUT_EXTENSION}'
    write_video(annotated_frames, str(output_path), fps=fps)
    print(f'  -> written to {output_path}')


def process_clip(
    clip_name: str,
    clip_path: str,
    players_model: YOLO,
    fy4c2_model: YOLO,
    fy4c2_class_index: int,
) -> None:
    """Runs both detectors on one clip, writes the comparison pair plus the fy4c2 all-classes video, and prints the summary."""
    frames = [frame for _, frame in load_video(clip_path)]
    fps = get_video_metadata(clip_path)['fps']
    print(f'\n{clip_name}: {len(frames)} frames loaded at {fps:.2f} fps')

    players_pt_dets = filter_to_class(run_inference(players_model, frames), PLAYERS_PT_CLASS_ID)
    write_annotated_video(draw_player_boxes(frames, players_pt_dets), clip_name, 'players_pt', fps)

    fy4c2_all_dets = run_inference(fy4c2_model, frames)
    fy4c2_player_dets = filter_to_class(fy4c2_all_dets, fy4c2_class_index)
    write_annotated_video(draw_player_boxes(frames, fy4c2_player_dets), clip_name, 'fy4c2', fps)
    write_annotated_video(draw_all_class_boxes(frames, fy4c2_all_dets, fy4c2_model.names), clip_name, 'fy4c2_all_classes', fps)

    print_summary_table(clip_name, len(frames), {
        'players.pt': players_pt_dets,
        'ball_fy4c2_replication.pt': fy4c2_player_dets,
    })
    print_fy4c2_class_counts(fy4c2_all_dets, fy4c2_model.names)


def main() -> None:
    confirm_clip_paths(CLIP_PATHS)
    confirm_model_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    players_model = YOLO(PLAYERS_MODEL_PATH)
    confirm_players_pt_class_index(players_model)
    print(f'{PLAYERS_MODEL_PATH}: using class index {PLAYERS_PT_CLASS_ID} (model.names={players_model.names})')

    fy4c2_model = YOLO(FY4C2_MODEL_PATH)
    fy4c2_class_index = resolve_player_class_index(fy4c2_model)
    print(f'{FY4C2_MODEL_PATH}: resolved Player class index {fy4c2_class_index} (model.names={fy4c2_model.names})')

    for clip_name, clip_path in CLIP_PATHS.items():
        process_clip(clip_name, clip_path, players_model, fy4c2_model, fy4c2_class_index)


if __name__ == '__main__':
    main()
