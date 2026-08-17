"""
label_ground_truth.py

Interactive OpenCV ground-truth labelling tool over the sampled frames
(every 10th) of clip_1, clip_2 and clip_3: the local-machine half of the
labelling workflow. Runs where a display
exists: JupyterHub has none ($DISPLAY is empty), while the local
machine has the clips but no torch/ultralytics, so this tool deliberately
imports nothing from the GPU stack. Boxes are pre-populated from
data/annotations/{clip}_seed.json (written on JupyterHub by
scripts/extract_labelling_seed.py) when present, and labelling works from
blank frames without seeds too. Progress persists to
data/annotations/{clip}_gt.txt in MOT Challenge format via
save_mot_annotations()/load_mot_annotations() from evaluation/ground_truth.py,
so labelling can span several sittings.

Keys: n/p next/previous sampled frame (crossing clip boundaries), s save all
clips, q save all and quit, d then click deletes a box, click inside a box
then type digits and Enter sets its ID, drag on empty space draws a new box
then prompts for an ID, Esc cancels the pending selection/ID/delete.

A ground-truth ID identifies one physical person for the whole clip: a
player who leaves frame and returns keeps their original ID. Label on-court
players only: no referees, bench or crowd. A save is refused (naming the
frame) while any frame holds duplicate IDs or a zero/negative-dimension
box; a labelled frame with fewer than 6 or more than 12 boxes warns but
does not block. Boxes never given an ID are working state and are not
written to the ground truth. A clip whose previously saved labels have all
been removed refuses to save (the on-disk file stays untouched, to be
re-labelled or deleted by hand deliberately), while a clip that was never
labelled is simply skipped.
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Running this file directly (`python scripts/label_ground_truth.py`) puts
# scripts/ on sys.path[0], not the repo root, so `basketball` and `evaluation`
# would not be importable. Insert the repo root explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basketball.utils.io_utils import load_video
from evaluation.ground_truth import GTAnnotation, load_mot_annotations, save_mot_annotations

CLIPS = ('clip_1', 'clip_2', 'clip_3')

# Ground truth is labelled on every 10th frame.
# Must match SAMPLE_STRIDE in scripts/extract_labelling_seed.py, pinned by a test.
SAMPLE_STRIDE = 10

CLIP_PATH_TEMPLATE = 'data/raw/{clip}.mp4'
SEED_PATH_TEMPLATE = 'data/annotations/{clip}_seed.json'
GT_PATH_TEMPLATE = 'data/annotations/{clip}_gt.txt'

MIN_EXPECTED_BOXES = 6
MAX_EXPECTED_BOXES = 12

# Seed boxes coinciding with a saved ground-truth box at or above this IoU are
# not restored on startup. Same convention as MOTEvaluator.IOU_THRESHOLD (0.5),
# deliberately not imported: evaluation.mot_metrics pulls in the GPU stack this
# tool must run without. The equality is pinned by a test.
SEED_MATCH_IOU_THRESHOLD = 0.5

WINDOW_NAME = 'label_ground_truth'
BANNER_HEIGHT_PX = 46
ASSIGNED_COLOUR = (0, 200, 0)      # BGR: green; box has a ground-truth ID
UNASSIGNED_COLOUR = (0, 165, 255)  # BGR: orange; seed/new box awaiting an ID, labelled ?
SELECTED_COLOUR = (255, 255, 255)  # BGR: white; box currently receiving digits
CLICK_TOLERANCE_PX = 5


@dataclass
class LabelledBox:
    bbox: list[float]           # [x1, y1, x2, y2]
    track_id: int | None = None


@dataclass
class ClipState:
    name: str
    sampled_indices: list[int]
    frames: dict[int, np.ndarray]
    frame_boxes: dict[int, list[LabelledBox]] = field(default_factory=dict)
    unsampled_gt_frames: list[int] = field(default_factory=list)  # gt.txt rows outside the current sample; blocks saving
    had_saved_gt: bool = False  # gt.txt holds >=1 annotation (found at load, or written this session); an emptied clip refuses to save


def sampled_frame_indices(frame_count: int, stride: int = SAMPLE_STRIDE) -> list[int]:
    """Frame indices carrying ground truth: every stride-th frame, zero-indexed."""
    return list(range(0, frame_count, stride))


def load_seed_boxes(path: str) -> dict[int, list[LabelledBox]]:
    """Load a seed JSON as unassigned LabelledBoxes keyed by int frame index; missing file means no seed."""
    if not Path(path).exists():
        return {}

    with open(path, 'r') as f:
        raw = json.load(f)

    return {
        int(frame): [LabelledBox(bbox=[float(v) for v in bbox]) for bbox in boxes]
        for frame, boxes in raw.items()
    }


def annotations_to_frame_boxes(annotations: list[GTAnnotation]) -> dict[int, list[LabelledBox]]:
    """Group loaded ground-truth annotations into per-frame LabelledBoxes with their IDs."""
    frame_boxes: dict[int, list[LabelledBox]] = {}
    for annotation in annotations:
        frame_boxes.setdefault(annotation.frame, []).append(
            LabelledBox(bbox=list(annotation.bbox), track_id=annotation.track_id)
        )
    return frame_boxes


def frame_boxes_to_annotations(frame_boxes: dict[int, list[LabelledBox]]) -> list[GTAnnotation]:
    """Flatten per-frame boxes into GTAnnotations; boxes without an assigned ID are working state and are dropped."""
    return [
        GTAnnotation(frame=frame, track_id=box.track_id, bbox=list(box.bbox))
        for frame in sorted(frame_boxes)
        for box in frame_boxes[frame]
        if box.track_id is not None
    ]


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def initial_frame_boxes(
    sampled_indices: list[int],
    gt_frame_boxes: dict[int, list[LabelledBox]],
    seed_frame_boxes: dict[int, list[LabelledBox]],
) -> dict[int, list[LabelledBox]]:
    """Startup state per sampled frame: saved ground truth first, plus seed boxes no saved box already covers."""
    # Saved boxes alone would strip a partially-labelled frame of its remaining
    # detector suggestions between sittings; restoring every seed box would pile
    # duplicates onto finished frames. Seed boxes come back only where no saved
    # box overlaps them at >= SEED_MATCH_IOU_THRESHOLD.
    frame_boxes: dict[int, list[LabelledBox]] = {}
    for frame in sampled_indices:
        saved = copy.deepcopy(gt_frame_boxes.get(frame) or [])
        unmatched_seeds = [
            copy.deepcopy(seed_box)
            for seed_box in seed_frame_boxes.get(frame) or []
            if not any(
                _box_iou(seed_box.bbox, saved_box.bbox) >= SEED_MATCH_IOU_THRESHOLD
                for saved_box in saved
            )
        ]
        frame_boxes[frame] = saved + unmatched_seeds
    return frame_boxes


def unsampled_gt_frames(gt_frame_boxes: dict[int, list[LabelledBox]], sampled_indices: list[int]) -> list[int]:
    """Frames present in the loaded ground truth but absent from the current sample, evidence of a stride change."""
    sampled = set(sampled_indices)
    return sorted(frame for frame in gt_frame_boxes if frame not in sampled)


def frame_validation_errors(frame_boxes: dict[int, list[LabelledBox]]) -> list[str]:
    """Blocking problems that refuse a save: duplicate assigned IDs or degenerate assigned boxes, per frame."""
    errors: list[str] = []
    for frame in sorted(frame_boxes):
        assigned = [box for box in frame_boxes[frame] if box.track_id is not None]

        ids = [box.track_id for box in assigned]
        duplicates = sorted({track_id for track_id in ids if ids.count(track_id) > 1})
        if duplicates:
            errors.append(f'frame {frame}: duplicate IDs {duplicates}')

        for box in assigned:
            x1, y1, x2, y2 = box.bbox
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                errors.append(f'frame {frame}: zero/negative-dimension box {box.bbox} (ID {box.track_id})')
    return errors


def frame_count_warnings(frame_boxes: dict[int, list[LabelledBox]]) -> list[str]:
    """Non-blocking warnings for labelled frames whose box count falls outside the expected on-court range."""
    warnings: list[str] = []
    for frame in sorted(frame_boxes):
        assigned = sum(1 for box in frame_boxes[frame] if box.track_id is not None)
        if 0 < assigned < MIN_EXPECTED_BOXES or assigned > MAX_EXPECTED_BOXES:
            warnings.append(
                f'frame {frame}: {assigned} labelled boxes '
                f'(expected {MIN_EXPECTED_BOXES}-{MAX_EXPECTED_BOXES} on-court players)'
            )
    return warnings


def labelled_frame_count(frame_boxes: dict[int, list[LabelledBox]]) -> int:
    """Number of frames carrying at least one ID-assigned box."""
    return sum(1 for boxes in frame_boxes.values() if any(box.track_id is not None for box in boxes))


def progress_line(clip_name: str, frame_boxes: dict[int, list[LabelledBox]], sampled_indices: list[int]) -> str:
    """One-line per-clip labelling progress, e.g. 'clip_1: 4/12 sampled frames labelled'."""
    return f'{clip_name}: {labelled_frame_count(frame_boxes)}/{len(sampled_indices)} sampled frames labelled'


def save_clip(clip: ClipState) -> bool:
    """Validate and persist one clip's assigned boxes to its MOT gt file; returns False (with reasons) on refusal."""
    # Saving rewrites the whole gt file from the sampled frames, so rows on
    # frames outside the sample would be silently dropped. A stride change is
    # a decision the user must make deliberately, not one the tool papers over.
    if clip.unsampled_gt_frames:
        print(
            f'[label] {clip.name}: NOT saved — {GT_PATH_TEMPLATE.format(clip=clip.name)} holds rows on '
            f'frames outside the current sample: {clip.unsampled_gt_frames}. The sampling stride '
            f'appears to have changed; saving would silently drop those rows.'
        )
        return False

    annotations = frame_boxes_to_annotations(clip.frame_boxes)
    if not annotations:
        if clip.had_saved_gt:
            # Every previously saved label has been removed this sitting. A skip
            # here would report a clean save while the old rows stay on disk and
            # reload next sitting; refuse instead, and let the labeller either
            # re-label or delete the file by hand deliberately.
            print(
                f'[label] {clip.name}: NOT saved — this clip previously had saved labels and they have '
                f'all been removed this sitting. {GT_PATH_TEMPLATE.format(clip=clip.name)} has been left '
                f'untouched; re-label, or delete the file by hand if clearing it is deliberate.'
            )
            return False

        # An empty gt.txt would be committed AND scores as a well-formed
        # all-zero MOTResult; an unlabelled clip must stay absent from the
        # evaluation, not look legitimately evaluated. Skipping is not a
        # refusal, so it never blocks q.
        print(f'[label] {clip.name}: skipped as unlabelled — no assigned boxes, not writing an empty ground-truth file.')
        return True

    errors = frame_validation_errors(clip.frame_boxes)
    if errors:
        print(f'[label] {clip.name}: NOT saved — fix these first:')
        for error in errors:
            print(f'  {error}')
        return False

    for warning in frame_count_warnings(clip.frame_boxes):
        print(f'[label] {clip.name} warning: {warning}')

    save_mot_annotations(annotations, GT_PATH_TEMPLATE.format(clip=clip.name))
    # An in-session save makes this clip "previously saved" too: clearing it and
    # saving again must hit the refusal above, not the never-labelled skip.
    clip.had_saved_gt = True
    print(f'[label] saved — {progress_line(clip.name, clip.frame_boxes, clip.sampled_indices)}')
    return True


def load_clip_state(clip_name: str) -> ClipState:
    """Load a clip's sampled frames and its startup boxes (saved ground truth plus uncovered seed boxes)."""
    clip_path = CLIP_PATH_TEMPLATE.format(clip=clip_name)
    if not Path(clip_path).exists():
        raise FileNotFoundError(f'{clip_name}: expected clip not found at {clip_path}.')

    frames: dict[int, np.ndarray] = {}
    frame_count = 0
    for index, frame in load_video(clip_path):
        frame_count = index + 1
        if index % SAMPLE_STRIDE == 0:
            frames[index] = frame

    sampled = sampled_frame_indices(frame_count)

    gt_path = GT_PATH_TEMPLATE.format(clip=clip_name)
    gt_boxes = annotations_to_frame_boxes(load_mot_annotations(gt_path)) if Path(gt_path).exists() else {}
    seed_boxes = load_seed_boxes(SEED_PATH_TEMPLATE.format(clip=clip_name))

    return ClipState(
        name=clip_name,
        sampled_indices=sampled,
        frames=frames,
        frame_boxes=initial_frame_boxes(sampled, gt_boxes, seed_boxes),
        unsampled_gt_frames=unsampled_gt_frames(gt_boxes, sampled),
        had_saved_gt=bool(gt_boxes),
    )


class LabellingSession:
    """Holds the interactive state (position, selection, pending ID digits, delete mode) and renders frames."""

    def __init__(self, clips: list[ClipState]) -> None:
        self.clips = clips
        self.positions = [
            (clip_index, sample_index)
            for clip_index, clip in enumerate(clips)
            for sample_index in range(len(clip.sampled_indices))
        ]
        self.position = 0
        self.selected: LabelledBox | None = None
        self.id_buffer = ''
        self.delete_mode = False
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None

    def current(self) -> tuple[ClipState, int]:
        clip_index, sample_index = self.positions[self.position]
        clip = self.clips[clip_index]
        return clip, clip.sampled_indices[sample_index]

    def current_boxes(self) -> list[LabelledBox]:
        clip, frame = self.current()
        return clip.frame_boxes[frame]

    def _clear_pending(self) -> None:
        self.selected = None
        self.id_buffer = ''
        self.delete_mode = False
        self.drag_start = None
        self.drag_current = None

    def move(self, step: int) -> None:
        self.position = max(0, min(len(self.positions) - 1, self.position + step))
        self._clear_pending()

    def _box_at(self, x: int, y: int) -> LabelledBox | None:
        containing = [
            box for box in self.current_boxes()
            if box.bbox[0] <= x <= box.bbox[2] and box.bbox[1] <= y <= box.bbox[3]
        ]
        if not containing:
            return None
        # The smallest containing box is the most specific under overlap.
        return min(containing, key=lambda box: (box.bbox[2] - box.bbox[0]) * (box.bbox[3] - box.bbox[1]))

    def _delete_box(self, target: LabelledBox) -> None:
        """Remove exactly the given box object; a value-equal twin must survive."""
        # LabelledBox is a dataclass with value-based __eq__, so list.remove()
        # would delete the first equal box rather than the clicked one.
        boxes = self.current_boxes()
        del boxes[next(i for i, box in enumerate(boxes) if box is target)]

        if target is self.selected:
            # Digits typed after the delete would otherwise land on a box that
            # is no longer in frame_boxes and be lost.
            self.selected = None
            self.id_buffer = ''

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.delete_mode:
                box = self._box_at(x, y)
                if box is not None:
                    self._delete_box(box)
                self.delete_mode = False
                return
            self.drag_start = (x, y)
            self.drag_current = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            start, self.drag_start, self.drag_current = self.drag_start, None, None
            moved = abs(x - start[0]) > CLICK_TOLERANCE_PX or abs(y - start[1]) > CLICK_TOLERANCE_PX

            if not moved:
                self.selected = self._box_at(x, y)
                self.id_buffer = ''
                return

            # A drag that began inside an existing box is treated as a click-select,
            # so a slipped click cannot silently spawn a box over a player.
            if self._box_at(*start) is not None:
                self.selected = self._box_at(*start)
                self.id_buffer = ''
                return

            x1, x2 = sorted((start[0], x))
            y1, y2 = sorted((start[1], y))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                return
            new_box = LabelledBox(bbox=[float(x1), float(y1), float(x2), float(y2)])
            self.current_boxes().append(new_box)
            self.selected = new_box
            self.id_buffer = ''

    def handle_key(self, key: int) -> bool:
        """Apply one key press; returns False when the session should end (q with a clean save)."""
        if key == ord('n'):
            self.move(1)
        elif key == ord('p'):
            self.move(-1)
        elif key == ord('d'):
            # Entering delete mode with a half-typed ID is ambiguous: drop the
            # pending selection and buffer rather than guessing.
            self.selected = None
            self.id_buffer = ''
            self.delete_mode = True
        elif key == 27:  # Esc
            self._clear_pending()
        elif ord('0') <= key <= ord('9') and self.selected is not None:
            self.id_buffer += chr(key)
        elif key == 13 and self.selected is not None:  # Enter
            if self.id_buffer:
                self.selected.track_id = int(self.id_buffer)
            self.selected = None
            self.id_buffer = ''
        elif key == ord('s'):
            for clip in self.clips:
                save_clip(clip)
        elif key == ord('q'):
            # Evaluate eagerly: a lazy all() would stop at the first refusal and
            # skip saving (and validating) every later clip.
            results = [save_clip(clip) for clip in self.clips]
            if all(results):
                return True
            print('[label] not quitting — fix the errors above and press q again (or close the window to abandon).')
        return False

    def render(self) -> np.ndarray:
        clip, frame_index = self.current()
        image = clip.frames[frame_index].copy()

        _, sample_index = self.positions[self.position]
        mode = ' [DELETE: click a box]' if self.delete_mode else ''
        pending = f' [ID: {self.id_buffer}_]' if self.selected is not None else ''
        header = (
            f'{clip.name}  frame {frame_index}  '
            f'{sample_index + 1}/{len(clip.sampled_indices)}  '
            f'{labelled_frame_count(clip.frame_boxes)}/{len(clip.sampled_indices)} labelled{mode}{pending}'
        )
        keys = 'n/p frame | s save | q save+quit | d+click delete | click+digits+Enter set ID | drag new box'

        # The banner goes down first so boxes near the top edge stay visible on
        # top of it; painting it last occluded their ID text, and an invisible
        # ID is how a labeller assigns a duplicate that then blocks saving.
        cv2.rectangle(image, (0, 0), (image.shape[1], BANNER_HEIGHT_PX), (0, 0, 0), -1)
        cv2.putText(image, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, keys, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        for box in self.current_boxes():
            x1, y1, x2, y2 = (int(v) for v in box.bbox)
            if box is self.selected:
                colour = SELECTED_COLOUR
            elif box.track_id is None:
                colour = UNASSIGNED_COLOUR
            else:
                colour = ASSIGNED_COLOUR
            label = '?' if box.track_id is None else str(box.track_id)
            if box is self.selected and self.id_buffer:
                label = self.id_buffer
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            # A label above a box whose top edge sits inside the banner strip
            # would collide with the banner text; drop it below the top edge.
            label_y = y1 + 16 if y1 < BANNER_HEIGHT_PX else y1 - 6
            cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        if self.drag_start is not None and self.drag_current is not None:
            cv2.rectangle(image, self.drag_start, self.drag_current, SELECTED_COLOUR, 1)

        return image


def main() -> None:
    clips = [load_clip_state(clip_name) for clip_name in CLIPS]
    for clip in clips:
        print(progress_line(clip.name, clip.frame_boxes, clip.sampled_indices))

    session = LabellingSession(clips)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, session.on_mouse)

    while True:
        # Checked before imshow, which would otherwise silently recreate a
        # closed window without its mouse callback.
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print('[label] window closed — abandoned without saving.')
            break

        cv2.imshow(WINDOW_NAME, session.render())
        key = cv2.waitKey(30) & 0xFF
        if key != 255 and session.handle_key(key):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
