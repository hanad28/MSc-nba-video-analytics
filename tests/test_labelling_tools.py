"""Unit tests for the ground-truth labelling toolchain's non-interactive logic.

Covers the seed extractor (scripts/extract_labelling_seed.py) and the pure
functions of the labelling tool (scripts/label_ground_truth.py): sampling,
seed JSON round-tripping, the save guards and progress reporting. The
interactive OpenCV loop itself needs a display and is deliberately untested.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2

import scripts.label_ground_truth as lgt
from evaluation.ground_truth import load_mot_annotations, save_mot_annotations
from scripts.extract_labelling_seed import SAMPLE_STRIDE as EXTRACTOR_STRIDE
from scripts.extract_labelling_seed import sample_seed_boxes, write_seed
from scripts.label_ground_truth import (
    SAMPLE_STRIDE,
    ClipState,
    LabelledBox,
    LabellingSession,
    annotations_to_frame_boxes,
    frame_boxes_to_annotations,
    frame_count_warnings,
    frame_validation_errors,
    initial_frame_boxes,
    labelled_frame_count,
    load_seed_boxes,
    progress_line,
    sampled_frame_indices,
    save_clip,
    unsampled_gt_frames,
)


def box(track_id: int | None, bbox: list[float] | None = None) -> LabelledBox:
    return LabelledBox(bbox=bbox or [10.0, 20.0, 40.0, 80.0], track_id=track_id)


def make_clip(
    name: str,
    frame_boxes: dict[int, list[LabelledBox]],
    unsampled: list[int] | None = None,
    had_saved_gt: bool = False,
) -> ClipState:
    return ClipState(
        name=name,
        sampled_indices=sorted(frame_boxes) or [0],
        frames={},
        frame_boxes=frame_boxes,
        unsampled_gt_frames=unsampled or [],
        had_saved_gt=had_saved_gt,
    )


# --- sampling ---------------------------------------------------------------

def test_sampled_frame_indices_takes_every_tenth_frame_from_zero():
    assert sampled_frame_indices(100) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_sampled_frame_indices_includes_a_final_frame_on_the_stride():
    assert sampled_frame_indices(101)[-1] == 100


def test_sampled_frame_indices_edge_cases():
    assert sampled_frame_indices(0) == []
    assert sampled_frame_indices(1) == [0]
    assert sampled_frame_indices(10) == [0]


def test_both_scripts_agree_on_the_sampling_stride():
    assert SAMPLE_STRIDE == EXTRACTOR_STRIDE == 10


def test_sample_seed_boxes_takes_every_tenth_cached_frame():
    track = SimpleNamespace(bbox=[1.0, 2.0, 3.0, 4.0])
    cached = [{1: track, 2: track} if frame % 20 == 0 else {1: track} for frame in range(25)]

    seed = sample_seed_boxes(cached)

    assert sorted(seed) == [0, 10, 20]
    assert seed[0] == [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    assert seed[10] == [[1.0, 2.0, 3.0, 4.0]]


# --- seed JSON round trip ---------------------------------------------------

def test_seed_json_round_trips_between_the_two_scripts(tmp_path):
    path = str(tmp_path / 'clip_1_seed.json')
    seed = {0: [[10.0, 20.0, 40.0, 80.0], [50.0, 20.0, 80.0, 80.0]], 10: [], 20: [[5.5, 6.5, 7.5, 8.5]]}

    write_seed(seed, path)
    loaded = load_seed_boxes(path)

    assert sorted(loaded) == [0, 10, 20]
    assert [b.bbox for b in loaded[0]] == seed[0]
    assert loaded[10] == []
    assert [b.bbox for b in loaded[20]] == seed[20]
    assert all(b.track_id is None for boxes in loaded.values() for b in boxes)


def test_load_seed_boxes_returns_empty_for_a_missing_file(tmp_path):
    assert load_seed_boxes(str(tmp_path / 'absent_seed.json')) == {}


# --- save guards ------------------------------------------------------------

def test_duplicate_ids_on_a_frame_block_the_save_naming_the_frame():
    frame_boxes = {0: [box(4), box(4), box(7)], 10: [box(1), box(2)]}

    errors = frame_validation_errors(frame_boxes)

    assert len(errors) == 1
    assert 'frame 0' in errors[0] and '4' in errors[0]


def test_unassigned_boxes_do_not_count_as_duplicate_ids():
    assert frame_validation_errors({0: [box(None), box(None), box(3)]}) == []


def test_a_degenerate_assigned_box_blocks_the_save_naming_the_frame():
    frame_boxes = {10: [box(5, bbox=[30.0, 20.0, 30.0, 80.0])]}

    errors = frame_validation_errors(frame_boxes)

    assert len(errors) == 1
    assert 'frame 10' in errors[0]


def test_a_degenerate_unassigned_box_does_not_block_because_it_is_never_saved():
    assert frame_validation_errors({0: [box(None, bbox=[30.0, 20.0, 30.0, 80.0]), box(1)]}) == []


def test_out_of_range_box_counts_warn_but_do_not_error():
    frame_boxes = {
        0: [box(i) for i in range(3)],     # too few
        10: [box(i) for i in range(8)],    # in range: silent
        20: [box(i) for i in range(13)],   # too many
        30: [box(None)],                   # unlabelled frame: silent
    }

    warnings = frame_count_warnings(frame_boxes)

    assert len(warnings) == 2
    assert 'frame 0' in warnings[0] and '3' in warnings[0]
    assert 'frame 20' in warnings[1] and '13' in warnings[1]
    assert frame_validation_errors(frame_boxes) == []


# --- progress ---------------------------------------------------------------

def test_progress_counts_a_frame_as_labelled_only_with_an_assigned_box():
    frame_boxes = {0: [box(1)], 10: [box(None)], 20: [], 30: [box(None), box(2)]}

    assert labelled_frame_count(frame_boxes) == 2
    assert progress_line('clip_1', frame_boxes, [0, 10, 20, 30, 40]) == 'clip_1: 2/5 sampled frames labelled'


# --- ground-truth persistence -----------------------------------------------

def test_assigned_boxes_round_trip_through_the_mot_file_and_unassigned_are_dropped(tmp_path):
    path = str(tmp_path / 'clip_1_gt.txt')
    frame_boxes = {
        0: [box(4, bbox=[10.0, 20.0, 40.0, 80.0]), box(None, bbox=[0.0, 0.0, 5.0, 5.0])],
        10: [box(7, bbox=[50.0, 20.0, 80.0, 80.0])],
    }

    save_mot_annotations(frame_boxes_to_annotations(frame_boxes), path)
    reloaded = annotations_to_frame_boxes(load_mot_annotations(path))

    assert sorted(reloaded) == [0, 10]
    assert [(b.track_id, b.bbox) for b in reloaded[0]] == [(4, [10.0, 20.0, 40.0, 80.0])]
    assert [(b.track_id, b.bbox) for b in reloaded[10]] == [(7, [50.0, 20.0, 80.0, 80.0])]


def test_initial_frame_boxes_suppresses_seed_boxes_coinciding_with_saved_gt():
    gt = {0: [box(4)]}
    seed = {0: [box(None), box(None)], 10: [box(None)]}  # frame-0 seeds share the saved box's bbox exactly

    initial = initial_frame_boxes([0, 10, 20], gt, seed)

    assert [b.track_id for b in initial[0]] == [4]      # coinciding seeds suppressed, not duplicated
    assert [b.track_id for b in initial[10]] == [None]  # seed fallback on an unlabelled frame
    assert initial[20] == []                            # blank


def test_startup_merge_restores_unmatched_seed_boxes_alongside_saved_gt():
    saved = [box(1, bbox=[0.0, 0.0, 10.0, 10.0]), box(2, bbox=[50.0, 50.0, 60.0, 60.0])]
    seeds = [
        box(None, bbox=[1.0, 0.0, 11.0, 10.0]),        # IoU 9/11 with saved 1: covered, suppressed
        box(None, bbox=[50.0, 50.0, 60.0, 60.0]),      # exact match with saved 2: suppressed
        box(None, bbox=[100.0, 100.0, 110.0, 110.0]),  # uncovered: restored
        box(None, bbox=[200.0, 0.0, 210.0, 10.0]),     # uncovered: restored
    ]

    initial = initial_frame_boxes([0], {0: saved}, {0: seeds})

    assigned = [b for b in initial[0] if b.track_id is not None]
    unassigned = [b for b in initial[0] if b.track_id is None]
    assert [b.track_id for b in assigned] == [1, 2]
    assert [b.bbox for b in unassigned] == [[100.0, 100.0, 110.0, 110.0], [200.0, 0.0, 210.0, 10.0]]


def test_seed_match_threshold_matches_the_evaluator_convention():
    from evaluation.mot_metrics import MOTEvaluator

    # The tool cannot import MOTEvaluator at runtime (evaluation.mot_metrics
    # pulls the GPU stack the local machine lacks), so the shared 0.5
    # convention is mirrored as a constant and pinned here instead.
    assert lgt.SEED_MATCH_IOU_THRESHOLD == MOTEvaluator.IOU_THRESHOLD


def test_initial_frame_boxes_copies_so_edits_do_not_alias_the_sources():
    seed = {0: [box(None)]}

    initial = initial_frame_boxes([0], {}, seed)
    initial[0][0].track_id = 9

    assert seed[0][0].track_id is None


# --- interactive-session edge cases -------------------------------------------

def test_quit_saves_and_validates_every_clip_even_after_a_refusal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lgt, 'GT_PATH_TEMPLATE', str(tmp_path / '{clip}_gt.txt'))
    bad = make_clip('clip_a', {0: [box(4), box(4)]})   # duplicate IDs: refused
    good = make_clip('clip_b', {0: [box(i) for i in range(1, 7)]})

    session = LabellingSession([bad, good])
    quit_now = session.handle_key(ord('q'))

    assert quit_now is False                            # the refusal still blocks the quit
    out = capsys.readouterr().out
    assert 'clip_a: NOT saved' in out
    # A lazy all() stopped at clip_a and never saved (or validated) clip_b.
    assert (tmp_path / 'clip_b_gt.txt').exists()
    assert load_mot_annotations(str(tmp_path / 'clip_b_gt.txt')) != []


def test_deleting_the_selected_box_clears_the_pending_id():
    target = box(None, bbox=[10.0, 10.0, 50.0, 50.0])
    clip = make_clip('clip_a', {0: [target]})
    session = LabellingSession([clip])
    session.selected = target
    session.id_buffer = '4'
    session.delete_mode = True

    session.on_mouse(cv2.EVENT_LBUTTONDOWN, 20, 20, 0, None)

    assert clip.frame_boxes[0] == []
    assert session.selected is None
    assert session.id_buffer == ''
    assert session.delete_mode is False


def test_entering_delete_mode_drops_a_half_typed_id():
    target = box(None, bbox=[10.0, 10.0, 50.0, 50.0])
    clip = make_clip('clip_a', {0: [target]})
    session = LabellingSession([clip])
    session.selected = target
    session.id_buffer = '12'

    session.handle_key(ord('d'))

    assert session.delete_mode is True
    assert session.selected is None
    assert session.id_buffer == ''


def test_unsampled_gt_frames_flags_rows_outside_the_current_sample():
    gt_boxes = {0: [box(1)], 5: [box(2)], 20: [box(3)], 25: [box(4)]}

    assert unsampled_gt_frames(gt_boxes, [0, 10, 20]) == [5, 25]
    assert unsampled_gt_frames({0: [box(1)]}, [0, 10]) == []


def test_save_refuses_when_the_ground_truth_holds_frames_outside_the_sample(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lgt, 'GT_PATH_TEMPLATE', str(tmp_path / '{clip}_gt.txt'))
    clip = make_clip('clip_a', {0: [box(1)]}, unsampled=[5, 25])

    assert save_clip(clip) is False
    out = capsys.readouterr().out
    assert '[5, 25]' in out and 'stride' in out
    assert not (tmp_path / 'clip_a_gt.txt').exists()   # nothing was dropped by a partial rewrite


def test_delete_removes_the_exact_object_not_its_value_equal_twin():
    twin_1 = box(None, bbox=[10.0, 10.0, 50.0, 50.0])
    twin_2 = box(None, bbox=[10.0, 10.0, 50.0, 50.0])
    assert twin_1 == twin_2 and twin_1 is not twin_2   # the dataclass __eq__ trap

    clip = make_clip('clip_a', {0: [twin_1, twin_2]})
    session = LabellingSession([clip])
    session.selected = twin_2
    session.id_buffer = '7'

    session._delete_box(twin_2)

    # list.remove() would have deleted twin_1 (the first equal box) and left
    # the removed twin_2 selected; identity deletion removes the right one.
    assert len(clip.frame_boxes[0]) == 1
    assert clip.frame_boxes[0][0] is twin_1
    assert session.selected is None
    assert session.id_buffer == ''


def test_delete_click_on_identical_twins_keeps_the_selection_consistent():
    twin_1 = box(None, bbox=[10.0, 10.0, 50.0, 50.0])
    twin_2 = box(None, bbox=[10.0, 10.0, 50.0, 50.0])
    clip = make_clip('clip_a', {0: [twin_1, twin_2]})
    session = LabellingSession([clip])
    session.selected = twin_2
    session.id_buffer = '7'
    session.delete_mode = True

    session.on_mouse(cv2.EVENT_LBUTTONDOWN, 20, 20, 0, None)

    # The click resolves to twin_1; exactly that object is removed, and the
    # untouched selected twin keeps its pending ID.
    assert len(clip.frame_boxes[0]) == 1
    assert clip.frame_boxes[0][0] is twin_2
    assert session.selected is twin_2
    assert session.id_buffer == '7'


def test_save_skips_a_never_labelled_clip_without_writing_a_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lgt, 'GT_PATH_TEMPLATE', str(tmp_path / '{clip}_gt.txt'))
    fresh = make_clip('clip_a', {0: [box(None), box(None)]})   # seed boxes only, nothing ever saved

    assert save_clip(fresh) is True                            # a skip is not a refusal; q may proceed
    assert 'skipped as unlabelled' in capsys.readouterr().out
    assert not (tmp_path / 'clip_a_gt.txt').exists()           # no committed-looking empty ground truth


def test_save_refuses_a_cleared_clip_and_leaves_the_old_file_untouched(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lgt, 'GT_PATH_TEMPLATE', str(tmp_path / '{clip}_gt.txt'))
    existing = tmp_path / 'clip_a_gt.txt'
    existing.write_text('1,4,10,20,30,40,1,-1,-1,-1\n')

    # Loaded with saved GT, then every label removed this sitting: without the
    # had_saved_gt flag this looked identical to a never-labelled clip and the
    # skip reported a clean save while the stale rows waited to reload.
    cleared = make_clip('clip_a', {0: [box(None)]}, had_saved_gt=True)

    assert save_clip(cleared) is False                          # blocks a silent q
    out = capsys.readouterr().out
    assert 'previously had saved labels' in out
    assert str(existing) in out                                 # names the untouched file
    assert existing.read_text() == '1,4,10,20,30,40,1,-1,-1,-1\n'


def test_a_clip_cleared_after_an_in_session_save_refuses_on_the_second_save(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lgt, 'GT_PATH_TEMPLATE', str(tmp_path / '{clip}_gt.txt'))
    clip = make_clip('clip_a', {0: [box(i) for i in range(1, 7)]})   # no gt.txt at load
    assert clip.had_saved_gt is False

    assert save_clip(clip) is True                       # first sitting-save writes the file
    gt_path = tmp_path / 'clip_a_gt.txt'
    written = gt_path.read_text()
    assert written != ''

    clip.frame_boxes[0].clear()                          # every label deleted, same session

    # With load-time-only tracking this took the never-labelled skip, reporting
    # a clean save while the just-written rows survived to reload next sitting.
    assert save_clip(clip) is False
    assert 'previously had saved labels' in capsys.readouterr().out
    assert gt_path.read_text() == written                # the written file is left intact
