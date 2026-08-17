"""Unit tests for the per-frame possession ground-truth labelling tool: ordering, answer parsing, append-only persistence, resume and rendering."""
from __future__ import annotations

import ast
import csv
import importlib
import inspect
import json
import weakref
import zlib
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import basketball.labelling.possession_gt as possession_gt
from basketball.detection.ball_detector import BallDetection
from basketball.detection.player_detector import PlayerTrack
from basketball.labelling.frame_rendering import BOX_COLOURS, draw_tracked_boxes
from basketball.labelling.possession_gt import (
    CLIPS,
    CSV_HEADER,
    INVALID,
    LABEL,
    NOBODY,
    STOP,
    UNCLEAR,
    ClipData,
    append_label,
    build_frame_order,
    default_frame_counts,
    frame_player_ids_for_all_clips,
    frame_player_ids_from_tracks,
    load_clip_data,
    load_existing_labels,
    load_labelled_rows,
    make_clip_loader,
    next_unlabelled_index,
    parse_answer,
    run_session,
)
from basketball.labelling.possession_rendering import (
    BALL_BOX_COLOUR,
    CAPTION_ORIGIN,
    gated_ball_bbox,
    render_labelling_frame,
    to_rgb,
)
from basketball.possession.possession_io import BallInput

matplotlib.use('Agg')  # headless: the label cell's plt.show() must never try to open a GUI window here

NOTEBOOK_PATH = 'scripts/label_possession_gt_per_frame.ipynb'


def track(track_id: int, bbox: tuple[float, float, float, float]) -> PlayerTrack:
    """One tracked player at a fixed bbox."""
    return PlayerTrack(track_id=track_id, bbox=list(bbox), confidence=0.9)


def scripted_prompt(answers: list[str]) -> tuple[Callable[[str], str], list[str]]:
    """A prompt callable that returns the scripted answers in order, plus the list it records the prompts it was shown into."""
    seen: list[str] = []

    def prompt(message: str) -> str:
        seen.append(message)
        return answers.pop(0)

    return prompt, seen


# --- frame ordering -----------------------------------------------------------

def test_build_frame_order_is_chronological_within_each_clip_and_clips_in_fixed_order() -> None:
    order = build_frame_order({'clip_1': 3, 'clip_2': 2, 'clip_3': 1})

    assert order == [
        ('clip_1', 0), ('clip_1', 1), ('clip_1', 2),
        ('clip_2', 0), ('clip_2', 1),
        ('clip_3', 0),
    ]


def test_build_frame_order_covers_every_frame_with_no_sampling_or_shuffling() -> None:
    # The real scope: 534 frames, every frame of all three clips.
    counts = {'clip_1': 117, 'clip_2': 174, 'clip_3': 243}

    order = build_frame_order(counts)

    assert len(order) == 534
    assert len(set(order)) == 534  # no duplicates
    for clip in CLIPS:
        assert [frame for c, frame in order if c == clip] == list(range(counts[clip]))


def test_default_frame_counts_reads_the_real_cached_track_length_per_clip(monkeypatch) -> None:
    # The "never a hardcoded number" guarantee: the 534-frame scope must come
    # from the caches themselves, so that a re-run of detection with different
    # clips cannot silently leave the labelling scope pointing at stale counts.
    # load_cache is stubbed because the caches are not on the local machine.
    requested: list[str] = []
    lengths = {'clip_1': 117, 'clip_2': 174, 'clip_3': 243}

    def fake_load_cache(path: str) -> list[dict[int, PlayerTrack]]:
        requested.append(path)
        clip = next(name for name in lengths if name in path)
        return [{} for _ in range(lengths[clip])]

    monkeypatch.setattr(possession_gt, 'load_cache', fake_load_cache)

    assert default_frame_counts() == lengths
    assert sum(default_frame_counts().values()) == 534
    # Read from the player-detection caches, the same ones the notebook builds
    # its per-frame track dicts from.
    assert requested[:3] == [
        'data/processed/clip_1/player_detections.pkl',
        'data/processed/clip_2/player_detections.pkl',
        'data/processed/clip_3/player_detections.pkl',
    ]


def test_frame_player_ids_from_tracks_maps_each_frame_to_its_visible_track_ids() -> None:
    tracks_by_clip = {
        'clip_1': [{4: track(4, (0, 0, 10, 20))}, {4: track(4, (1, 1, 11, 21)), 7: track(7, (5, 5, 15, 25))}],
    }

    ids = frame_player_ids_from_tracks(tracks_by_clip)

    assert ids[('clip_1', 0)] == {4}
    assert ids[('clip_1', 1)] == {4, 7}


# --- per-clip lazy loading -----------------------------------------------------
#
# 534 frames across all three clips, decoded and held at once, is roughly
# 1.5-3.3GB resident depending on clip resolution -- but only one frame is
# ever displayed at a time. load_clip_data()/make_clip_loader() decode and
# hold at most ONE clip's frames at a time instead; the tests below prove
# both the loading (against fakes, the same load_video() misuse class of
# risk as before, just relocated here) and the eviction (via weakref, since
# measuring process memory directly is noisy and allocator-dependent).

FAKE_FRAME_COUNT = 2
FAKE_FRAME_SHAPE = (50, 60, 3)


def _fake_clip_frame(clip: str, idx: int) -> np.ndarray:
    """A small, real BGR ndarray whose fill value is deterministic per (clip, idx), so two different frames are never accidentally equal."""
    # zlib.crc32, not hash(): str hashing is randomised per PROCESS (a
    # security feature against hash-flooding), so two clips' values could
    # coincidentally collide modulo 256 on an unlucky process -- this
    # actually happened once while writing these tests, intermittently
    # failing test_make_clip_loader_reloads_for_a_different_clip. crc32 is
    # stable across runs, so the same (clip, idx) always gets the same value.
    value = (zlib.crc32(clip.encode()) + idx) % 256
    return np.full(FAKE_FRAME_SHAPE, value, dtype=np.uint8)


def _install_clip_loader_fakes(monkeypatch, frame_count: int = FAKE_FRAME_COUNT) -> None:
    """
    Patch the three real, file/model-backed calls load_clip_data() makes --
    on possession_gt's OWN namespace, not the original source modules.
    possession_gt.py already bound these names at ITS OWN import time via
    from basketball.x.y import z; patching basketball.x.y.z afterwards
    would leave possession_gt's already-bound reference untouched and this
    would silently exercise the real files instead of the fakes.
    """
    def fake_load_video(video_path: str):
        clip = Path(video_path).stem
        for idx in range(frame_count):
            yield idx, _fake_clip_frame(clip, idx)  # a (frame_idx, frame) tuple, exactly like the real load_video()

    def fake_load_cache(_path: str) -> list[dict]:
        return [{} for _ in range(frame_count)]  # empty player-track dicts; most of these tests don't inspect track content

    def fake_possession_ball_input(_raw: list[dict]) -> BallInput:
        empty = [{} for _ in range(frame_count)]
        return BallInput(gated=empty, filled=empty)

    monkeypatch.setattr(possession_gt, 'load_video', fake_load_video)
    monkeypatch.setattr(possession_gt, 'load_cache', fake_load_cache)
    monkeypatch.setattr(possession_gt, 'possession_ball_input', fake_possession_ball_input)


def test_load_clip_data_decodes_every_frame_into_an_indexable_dict(monkeypatch) -> None:
    """Regression test for the original load_video() misuse, retargeted to where the loading logic now actually lives: load_video() yields (frame_idx, frame) tuples, not frames."""
    _install_clip_loader_fakes(monkeypatch)

    data = load_clip_data('clip_1')

    assert isinstance(data, ClipData)
    assert isinstance(data.frames, dict)  # not a bare generator: len() and frames[idx] both need this
    assert len(data.frames) == FAKE_FRAME_COUNT
    for idx in range(FAKE_FRAME_COUNT):
        assert isinstance(data.frames[idx], np.ndarray)
        assert not isinstance(data.frames[idx], tuple)
        assert np.array_equal(data.frames[idx], _fake_clip_frame('clip_1', idx))
    assert len(data.tracks) == FAKE_FRAME_COUNT
    assert len(data.gated) == FAKE_FRAME_COUNT


def test_load_clip_data_uses_the_gated_ball_detections_never_the_filled_ones(monkeypatch) -> None:
    """.gated is what a labeller must be shown; .filled is the interpolated track and is never read here -- see possession_rendering's own contract."""
    def fake_possession_ball_input(_raw: list[dict]) -> BallInput:
        return BallInput(gated=['GATED_SENTINEL'], filled=['FILLED_SENTINEL'])

    _install_clip_loader_fakes(monkeypatch, frame_count=1)
    monkeypatch.setattr(possession_gt, 'possession_ball_input', fake_possession_ball_input)

    data = load_clip_data('clip_1')

    assert data.gated == ['GATED_SENTINEL']


def test_load_clip_data_prints_the_frame_count_and_size_in_memory(monkeypatch, capsys) -> None:
    """The diagnostic the fix's memory-boundedness is meant to be checkable from, per the task: a clear per-clip line naming how many frames and how many MB just entered memory."""
    _install_clip_loader_fakes(monkeypatch, frame_count=3)

    load_clip_data('clip_2')

    out = capsys.readouterr().out
    assert 'clip_2' in out
    assert '3 frames' in out
    assert 'MB' in out


def test_make_clip_loader_reuses_the_same_data_for_the_same_clip(monkeypatch) -> None:
    _install_clip_loader_fakes(monkeypatch)
    clip_loader = make_clip_loader()

    first = clip_loader('clip_1')
    second = clip_loader('clip_1')

    assert first is second  # no reload for a repeat request of the same clip


def test_make_clip_loader_reloads_for_a_different_clip(monkeypatch) -> None:
    _install_clip_loader_fakes(monkeypatch)
    clip_loader = make_clip_loader()

    clip_1_data = clip_loader('clip_1')
    clip_2_data = clip_loader('clip_2')

    assert clip_1_data is not clip_2_data
    assert not np.array_equal(clip_1_data.frames[0], clip_2_data.frames[0])


def test_make_clip_loader_discards_the_previous_clips_frames_on_a_different_clip(monkeypatch) -> None:
    """
    The memory fix's core guarantee. A weakref to clip_1's first frame array
    is used rather than measuring process memory (noisy, allocator- and
    OS-dependent): if nothing else holds a reference, the weakref dies the
    instant CPython's refcounting collects the object, which happens
    deterministically as soon as make_clip_loader()'s own internal reference
    is reassigned to a different clip -- proving eviction, not just that the
    same clip is never re-decoded.
    """
    _install_clip_loader_fakes(monkeypatch)
    clip_loader = make_clip_loader()

    first_frame = clip_loader('clip_1').frames[0]
    watch = weakref.ref(first_frame)
    del first_frame  # this test's own reference must not be what keeps it alive

    assert watch() is not None  # still resident while clip_1 remains current

    clip_loader('clip_2')  # a different clip -- clip_1's data must now be discarded

    assert watch() is None, "clip_1's frame array was still referenced after moving to clip_2"


def test_make_clip_loader_never_holds_more_than_one_clips_frames_across_a_full_pass(monkeypatch) -> None:
    """Same guarantee, checked across a full clip_1 -> clip_2 -> clip_3 pass (the real labelling scope) rather than one transition: at every point, at most the CURRENT clip's frames are reachable."""
    _install_clip_loader_fakes(monkeypatch)
    clip_loader = make_clip_loader()

    watches: list[weakref.ReferenceType] = []
    for clip in CLIPS:
        first_frame = clip_loader(clip).frames[0]
        watches.append(weakref.ref(first_frame))
        del first_frame

        for earlier_watch in watches[:-1]:
            assert earlier_watch() is None, f'an earlier clip was still resident once {clip} was loaded'
        assert watches[-1]() is not None

    assert watches[-1]() is not None  # the last clip loaded stays resident afterwards


def test_frame_player_ids_for_all_clips_matches_the_underlying_helper_called_on_every_clip_at_once(monkeypatch) -> None:
    tracks_by_clip = {
        'clip_1': [{4: track(4, (0, 0, 10, 20))}],
        'clip_2': [{7: track(7, (0, 0, 10, 20))}, {}],
    }

    def fake_load_cache(path: str) -> list[dict]:
        clip = next(name for name in tracks_by_clip if name in path)
        return tracks_by_clip[clip]

    monkeypatch.setattr(possession_gt, 'load_cache', fake_load_cache)

    result = frame_player_ids_for_all_clips(clips=('clip_1', 'clip_2'))

    assert result == frame_player_ids_from_tracks(tracks_by_clip)


def test_frame_player_ids_for_all_clips_loads_one_clips_tracks_at_a_time(monkeypatch) -> None:
    """clip_tracks is reassigned, not accumulated, each iteration: load_cache is called exactly once per clip, in clip order, never with more than one clip's list reachable through a persisted structure."""
    calls: list[str] = []

    def fake_load_cache(path: str) -> list[dict]:
        calls.append(path)
        return [{}]

    monkeypatch.setattr(possession_gt, 'load_cache', fake_load_cache)

    frame_player_ids_for_all_clips(clips=('clip_1', 'clip_2', 'clip_3'))

    assert calls == [
        'data/processed/clip_1/player_detections.pkl',
        'data/processed/clip_2/player_detections.pkl',
        'data/processed/clip_3/player_detections.pkl',
    ]


# --- answer parsing -----------------------------------------------------------

def test_parse_answer_accepts_a_visible_track_id_as_a_string_holder() -> None:
    parsed = parse_answer('7', {4, 7})

    assert parsed.outcome == LABEL
    # One column, one type: the id is stringified rather than stored as an int
    # alongside the 'nobody'/'unclear' sentinels.
    assert parsed.value == '7'
    assert isinstance(parsed.value, str)


@pytest.mark.parametrize(
    ('answer', 'expected'),
    [('n', NOBODY), ('u', UNCLEAR), ('N', NOBODY), (' u ', UNCLEAR)],
)
def test_parse_answer_accepts_the_sentinel_keys_case_insensitively(answer: str, expected: str) -> None:
    parsed = parse_answer(answer, {4})

    assert parsed.outcome == LABEL
    assert parsed.value == expected


@pytest.mark.parametrize('answer', ['s', 'S', ' s '])
def test_parse_answer_recognises_the_stop_key(answer: str) -> None:
    assert parse_answer(answer, {4}).outcome == STOP


@pytest.mark.parametrize('answer', ['', 'x', 'nobody', '3.5', '?', 'n7'])
def test_parse_answer_rejects_an_unrecognised_answer(answer: str) -> None:
    parsed = parse_answer(answer, {4})

    assert parsed.outcome == INVALID
    assert 'Unrecognised answer' in parsed.value


def test_parse_answer_rejects_a_track_id_not_present_in_this_frame_and_names_the_valid_ids() -> None:
    parsed = parse_answer('9', {4, 7})

    assert parsed.outcome == INVALID
    assert '9' in parsed.value
    assert '[4, 7]' in parsed.value  # the valid ids, sorted


# --- CSV persistence ----------------------------------------------------------

def test_append_label_writes_a_header_then_one_row_per_label(tmp_path) -> None:
    path = str(tmp_path / 'possession_gt.csv')

    append_label('clip_1', 0, '7', path=path, labelled_at='t0')
    append_label('clip_1', 1, NOBODY, path=path, labelled_at='t1')

    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER
    assert rows[1] == ['clip_1', '0', '7', 't0']
    assert rows[2] == ['clip_1', '1', 'nobody', 't1']


def test_append_label_writes_the_header_when_the_file_exists_but_is_zero_bytes(tmp_path) -> None:
    # A crash between open and the first write leaves an empty file behind.
    # Checking existence alone would make the first real label the header,
    # silently dropping it and corrupting every later read.
    path = tmp_path / 'possession_gt.csv'
    path.touch()
    assert path.stat().st_size == 0

    append_label('clip_1', 0, '7', path=str(path), labelled_at='t0')

    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER
    assert rows[1] == ['clip_1', '0', '7', 't0']
    assert load_labelled_rows(str(path))[0]['holder'] == '7'


@pytest.mark.parametrize('holder', [7, None, '', 'somebody', '-3', '7.0'])
def test_append_label_refuses_a_holder_that_breaks_the_one_type_contract(tmp_path, holder: object) -> None:
    # The holder column is one string type end to end: a digit string, or one
    # of the two sentinels. Enforced at the write boundary so a caller passing
    # an int (which csv would silently stringify) or None (which csv would
    # write as an empty field) cannot put a value on disk that every reader
    # then has to guess the type of.
    path = tmp_path / 'possession_gt.csv'

    with pytest.raises(ValueError, match='holder must be'):
        append_label('clip_1', 0, holder, path=str(path), labelled_at='t0')

    # And nothing is written, so a rejected write cannot leave a header-only
    # or part-written file behind.
    assert not path.exists()


@pytest.mark.parametrize('holder', ['7', '0', '134', NOBODY, UNCLEAR])
def test_append_label_accepts_a_track_id_and_both_sentinels(tmp_path, holder: str) -> None:
    path = str(tmp_path / 'possession_gt.csv')

    append_label('clip_1', 0, holder, path=path, labelled_at='t0')

    assert load_labelled_rows(path)[0]['holder'] == holder


@pytest.mark.parametrize('answer', ['7', '134', 'n', 'u'])
def test_every_holder_parse_answer_accepts_is_writable(tmp_path, answer: str) -> None:
    # The two validation boundaries must agree: run_session() feeds
    # parse_answer's value straight to append_label, so anything the parser
    # accepts and the writer then refuses would abort a labelling session on a
    # legitimate answer.
    path = str(tmp_path / 'possession_gt.csv')
    parsed = parse_answer(answer, {7, 134})
    assert parsed.outcome == LABEL

    append_label('clip_1', 0, parsed.value, path=path, labelled_at='t0')

    assert load_labelled_rows(path)[0]['holder'] == parsed.value


def test_append_label_creates_the_parent_directory(tmp_path) -> None:
    path = str(tmp_path / 'nested' / 'annotations' / 'possession_gt.csv')

    append_label('clip_1', 0, '7', path=path, labelled_at='t0')

    assert Path(path).exists()


def test_load_labelled_rows_returns_empty_for_a_missing_file(tmp_path) -> None:
    assert load_labelled_rows(str(tmp_path / 'absent.csv')) == []


def test_load_labelled_rows_keeps_the_last_row_for_a_corrected_frame(tmp_path) -> None:
    # Writes stay append-only for crash safety, so a correction is a NEW row.
    # Last-wins dedup lives in this one function, which every consumer reads
    # through, so the correction supersedes its earlier row in one place.
    path = str(tmp_path / 'possession_gt.csv')
    append_label('clip_1', 0, '7', path=path, labelled_at='t0')
    append_label('clip_1', 1, NOBODY, path=path, labelled_at='t1')
    append_label('clip_1', 0, '4', path=path, labelled_at='t2')  # the correction

    rows = load_labelled_rows(path)

    assert len(rows) == 2
    assert {(row['clip'], row['frame_idx']): row['holder'] for row in rows} == {
        ('clip_1', '0'): '4',
        ('clip_1', '1'): 'nobody',
    }
    # Order is first-appearance (labelling order), so a correction does not
    # jump its frame to the end of the file's logical order.
    assert [row['frame_idx'] for row in rows] == ['0', '1']


def test_load_existing_labels_dedups_a_corrected_frame_to_one_entry(tmp_path) -> None:
    path = str(tmp_path / 'possession_gt.csv')
    append_label('clip_1', 0, '7', path=path, labelled_at='t0')
    append_label('clip_1', 0, '4', path=path, labelled_at='t1')

    assert load_existing_labels(path) == {('clip_1', 0)}


# --- resume -------------------------------------------------------------------

def test_next_unlabelled_index_returns_the_first_gap_not_the_next_one_forward() -> None:
    # A forward-only cursor starting from the session's current position would
    # skip frame 1 forever once frame 2 was labelled; the full scan finds it.
    order = [('clip_1', 0), ('clip_1', 1), ('clip_1', 2)]

    assert next_unlabelled_index(order, {('clip_1', 0), ('clip_1', 2)}) == 1


def test_next_unlabelled_index_returns_none_when_every_frame_is_labelled() -> None:
    # None rather than len(order): callers cannot then index past the end.
    order = [('clip_1', 0), ('clip_1', 1)]

    assert next_unlabelled_index(order, {('clip_1', 0), ('clip_1', 1)}) is None


def test_next_unlabelled_index_starts_at_zero_for_an_empty_label_set() -> None:
    assert next_unlabelled_index([('clip_1', 0)], set()) == 0


# --- the session loop ---------------------------------------------------------

def test_run_session_labels_frames_in_order_and_stops_on_request(tmp_path) -> None:
    path = str(tmp_path / 'possession_gt.csv')
    order = [('clip_1', 0), ('clip_1', 1), ('clip_1', 2)]
    ids = {key: {4, 7} for key in order}
    shown: list[tuple[str, int]] = []
    prompt, _seen = scripted_prompt(['7', 'n', 's'])

    written = run_session(order, ids, lambda clip, frame: shown.append((clip, frame)), prompt, path=path)

    assert written == 2
    assert shown == [('clip_1', 0), ('clip_1', 1), ('clip_1', 2)]
    assert [(row['frame_idx'], row['holder']) for row in load_labelled_rows(path)] == [('0', '7'), ('1', 'nobody')]


def test_run_session_re_presents_the_same_frame_after_an_invalid_answer_without_writing(tmp_path) -> None:
    path = str(tmp_path / 'possession_gt.csv')
    order = [('clip_1', 0), ('clip_1', 1)]
    ids = {key: {4} for key in order}
    shown: list[tuple[str, int]] = []
    # 'x' is unrecognised, '9' is a track id not in the frame, then a valid 4.
    prompt, _seen = scripted_prompt(['x', '9', '4', 's'])

    written = run_session(order, ids, lambda clip, frame: shown.append((clip, frame)), prompt, path=path)

    assert written == 1
    # Frame 0 shown three times (two rejections), then frame 1.
    assert shown == [('clip_1', 0), ('clip_1', 0), ('clip_1', 0), ('clip_1', 1)]
    assert [row['frame_idx'] for row in load_labelled_rows(path)] == ['0']


def test_run_session_resumes_at_the_first_unlabelled_frame(tmp_path) -> None:
    path = str(tmp_path / 'possession_gt.csv')
    order = [('clip_1', 0), ('clip_1', 1), ('clip_1', 2)]
    ids = {key: {4} for key in order}
    append_label('clip_1', 0, '4', path=path, labelled_at='t0')
    append_label('clip_1', 2, NOBODY, path=path, labelled_at='t1')
    shown: list[tuple[str, int]] = []
    prompt, _seen = scripted_prompt(['4', 's'])

    run_session(order, ids, lambda clip, frame: shown.append((clip, frame)), prompt, path=path)

    # Frame 1 is the gap; frame 2 is already labelled and is not re-presented.
    assert shown == [('clip_1', 1)]


def test_run_session_reports_completion_without_indexing_past_the_end(tmp_path, capsys) -> None:
    path = str(tmp_path / 'possession_gt.csv')
    order = [('clip_1', 0)]
    append_label('clip_1', 0, '4', path=path, labelled_at='t0')

    def must_not_be_called(clip: str, frame_idx: int) -> None:
        raise AssertionError('no frame should be shown when everything is labelled')

    def must_not_prompt(message: str) -> str:
        raise AssertionError('no prompt should be issued when everything is labelled')

    written = run_session(order, {('clip_1', 0): {4}}, must_not_be_called, must_not_prompt, path=path)

    assert written == 0
    assert 'All 1 frames are labelled' in capsys.readouterr().out


def test_run_session_flushes_each_label_before_the_next_prompt(tmp_path) -> None:
    # Crash safety: a kernel death must lose at most the frame in progress, so
    # the row must already be on disk while the NEXT frame is being answered.
    path = str(tmp_path / 'possession_gt.csv')
    order = [('clip_1', 0), ('clip_1', 1)]
    ids = {key: {4} for key in order}
    on_disk_during_second_prompt: list[int] = []

    answers = ['4', 's']

    def prompt(message: str) -> str:
        on_disk_during_second_prompt.append(len(load_labelled_rows(path)))
        return answers.pop(0)

    run_session(order, ids, lambda clip, frame: None, prompt, path=path)

    # Nothing written before the first answer, frame 0's row already on disk
    # by the time the second prompt is issued.
    assert on_disk_during_second_prompt == [0, 1]


# --- rendering ----------------------------------------------------------------

def frame_with_players() -> np.ndarray:
    """A mid-grey frame big enough for boxes and captions to be drawn inside."""
    return np.full((200, 300, 3), 128, dtype=np.uint8)


def test_gated_ball_bbox_returns_the_detection_bbox_or_none() -> None:
    assert gated_ball_bbox({1: BallDetection(bbox=[1, 2, 3, 4], confidence=0.8)}) == [1, 2, 3, 4]
    assert gated_ball_bbox({}) is None


def test_render_labelling_frame_draws_the_ball_box_in_a_colour_no_player_box_uses() -> None:
    frame = frame_with_players()
    tracks = {4: track(4, (10, 60, 50, 160)), 7: track(7, (100, 60, 140, 160))}
    gated = {1: BallDetection(bbox=[200, 100, 220, 120], confidence=0.9)}

    annotated = render_labelling_frame(frame, tracks, gated)

    # The ball's colour is not in the player palette, so however the per-frame
    # rank assignment falls out, the ball can never be confused with a player.
    assert BALL_BOX_COLOUR not in BOX_COLOURS
    # Checked ON the rectangle's own edge rather than anywhere in the image,
    # so removing the box (and leaving only its white caption) fails this.
    assert tuple(int(c) for c in annotated[100, 210]) == BALL_BOX_COLOUR  # top edge
    assert tuple(int(c) for c in annotated[120, 210]) == BALL_BOX_COLOUR  # bottom edge


def test_render_labelling_frame_captions_a_frame_with_no_gated_detection() -> None:
    frame = frame_with_players()
    tracks = {4: track(4, (10, 60, 50, 160))}

    without_ball = render_labelling_frame(frame, tracks, {})

    # Compared against the player boxes ALONE: the no-detection caption is the
    # only thing this render may add, so if the caption were not drawn the two
    # would be identical. Asserting merely "differs from the bare frame" would
    # pass on the player boxes alone and prove nothing about the caption.
    players_only = draw_tracked_boxes(frame, tracks)
    assert not np.array_equal(without_ball, players_only)
    # And the caption is drawn in the caption band, in the ball's own colour.
    band = without_ball[:CAPTION_ORIGIN[1] + 1]
    assert any(
        tuple(int(c) for c in pixel) == BALL_BOX_COLOUR for row in band for pixel in row
    )


def test_render_labelling_frame_never_captions_no_detection_when_one_exists() -> None:
    # The two branches are mutually exclusive: a frame with a real detection
    # must not also claim there was none.
    frame = frame_with_players()
    tracks = {4: track(4, (10, 60, 50, 160))}

    with_ball = render_labelling_frame(frame, tracks, {1: BallDetection(bbox=[200, 100, 220, 120], confidence=0.9)})
    without_ball = render_labelling_frame(frame, tracks, {})

    caption_band = slice(0, CAPTION_ORIGIN[1] + 1)
    assert not np.array_equal(with_ball[caption_band], without_ball[caption_band])


def test_render_labelling_frame_does_not_mutate_the_input_frame() -> None:
    frame = frame_with_players()
    before = frame.copy()

    render_labelling_frame(frame, {4: track(4, (10, 60, 50, 160))}, {})

    assert np.array_equal(frame, before)


def test_to_rgb_swaps_the_blue_and_red_channels() -> None:
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # pure blue in BGR

    rgb = to_rgb(bgr)

    assert tuple(rgb[0, 0]) == (0, 0, 255)


# --- notebook wiring ----------------------------------------------------------

def notebook_source() -> str:
    """Every source line of the labelling notebook, markdown prose included."""
    notebook = json.loads(Path(NOTEBOOK_PATH).read_text(encoding='utf-8'))
    return '\n'.join(''.join(cell['source']) for cell in notebook['cells'])


def notebook_code() -> str:
    """Only the notebook's executable code, with markdown cells and comment lines stripped."""
    # Absence checks must read code, not prose: the notebook deliberately
    # DISCUSSES ipywidgets, .filled and fill_missing() in its markdown and
    # comments to record why it avoids each, and a substring match over the
    # whole document would read those explanations as the thing itself.
    notebook = json.loads(Path(NOTEBOOK_PATH).read_text(encoding='utf-8'))
    return '\n'.join(
        line
        for cell in notebook['cells'] if cell['cell_type'] == 'code'
        for line in ''.join(cell['source']).splitlines()
        if not line.strip().startswith('#')
    )


def test_the_notebook_shows_only_gated_detections_never_interpolated_ones() -> None:
    # The contamination risk this whole tool is designed around: showing an
    # interpolated ball would lead the labeller to a fabricated answer.
    code = notebook_code()

    assert '.gated' in code
    assert '.filled' not in code


def test_the_notebook_uses_matplotlib_and_input_not_ipywidgets() -> None:
    # ipywidgets does not render on this project's JupyterHub, so a widget UI
    # would be unusable there however correct it looked locally.
    code = notebook_code()

    assert '%matplotlib inline' in code
    assert 'prompt=input' in code
    assert 'ipywidgets' not in code
    assert 'widgets.' not in code
    # The reason is recorded in the notebook's own prose, not only in code.
    assert 'jupyterlab_widgets' in notebook_source()


def test_the_notebook_builds_its_ball_input_through_the_shared_helper() -> None:
    # The call itself now lives in possession_gt.load_clip_data() (moved
    # there so the notebook's own load cell could become a one-line
    # make_clip_loader() call -- see the per-clip lazy loading tests below),
    # so this is checked on load_clip_data()'s own source rather than the
    # notebook's, mirroring the check that used to live here.
    #
    # Comments are stripped before the checks: load_clip_data()'s own
    # comment ALSO names possession_ball_input() to explain why it is
    # called, and a substring match over the raw source (inspect.getsource()
    # includes comments) would read that explanation as the call itself --
    # caught by mutation-testing a version that bypassed the real call but
    # left the comment untouched, which this used to pass.
    source = inspect.getsource(possession_gt.load_clip_data)
    code = '\n'.join(line for line in source.splitlines() if not line.strip().startswith('#'))

    assert 'possession_ball_input(' in code
    assert 'fill_missing(' not in code
    assert 'filter_detections_global_trajectory(' not in code  # the helper owns the ordering


def test_every_name_the_notebook_imports_actually_exists() -> None:
    # This test suite carries no real video files or caches of its own (this
    # notebook's data lives in data/raw/ and data/processed/, neither
    # committed), so a renamed or mistyped import would otherwise only
    # surface at label time. Resolving the imports here is checkable offline
    # regardless of what data happens to exist on whichever machine runs it.
    tree = ast.parse(notebook_code().replace('%matplotlib inline', ''))
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith(('basketball', 'evaluation')):
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            assert hasattr(module, alias.name), f'{node.module} has no {alias.name}'
            checked += 1
    assert checked > 0  # the walk found imports at all, rather than vacuously passing


def test_the_frame_order_and_the_visible_track_ids_cover_exactly_the_same_frames() -> None:
    # run_session() looks up frame_player_ids[(clip, frame_idx)] for every
    # frame in the order, so a mismatch between the two would KeyError
    # mid-session. Both derive from the same cached track list length, and
    # this pins that they stay consistent.
    tracks_by_clip = {
        'clip_1': [{4: track(4, (0, 0, 10, 20))} for _ in range(3)],
        'clip_2': [{7: track(7, (0, 0, 10, 20))} for _ in range(2)],
        'clip_3': [{9: track(9, (0, 0, 10, 20))} for _ in range(1)],
    }
    frame_counts = {clip: len(frames) for clip, frames in tracks_by_clip.items()}

    order = build_frame_order(frame_counts)
    ids = frame_player_ids_from_tracks(tracks_by_clip)

    assert set(order) == set(ids)
    for key in order:
        assert ids[key]  # every frame has at least one selectable track id


# --- notebook cell EXECUTION against fakes -------------------------------------
#
# Every test above either checks substrings/imports or hand-builds namespace
# data directly, never running the notebook's own cell bodies. That gap is
# exactly how the load cell's ORIGINAL load_video() misuse (assigning its
# generator straight to frames_by_clip[clip], then len()-ing and indexing it)
# shipped despite this file being "wiring tested". The tests below close it:
# they exec the notebook's actual load, frame-order and label cell source
# (still via json, matching this file's own convention, not nbformat)
# against the SAME fakes the per-clip loading tests above use, reusing
# _install_clip_loader_fakes() since the notebook's load cell now just calls
# make_clip_loader() -- the loading logic itself is exercised directly by
# the tests above; these confirm the notebook's own cells still wire it up
# correctly.

def _notebook_code_cells() -> list[str]:
    """Every code cell's raw source from the notebook, in cell order, unstripped (unlike notebook_code()) so cells can be exec'd individually."""
    notebook = json.loads(Path(NOTEBOOK_PATH).read_text(encoding='utf-8'))
    return [
        ''.join(cell['source']).replace('%matplotlib inline', '')
        for cell in notebook['cells'] if cell['cell_type'] == 'code'
    ]


def test_the_load_cell_creates_a_working_per_clip_lazy_loader(monkeypatch) -> None:
    """
    Regression test for the ORIGINAL load_video() misuse, retargeted: that
    bug lived in a cell that decoded every clip's frames eagerly into
    frames_by_clip, which the memory fix below replaced with a one-line
    make_clip_loader() call. This pins that the cell's own source does that
    correctly against small fakes -- not, for instance, forgetting to call
    make_clip_loader() or naming the result something the label cell never
    looks up -- and that the returned loader hands back a real indexable
    per-frame dict, not a bare generator, for whichever clip is asked for.
    """
    _install_clip_loader_fakes(monkeypatch)
    cells = _notebook_code_cells()
    namespace: dict = {'__name__': '__main__'}
    exec(compile(cells[1], '<imports>', 'exec'), namespace)  # the real imports cell
    exec(compile(cells[2], '<load>', 'exec'), namespace)     # the cell under test

    assert callable(namespace['clip_loader'])
    data = namespace['clip_loader']('clip_1')
    assert isinstance(data.frames, dict)  # not a bare generator: len() and frames[idx] both need this
    assert len(data.frames) == FAKE_FRAME_COUNT
    for idx in range(FAKE_FRAME_COUNT):
        assert isinstance(data.frames[idx], np.ndarray)
        assert not isinstance(data.frames[idx], tuple)
        assert np.array_equal(data.frames[idx], _fake_clip_frame('clip_1', idx))


def test_show_passes_the_real_per_frame_array_into_the_renderer_not_a_load_video_tuple(monkeypatch, tmp_path) -> None:
    """
    Companion regression test at the point closest to the bug's user-visible
    symptom: show() must hand render_labelling_frame() the actual per-frame
    ndarray via clip_loader(clip), never the raw (frame_idx, frame) tuple
    load_video() actually yields. Runs the load, frame-order and label
    cells' own source against fakes, with render_labelling_frame spied (not
    replaced) so a wrong value still exercises the real rendering path and
    fails there if it would fail for a real labeller. CSV_PATH is
    overridden to a scratch file and input scripted to stop immediately, so
    this can never write into the real
    data/annotations/possession_gt_per_frame.csv.
    """
    _install_clip_loader_fakes(monkeypatch)
    cells = _notebook_code_cells()
    namespace: dict = {'__name__': '__main__'}
    exec(compile(cells[1], '<imports>', 'exec'), namespace)

    captured_frames: list[object] = []
    real_render = namespace['render_labelling_frame']

    def spy_render(frame: object, tracks: dict, gated_frame: dict) -> np.ndarray:
        captured_frames.append(frame)
        return real_render(frame, tracks, gated_frame)  # calls through: a tuple fails here too, inside draw_tracked_boxes' frame.copy()

    namespace['render_labelling_frame'] = spy_render
    namespace['CSV_PATH'] = str(tmp_path / 'scratch.csv')
    namespace['input'] = lambda _message: 's'  # stop immediately after the first frame is shown; never a real session

    exec(compile(cells[2], '<load>', 'exec'), namespace)
    exec(compile(cells[3], '<frame_order>', 'exec'), namespace)
    exec(compile(cells[4], '<label>', 'exec'), namespace)

    assert len(captured_frames) == 1  # show() was called exactly once, for frame_order[0]
    frame = captured_frames[0]
    assert isinstance(frame, np.ndarray), f'show() passed a {type(frame).__name__}, not a real frame array'
    assert not isinstance(frame, tuple)
    assert np.array_equal(frame, namespace['clip_loader']('clip_1').frames[0])
    # The scripted 's' answers every prompt, so run_session() never reaches
    # append_label() -- the scratch file must not even have been created,
    # confirming this test wrote nothing anywhere, real CSV_PATH included.
    assert not Path(namespace['CSV_PATH']).exists()


def test_running_the_label_cell_across_a_clip_boundary_evicts_the_earlier_clips_frames(monkeypatch, tmp_path) -> None:
    """
    End-to-end proof of the memory fix through the notebook's OWN show()
    wiring, not just make_clip_loader() called directly (see the per-clip
    loading tests above for that): frame_order/frame_player_ids are
    overridden to exactly one frame per clip, so answering clip_1's single
    frame and stopping as soon as clip_2's is shown drives run_session()
    straight across the clip_1 -> clip_2 boundary. By then, clip_1's frame
    array -- watched from before the label cell even ran -- must be
    unreachable.
    """
    _install_clip_loader_fakes(monkeypatch, frame_count=1)
    cells = _notebook_code_cells()
    namespace: dict = {'__name__': '__main__'}
    exec(compile(cells[1], '<imports>', 'exec'), namespace)
    exec(compile(cells[2], '<load>', 'exec'), namespace)

    clip_1_frame = namespace['clip_loader']('clip_1').frames[0]
    watch = weakref.ref(clip_1_frame)
    del clip_1_frame

    namespace['frame_order'] = [('clip_1', 0), ('clip_2', 0), ('clip_3', 0)]
    namespace['frame_player_ids'] = {key: set() for key in namespace['frame_order']}
    namespace['CSV_PATH'] = str(tmp_path / 'scratch.csv')
    answers = iter(['n', 's'])  # label clip_1's one frame, then stop once clip_2's is shown
    namespace['input'] = lambda _message: next(answers)

    exec(compile(cells[4], '<label>', 'exec'), namespace)

    assert watch() is None, "clip_1's frame array was still referenced after the label cell moved past it"
