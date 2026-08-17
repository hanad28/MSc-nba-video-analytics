"""Unit tests for the court-keypoint identification audit tool and its notebook wiring."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import basketball.labelling.keypoint_audit as audit
from basketball.keypoints.court_keypoints import Keypoint
from basketball.labelling.keypoint_audit import (
    CORRECT,
    NOT_ON_COURT,
    UNCLEAR,
    WRONG_LANDMARK,
    append_verdict,
    build_verdict_order,
    confident_indices,
    eligible_frames,
    keypoint_confidence,
    load_existing_verdicts,
    load_labelled_rows,
    next_unverdicted_index,
    parse_actual_index,
    parse_verdict,
    run_session,
    sample_frames,
)

NOTEBOOK_PATH = 'scripts/audit_keypoints.ipynb'


def frame_with(indices: list[int], confidence: float = 0.9) -> list[Keypoint]:
    """One frame's 18 keypoints, with only the named indices above the confidence threshold."""
    return [
        Keypoint(
            index=i, x=float(i * 10), y=float(i * 5),
            confidence=confidence if i in indices else 0.1,
        )
        for i in range(18)
    ]


def clip_of(per_frame: list[list[int]]) -> list[list[Keypoint]]:
    """A clip's worth of frames, each carrying the confident indices given for it."""
    return [frame_with(indices) for indices in per_frame]


# --- sampling -----------------------------------------------------------------

def test_sampling_returns_eight_eight_and_nine():
    eligible = {clip: list(range(50)) for clip in ('clip_1', 'clip_2', 'clip_3')}

    sampled = sample_frames(eligible)

    assert [len(sampled[clip]) for clip in ('clip_1', 'clip_2', 'clip_3')] == [8, 8, 9]
    assert sum(len(frames) for frames in sampled.values()) == 25


def test_sampling_is_reproducible_under_the_fixed_seed():
    eligible = {clip: list(range(50)) for clip in ('clip_1', 'clip_2', 'clip_3')}

    assert sample_frames(eligible) == sample_frames(eligible)


def test_sampling_differs_under_a_different_seed():
    # Confirms the seed is actually driving the draw rather than the result
    # being fixed by construction.
    eligible = {clip: list(range(50)) for clip in ('clip_1', 'clip_2', 'clip_3')}

    assert sample_frames(eligible, seed=1) != sample_frames(eligible, seed=2)


def test_a_frame_with_one_confident_keypoint_is_eligible():
    # The rule that matters: the lead motivating this audit was a clip_3 frame
    # with a SINGLE confident keypoint placed wrong at confidence 1.0.
    # Restricting to well-populated frames would design that case out.
    keypoints = clip_of([[], [3], [0, 5, 8, 9], []])

    assert eligible_frames(keypoints) == [1, 2]


def test_frames_with_no_confident_keypoint_are_not_eligible():
    assert eligible_frames(clip_of([[], []])) == []


def test_sampling_draws_from_sparse_frames_not_only_well_populated_ones():
    # Every eligible frame here is sparse (one or two confident keypoints), so
    # a rule that required four or more would sample nothing at all.
    sparse = clip_of([[3]] * 20)
    eligible = eligible_frames(sparse)

    sampled = sample_frames({'clip_1': eligible, 'clip_2': eligible, 'clip_3': eligible})

    assert len(sampled['clip_1']) == 8
    assert all(frame_idx in eligible for frame_idx in sampled['clip_1'])


def test_a_clip_with_fewer_eligible_frames_than_its_quota_contributes_all_of_them():
    sampled = sample_frames({'clip_1': [2, 7], 'clip_2': list(range(20)), 'clip_3': list(range(20))})

    assert sampled['clip_1'] == [2, 7]


# --- presentation order -------------------------------------------------------

def test_presentation_order_is_shuffled_not_clip_chronological():
    # Matches the team GT tool rather than the possession tool: a keypoint's
    # correctness is judged from the single frame, so shuffling costs no
    # context and stops a verdict propagating across near-identical neighbours.
    sampled = {clip: list(range(8)) for clip in ('clip_1', 'clip_2', 'clip_3')}
    keypoints = {clip: clip_of([[0, 5]] * 8) for clip in ('clip_1', 'clip_2', 'clip_3')}

    order = build_verdict_order(sampled, keypoints)
    frames = [(clip, frame_idx) for clip, frame_idx, _ in order]
    chronological = [(clip, idx) for clip in sorted(sampled) for idx in sampled[clip]]

    # Compared as SEQUENCES: dict.fromkeys compares by key set, so it would
    # report these equal however they were ordered.
    presented = list(dict.fromkeys(frames))

    assert presented != chronological
    assert sorted(presented) == sorted(chronological)


def test_every_keypoint_on_a_frame_is_consecutive_so_one_render_serves_them_all():
    sampled = {'clip_1': [0, 1], 'clip_2': [], 'clip_3': []}
    keypoints = {'clip_1': clip_of([[0, 5, 8], [2, 9]]), 'clip_2': [], 'clip_3': []}

    order = build_verdict_order(sampled, keypoints)
    frames = [(clip, frame_idx) for clip, frame_idx, _ in order]

    # Each frame appears as one unbroken run, so show() is called once per frame.
    assert len(list(dict.fromkeys(frames))) == len(set(frames))


def test_the_order_covers_only_confident_keypoints():
    sampled = {'clip_1': [0], 'clip_2': [], 'clip_3': []}
    keypoints = {'clip_1': clip_of([[4, 11]]), 'clip_2': [], 'clip_3': []}

    assert build_verdict_order(sampled, keypoints) == [('clip_1', 0, 4), ('clip_1', 0, 11)]


def test_confident_indices_uses_the_threshold():
    frame = [
        Keypoint(index=0, x=1.0, y=1.0, confidence=audit.KEYPOINT_CONFIDENCE_THRESHOLD),
        Keypoint(index=1, x=1.0, y=1.0, confidence=audit.KEYPOINT_CONFIDENCE_THRESHOLD - 0.01),
    ]

    assert confident_indices(frame) == [0]


# --- verdict parsing ----------------------------------------------------------

@pytest.mark.parametrize('answer,expected', [
    ('c', CORRECT), ('w', WRONG_LANDMARK), ('n', NOT_ON_COURT), ('u', UNCLEAR),
    ('C', CORRECT), (' w ', WRONG_LANDMARK),
])
def test_the_primary_verdicts_are_accepted(answer, expected):
    parsed = parse_verdict(answer)

    assert (parsed.outcome, parsed.value) == (audit.LABEL, expected)


def test_stop_is_recognised():
    assert parse_verdict('s').outcome == audit.STOP


@pytest.mark.parametrize('answer', ['', 'x', 'correct', '3', 'cw'])
def test_an_unrecognised_verdict_is_rejected_naming_the_valid_answers(answer):
    parsed = parse_verdict(answer)

    assert parsed.outcome == audit.INVALID
    assert 'correct' in parsed.value and 'stop' in parsed.value


@pytest.mark.parametrize('answer,expected', [('0', '0'), ('17', '17'), (' 9 ', '9'), ('?', '?')])
def test_the_follow_up_accepts_an_index_or_cannot_tell(answer, expected):
    parsed = parse_actual_index(answer)

    assert (parsed.outcome, parsed.value) == (audit.LABEL, expected)


@pytest.mark.parametrize('answer', ['18', '-1', '--3', '3-', '+3', 'x', '', '1.5'])
def test_the_follow_up_rejects_an_out_of_range_index(answer):
    # '--3' is the crash case: lstrip('-') left '3', which passed isdigit(),
    # and int('--3') then aborted the whole session rather than re-prompting.
    assert parse_actual_index(answer).outcome == audit.INVALID


# --- persistence --------------------------------------------------------------

def test_append_writes_a_header_then_one_row(tmp_path):
    path = str(tmp_path / 'audit.csv')

    append_verdict('clip_1', 3, 5, CORRECT, '', 0.91, path=path)

    with open(path, newline='') as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == audit.CSV_HEADER
    assert rows[1][:6] == ['clip_1', '3', '5', CORRECT, '', '0.910000']


def test_a_zero_byte_file_still_gets_its_header(tmp_path):
    # A crash between open and the first write leaves an empty file; checking
    # existence alone would make the first real verdict the header.
    path = tmp_path / 'audit.csv'
    path.touch()

    append_verdict('clip_1', 0, 1, CORRECT, '', 0.8, path=str(path))

    with open(path, newline='') as handle:
        assert list(csv.reader(handle))[0] == audit.CSV_HEADER


def test_appending_never_rewrites_earlier_rows(tmp_path):
    path = str(tmp_path / 'audit.csv')

    append_verdict('clip_1', 0, 1, CORRECT, '', 0.8, path=path)
    append_verdict('clip_1', 0, 2, NOT_ON_COURT, '', 0.7, path=path)

    with open(path, newline='') as handle:
        assert len(list(csv.reader(handle))) == 3


def test_a_re_verdicted_keypoint_supersedes_its_earlier_row(tmp_path):
    path = str(tmp_path / 'audit.csv')

    append_verdict('clip_1', 0, 1, CORRECT, '', 0.8, path=path)
    append_verdict('clip_1', 0, 1, WRONG_LANDMARK, '6', 0.8, path=path)

    rows = load_labelled_rows(path)

    assert len(rows) == 1
    assert rows[0]['verdict'] == WRONG_LANDMARK
    assert rows[0]['actual_index'] == '6'


def test_dedup_is_keyed_on_all_three_parts(tmp_path):
    # Two different keypoints on the same frame are distinct rows, not one
    # superseding the other -- the keypoint index is part of the key.
    path = str(tmp_path / 'audit.csv')

    append_verdict('clip_1', 0, 1, CORRECT, '', 0.8, path=path)
    append_verdict('clip_1', 0, 2, CORRECT, '', 0.8, path=path)
    append_verdict('clip_2', 0, 1, CORRECT, '', 0.8, path=path)

    assert len(load_labelled_rows(path)) == 3


def test_actual_index_must_be_empty_for_a_non_wrong_verdict(tmp_path):
    path = str(tmp_path / 'audit.csv')

    with pytest.raises(ValueError, match='must be empty'):
        append_verdict('clip_1', 0, 1, CORRECT, '6', 0.8, path=path)


def test_an_unknown_verdict_string_is_refused_at_the_write_boundary(tmp_path):
    path = str(tmp_path / 'audit.csv')

    with pytest.raises(ValueError, match='verdict must be one of'):
        append_verdict('clip_1', 0, 1, 'maybe', '', 0.8, path=path)


def test_loading_a_missing_file_returns_no_rows(tmp_path):
    assert load_labelled_rows(str(tmp_path / 'absent.csv')) == []


# --- resume -------------------------------------------------------------------

def test_resume_returns_the_first_unverdicted_key():
    order = [('clip_1', 0, 1), ('clip_1', 0, 2), ('clip_2', 5, 3)]

    assert next_unverdicted_index(order, {('clip_1', 0, 1)}) == 1


def test_resume_does_not_skip_a_corrected_earlier_keypoint():
    # A forward-only cursor would pass over a hole left by a correction and
    # never come back to it.
    order = [('clip_1', 0, 1), ('clip_1', 0, 2), ('clip_2', 5, 3)]

    assert next_unverdicted_index(order, {('clip_1', 0, 2), ('clip_2', 5, 3)}) == 0


def test_resume_returns_none_when_everything_is_verdicted():
    order = [('clip_1', 0, 1)]

    assert next_unverdicted_index(order, {('clip_1', 0, 1)}) is None


# --- the session loop ---------------------------------------------------------

def test_a_session_records_a_verdict_with_its_confidence(tmp_path):
    path = str(tmp_path / 'audit.csv')
    keypoints = {'clip_1': clip_of([[4]])}
    keypoints['clip_1'][0][4] = Keypoint(index=4, x=1.0, y=2.0, confidence=0.77)
    answers = iter(['c', 's'])

    written = run_session(
        [('clip_1', 0, 4)], keypoints, show=lambda clip, idx, keypoint_index: None,
        prompt=lambda message: next(answers), path=path,
    )

    rows = load_labelled_rows(path)
    assert written == 1
    assert rows[0]['verdict'] == CORRECT
    assert float(rows[0]['confidence']) == pytest.approx(0.77)


def test_the_follow_up_is_asked_only_on_a_wrong_verdict(tmp_path):
    path = str(tmp_path / 'audit.csv')
    keypoints = {'clip_1': clip_of([[4, 5]])}
    asked: list[str] = []

    def prompt(message: str) -> str:
        asked.append(message)
        # Keypoint 4 is correct, keypoint 5 is wrong and sits on index 6.
        return {1: 'c', 2: 'w', 3: '6'}.get(len(asked), 's')

    run_session(
        [('clip_1', 0, 4), ('clip_1', 0, 5)], keypoints,
        show=lambda clip, idx, keypoint_index: None, prompt=prompt, path=path,
    )

    follow_ups = [message for message in asked if 'actually sit on' in message]
    assert len(follow_ups) == 1
    rows = {int(row['keypoint_index']): row for row in load_labelled_rows(path)}
    assert rows[4]['actual_index'] == ''
    assert rows[5]['actual_index'] == '6'


def test_an_invalid_verdict_writes_nothing_and_re_presents(tmp_path):
    path = str(tmp_path / 'audit.csv')
    keypoints = {'clip_1': clip_of([[4]])}
    answers = iter(['x', 'c', 's'])

    written = run_session(
        [('clip_1', 0, 4)], keypoints, show=lambda clip, idx, keypoint_index: None,
        prompt=lambda message: next(answers), path=path,
    )

    assert written == 1
    assert len(load_labelled_rows(path)) == 1


def test_an_invalid_follow_up_writes_nothing_and_re_presents(tmp_path):
    path = str(tmp_path / 'audit.csv')
    keypoints = {'clip_1': clip_of([[4]])}
    answers = iter(['w', '99', 'w', '6', 's'])

    run_session(
        [('clip_1', 0, 4)], keypoints, show=lambda clip, idx, keypoint_index: None,
        prompt=lambda message: next(answers), path=path,
    )

    rows = load_labelled_rows(path)
    assert len(rows) == 1
    assert rows[0]['actual_index'] == '6'


def test_each_verdict_renders_with_its_own_keypoint_highlighted(tmp_path):
    # Re-rendered per verdict rather than once per frame: the point under
    # judgement must be highlighted, and hunting for the prompted index among
    # identical dots invites verdicting the wrong point -- invisible
    # afterwards in ground truth that scores K3 and K5.
    path = str(tmp_path / 'audit.csv')
    keypoints = {'clip_1': clip_of([[4, 5, 8]])}
    rendered: list[tuple[str, int, int]] = []
    answers = iter(['c', 'c', 'c', 's'])

    run_session(
        [('clip_1', 0, 4), ('clip_1', 0, 5), ('clip_1', 0, 8)], keypoints,
        show=lambda clip, idx, keypoint_index: rendered.append((clip, idx, keypoint_index)),
        prompt=lambda message: next(answers), path=path,
    )

    assert rendered == [('clip_1', 0, 4), ('clip_1', 0, 5), ('clip_1', 0, 8)]


def test_stopping_leaves_the_remaining_keypoints_unverdicted(tmp_path):
    path = str(tmp_path / 'audit.csv')
    keypoints = {'clip_1': clip_of([[4, 5]])}
    answers = iter(['c', 's'])

    written = run_session(
        [('clip_1', 0, 4), ('clip_1', 0, 5)], keypoints, show=lambda clip, idx, keypoint_index: None,
        prompt=lambda message: next(answers), path=path,
    )

    assert written == 1
    assert load_existing_verdicts(path) == {('clip_1', 0, 4)}


def test_keypoint_confidence_reads_the_detectors_value():
    frame = frame_with([3])

    assert keypoint_confidence(frame, 3) == pytest.approx(0.9)
    assert keypoint_confidence(frame, 99) == 0.0


def test_detection_runs_over_the_whole_clip_not_only_the_sampled_frames(monkeypatch, tmp_path):
    # The fingerprint includes n_frames, so passing only the sampled frames can
    # never validate against a full-clip cache: the stage would re-infer and
    # then OVERWRITE the production cache with a handful of entries.
    frames = {idx: np.full((4, 6, 3), idx, dtype=np.uint8) for idx in range(20)}
    monkeypatch.setattr(audit, 'load_video', lambda path: list(frames.items()))
    monkeypatch.setattr(audit, 'CourtKeypoints', _StubDetector)
    _StubDetector.calls.clear()

    audit.load_clip_data('clip_1', [3, 11])

    assert len(_StubDetector.calls) == 1
    call = _StubDetector.calls[0]
    assert len(call['frames']) == 20, 'detection must see the whole clip'
    assert call['cache_path'] == audit.KEYPOINTS_CACHE_TEMPLATE.format(clip='clip_1')


def test_sampled_keypoints_are_indexed_absolutely_not_positionally(monkeypatch, tmp_path):
    # The sampled frames are a scattered subset, so positional indexing would
    # pair a frame's image with a different frame's keypoints.
    frames = {idx: np.full((4, 6, 3), idx, dtype=np.uint8) for idx in range(20)}
    monkeypatch.setattr(audit, 'load_video', lambda path: list(frames.items()))

    class _PerFrameDetector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_detection(self, **kwargs: object) -> list[list[Keypoint]]:
            # Frame k's keypoint 4 carries confidence encoding k, so a
            # misaligned pairing is visible in the value.
            return [
                [
                    Keypoint(index=i, x=0.0, y=0.0, confidence=0.5 + k / 100 if i == 4 else 0.1)
                    for i in range(18)
                ]
                for k in range(len(kwargs['frames']))
            ]

    monkeypatch.setattr(audit, 'CourtKeypoints', _PerFrameDetector)

    data = audit.load_clip_data('clip_1', [3, 11])

    assert set(data.frames) == {3, 11}
    assert keypoint_confidence(data.keypoints[3], 4) == pytest.approx(0.53)
    assert keypoint_confidence(data.keypoints[11], 4) == pytest.approx(0.61)


def test_a_keypoint_frame_count_mismatch_raises_naming_both_counts(monkeypatch, tmp_path):
    frames = {idx: np.zeros((4, 6, 3), dtype=np.uint8) for idx in range(20)}
    monkeypatch.setattr(audit, 'load_video', lambda path: list(frames.items()))

    class _ShortDetector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_detection(self, **kwargs: object) -> list[list[Keypoint]]:
            return [frame_with([4])] * 5

    monkeypatch.setattr(audit, 'CourtKeypoints', _ShortDetector)

    with pytest.raises(ValueError, match='5 keypoint frames for 20 video frames'):
        audit.load_clip_data('clip_1', [3, 11])


def test_a_session_never_rewrites_the_production_keypoint_cache(monkeypatch, tmp_path):
    # The artefact measure_court_keypoints.py produced must survive an audit
    # session byte-identical; overwriting it would also break this tool's own
    # resume on the next run.
    cache_path = tmp_path / 'keypoints.pkl'
    cache_path.write_bytes(b'measurement cache contents')
    before = cache_path.read_bytes()

    frames = {idx: np.zeros((4, 6, 3), dtype=np.uint8) for idx in range(20)}
    monkeypatch.setattr(audit, 'load_video', lambda path: list(frames.items()))
    monkeypatch.setattr(audit, 'KEYPOINTS_CACHE_TEMPLATE', str(cache_path))
    monkeypatch.setattr(audit, 'CourtKeypoints', _StubDetector)
    _StubDetector.calls.clear()

    audit.load_clip_data('clip_1', [3, 11])

    assert cache_path.read_bytes() == before

# --- notebook wiring ----------------------------------------------------------

def notebook_cells() -> list[str]:
    """Every code cell's source from the audit notebook, in order."""
    notebook = json.loads(Path(NOTEBOOK_PATH).read_text(encoding='utf-8'))
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']


def notebook_code() -> str:
    """Only the notebook's executable code, with markdown cells and comment lines stripped."""
    # Absence checks must read code, not prose: the notebook deliberately
    # discusses ipywidgets in its markdown to record why it avoids it, and a
    # substring match over the whole document would read that as the thing.
    return '\n'.join(
        line for cell in notebook_cells() for line in cell.splitlines()
        if not line.strip().startswith('#')
    )


def test_the_notebook_uses_matplotlib_and_input_not_ipywidgets():
    code = notebook_code()

    assert '%matplotlib inline' in code
    assert 'prompt=input' in code
    assert 'ipywidgets' not in code


def test_the_notebook_loads_keypoints_through_run_detection_not_load_cache():
    # Verdicts attached to a silently stale cache would be worse than no
    # verdicts, and load_cache skips the fingerprint check entirely.
    code = notebook_code()

    assert 'run_detection(' in code
    assert 'load_cache(' not in code


def test_the_notebook_prints_the_index_reference():
    # Verdicting index 13 requires knowing what 13 means; expecting the
    # labeller to hold 18 definitions in memory is a source of error.
    code = notebook_code()

    assert 'KEYPOINT_NAMES' in code


def test_the_notebook_load_cell_yields_an_indexable_frame_mapping(monkeypatch):
    # The load_video() misuse that has bitten this project before: assigning
    # the generator straight to a mapping leaves each element a (idx, frame)
    # tuple, which fails later inside the renderer rather than here.
    # A contiguous clip, as load_video really yields: every index from 0.
    frames = {idx: np.full((4, 6, 3), idx, dtype=np.uint8) for idx in range(5)}
    monkeypatch.setattr(
        audit, 'load_video', lambda path: [(idx, frame) for idx, frame in frames.items()],
    )
    monkeypatch.setattr(audit, 'CourtKeypoints', _StubDetector)
    _StubDetector.calls.clear()

    data = audit.load_clip_data('clip_1', [1, 3])

    assert isinstance(data.frames[1], np.ndarray)
    assert not isinstance(data.frames[1], tuple)
    assert np.array_equal(data.frames[1], frames[1])


def test_the_notebook_show_cell_passes_a_real_array_into_the_renderer(monkeypatch, tmp_path):
    # Companion regression at the point closest to the symptom: show() must
    # hand the annotator the per-frame ndarray, never load_video()'s tuple.
    frames = {0: np.zeros((8, 12, 3), dtype=np.uint8)}
    keypoints = {'clip_1': clip_of([[4]])}
    received: list[object] = []

    class _SpyAnnotator:
        def __init__(self, render_threshold: float | None = None) -> None:
            pass

        def draw(self, frame_list: list, keypoint_list: list) -> list:
            received.append(frame_list[0])
            return [frame_list[0]]

    namespace = {
        'clip_loader': lambda clip: audit.ClipData(frames=frames, keypoints=keypoints['clip_1']),
        'KeypointAnnotator': _SpyAnnotator,
        'KEYPOINT_CONFIDENCE_THRESHOLD': 0.5,
        'KEYPOINT_NAMES': {index: f'landmark {index}' for index in range(18)},
        'cv2': cv2,
        'plt': _StubPlt(),
    }
    show_cell = next(cell for cell in notebook_cells() if 'def show(' in cell)
    exec(compile(show_cell.split('written = run_session')[0], '<show>', 'exec'), namespace)

    namespace['show']('clip_1', 0, 4)

    assert isinstance(received[0], np.ndarray)
    assert not isinstance(received[0], tuple)


def test_the_notebook_highlights_the_judged_keypoint_distinctly():
    # The judged point must not be mistakable for its neighbours, which stay
    # in KeypointAnnotator's yellow for context.
    code = notebook_code()

    assert 'JUDGED_COLOUR' in code
    assert 'draw_judged' in code


def test_the_notebook_renders_a_zoomed_inset():
    code = notebook_code()

    assert 'inset_crop' in code
    assert 'INSET_HALF_SIZE' in code


def test_the_notebook_inset_crop_is_clamped_at_every_frame_edge():
    # A keypoint at an edge must still yield a full-size window rather than an
    # empty or truncated one -- edge keypoints are exactly those K3 expects to
    # localise worst.
    namespace: dict = {'cv2': cv2, 'plt': _StubPlt()}
    show_cell = next(cell for cell in notebook_cells() if 'def show(' in cell)
    exec(compile(show_cell.split('def show(')[0], '<helpers>', 'exec'), namespace)

    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    half = namespace['INSET_HALF_SIZE']
    for x, y in ((0, 0), (599, 399), (5, 395), (300, 200)):
        crop = namespace['inset_crop'](frame, Keypoint(index=0, x=float(x), y=float(y), confidence=0.9))
        assert crop.shape[:2] == (2 * half, 2 * half), f'edge point ({x}, {y}) gave {crop.shape}'


def test_the_notebook_sets_the_working_directory_before_using_any_path():
    # The notebook lives in scripts/, so a relative path would resolve against
    # scripts/ and load_video would raise OSError: Cannot open video.
    cells = notebook_cells()

    assert 'os.chdir' in cells[0]
    assert cells[0].index('os.chdir') < cells[0].index('import matplotlib')

# --- helpers ------------------------------------------------------------------

class _StubDetector:
    """Stands in for CourtKeypoints, returning one entry per supplied frame and recording how it was called."""

    # One entry PER SUPPLIED FRAME, not a fixed count: a stub that cannot
    # reproduce the caller's real shape cannot catch a shape bug, and a
    # fixed 10 is exactly why the whole-clip/sampled-frames mismatch and the
    # positional indexing were both invisible to these tests.
    calls: list[dict] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def run_detection(self, **kwargs: object) -> list[list[Keypoint]]:
        _StubDetector.calls.append(kwargs)
        return [frame_with([4]) for _ in kwargs['frames']]


class _StubAxis:
    """One absorbed matplotlib axis, recording the image it was asked to draw."""

    def __init__(self) -> None:
        self.images: list = []

    def imshow(self, image: object, *args: object, **kwargs: object) -> None:
        self.images.append(image)

    def axis(self, *args: object, **kwargs: object) -> None:
        pass

    def set_title(self, *args: object, **kwargs: object) -> None:
        pass


class _StubPlt:
    """Absorbs the notebook's matplotlib calls so the show cell can run headless."""

    def __init__(self) -> None:
        self.axes = (_StubAxis(), _StubAxis())

    def subplots(self, *args: object, **kwargs: object) -> tuple:
        return (None, self.axes)

    def tight_layout(self, *args: object, **kwargs: object) -> None:
        pass

    def figure(self, *args: object, **kwargs: object) -> None:
        pass

    def imshow(self, *args: object, **kwargs: object) -> None:
        pass

    def axis(self, *args: object, **kwargs: object) -> None:
        pass

    def title(self, *args: object, **kwargs: object) -> None:
        pass

    def show(self, *args: object, **kwargs: object) -> None:
        pass
