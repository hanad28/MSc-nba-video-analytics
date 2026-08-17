"""
Offline exercise of scripts/label_team_gt_per_frame.ipynb's UI-cell wiring,
without a real ipywidgets/IPython/Jupyter kernel.
"""
# A minimal stand-in for ipywidgets' Button/Image/Label/Box classes is
# injected into sys.modules, the notebook's cells are read via nbformat and
# exec()'d against small stubbed frame/track data, and simulated button
# clicks exercise the actual callback code the notebook would run. This is
# not "unit testing the ipywidgets rendering" (nothing here renders to a
# real widget or a browser); it is exercising the plain-Python control flow
# the notebook wires those widgets to, in particular the fully-labelled
# state that used to raise IndexError.
#
# nbformat and ipywidgets are pinned in requirements.txt. This module still
# skips entirely if nbformat is not importable, a defensive fallback for
# an environment that hasn't reinstalled requirements since the pin was
# added, not because either dependency is meant to be optional, so this
# check degrades gracefully rather than failing the whole suite there.

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from basketball.labelling.team_gt_sampling import load_labelled_rows

nbformat = pytest.importorskip('nbformat')

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'label_team_gt_per_frame.ipynb'


class _FakeWidget:
    """Minimal stand-in for an ipywidgets widget: keeps whatever kwargs it's given and records click handlers."""

    def __init__(self, children: list | None = None, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.children = kwargs.get('children', children or [])
        self._click_handlers: list[Callable[['_FakeWidget'], None]] = []

    def on_click(self, handler: Callable[['_FakeWidget'], None]) -> None:
        self._click_handlers.append(handler)

    def click(self) -> None:
        for handler in self._click_handlers:
            handler(self)


@pytest.fixture
def stub_ipywidgets(monkeypatch):
    """Install fake ipywidgets/IPython.display modules for the duration of one test."""
    ipywidgets_stub = types.ModuleType('ipywidgets')
    for name in ('Image', 'Label', 'VBox', 'HBox', 'Button'):
        setattr(ipywidgets_stub, name, type(name, (_FakeWidget,), {}))
    ipywidgets_stub.Layout = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, 'ipywidgets', ipywidgets_stub)

    ipython_display_stub = types.ModuleType('IPython.display')
    ipython_display_stub.display = lambda *args, **kwargs: None
    ipython_module = types.ModuleType('IPython')
    ipython_module.display = ipython_display_stub
    monkeypatch.setitem(sys.modules, 'IPython', ipython_module)
    monkeypatch.setitem(sys.modules, 'IPython.display', ipython_display_stub)


def _track(bbox: tuple[float, ...]) -> types.SimpleNamespace:
    return types.SimpleNamespace(bbox=list(bbox))


def _code_cells() -> list[str]:
    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    return [cell.source for cell in nb.cells if cell.cell_type == 'code']


def _run_imports_cell(namespace: dict) -> None:
    code_cells = _code_cells()
    # code_cells[0] is the os.chdir setup cell -- skipped, tests already run
    # from the repo root via pytest's rootdir. code_cells[1] is the imports
    # cell, run for real against the actual (torch-free) labelling module.
    exec(compile(code_cells[1], '<imports>', 'exec'), namespace)


def _run_ui_cell(namespace: dict) -> None:
    code_cells = _code_cells()
    # code_cells[5] is the UI cell: 0=setup, 1=imports, 2=sampling print,
    # 3=tracks/frames load, 4=shuffled order/resume, 5=UI, 6=audit.
    exec(compile(code_cells[5], '<ui>', 'exec'), namespace)


def _base_namespace(tmp_path, position: int) -> dict:
    namespace: dict = {'__name__': '__main__'}
    _run_imports_cell(namespace)

    namespace['CLIPS'] = ('clip_1',)
    namespace['sample_frames'] = {'clip_1': [0, 1]}
    namespace['tracks_by_clip'] = {
        'clip_1': [
            {1: _track((10, 10, 40, 60)), 2: _track((60, 10, 90, 60))},
            {1: _track((12, 10, 42, 60))},
        ]
    }
    namespace['frames_by_clip'] = {
        'clip_1': {
            0: np.zeros((100, 100, 3), dtype=np.uint8),
            1: np.zeros((100, 100, 3), dtype=np.uint8),
        }
    }
    namespace['CSV_PATH'] = str(tmp_path / 'labels.csv')

    exec(
        compile(
            'shuffled_order = build_shuffled_order(sample_frames)\n'
            'frame_player_ids = frame_player_ids_from_tracks(sample_frames, tracks_by_clip)\n'
            'labelled = load_existing_labels(CSV_PATH)\n',
            '<resume>',
            'exec',
        ),
        namespace,
    )
    namespace['position'] = position
    return namespace


def _multi_frame_namespace(tmp_path, num_frames: int, position: int) -> dict:
    """Like _base_namespace, but with num_frames sampled frames (each two players: 1, 2), for tests needing more than two items in shuffled_order."""
    namespace: dict = {'__name__': '__main__'}
    _run_imports_cell(namespace)

    namespace['CLIPS'] = ('clip_1',)
    namespace['sample_frames'] = {'clip_1': list(range(num_frames))}
    namespace['tracks_by_clip'] = {
        'clip_1': [{1: _track((10, 10, 40, 60)), 2: _track((60, 10, 90, 60))} for _ in range(num_frames)]
    }
    namespace['frames_by_clip'] = {
        'clip_1': {idx: np.zeros((100, 100, 3), dtype=np.uint8) for idx in range(num_frames)}
    }
    namespace['CSV_PATH'] = str(tmp_path / 'labels.csv')

    exec(
        compile(
            'shuffled_order = build_shuffled_order(sample_frames)\n'
            'frame_player_ids = frame_player_ids_from_tracks(sample_frames, tracks_by_clip)\n'
            'labelled = load_existing_labels(CSV_PATH)\n',
            '<resume>',
            'exec',
        ),
        namespace,
    )
    namespace['position'] = position
    return namespace


def test_ui_cell_renders_normally_when_items_remain(stub_ipywidgets, tmp_path):
    namespace = _base_namespace(tmp_path, position=0)

    _run_ui_cell(namespace)  # must not raise

    assert namespace['player_buttons']  # at least one player row was built
    assert b'' != namespace['main_image'].value


def test_ui_cell_does_not_raise_when_every_sampled_frame_is_fully_labelled(stub_ipywidgets, tmp_path):
    """
    Regression test: resume_index() (and on_next_clicked() advancing past
    the last unlabelled item) can legitimately return
    position == len(shuffled_order) once nothing remains. The UI cell used
    to index shuffled_order[position] unconditionally in render(), raising
    IndexError in exactly this state.
    """
    namespace = _base_namespace(tmp_path, position=2)  # == len(shuffled_order): nothing left

    _run_ui_cell(namespace)  # must not raise IndexError

    assert namespace['player_buttons'] == {}
    assert 'fully labelled' in namespace['status_label'].value.lower()
    assert 'all' in namespace['progress_label'].value.lower()
    assert 'Frame' not in namespace['progress_label'].value  # not "Frame N+1 of N" nonsense


def test_next_button_reaches_the_fully_labelled_state_without_raising(stub_ipywidgets, tmp_path):
    """End-to-end: label every visible player on every sampled frame via simulated clicks, confirm no crash."""
    namespace = _base_namespace(tmp_path, position=0)
    _run_ui_cell(namespace)

    while True:
        clip, frame_idx = namespace['shuffled_order'][namespace['position']]
        for player_id, buttons in list(namespace['player_buttons'].items()):
            if (clip, frame_idx, player_id) not in namespace['labelled']:
                buttons['unclear'].click()
        if namespace['position'] >= len(namespace['shuffled_order']) - 1:
            break
        namespace['next_button'].click()

    # One more click while sitting on the last (now fully labelled) item:
    # on_next_clicked's own guard should report status, not crash, and a
    # manual render() (as startup would call after a resume_index() that
    # returns len(shuffled_order)) must also survive.
    namespace['next_button'].click()
    namespace['position'] = len(namespace['shuffled_order'])
    namespace['render']()


def test_next_button_shows_completion_after_the_last_item_not_a_repeated_frame(stub_ipywidgets, tmp_path):
    """
    Regression test: on_next_clicked() used to clamp the advanced position to
    len(shuffled_order) - 1 (a valid index), so clicking Next Frame from the
    second-to-last item after labelling the last one landed back on that
    already-labelled last frame instead of reaching the fully-labelled
    completion state render() already shows at startup.
    """
    namespace = _base_namespace(tmp_path, position=0)
    _run_ui_cell(namespace)
    assert len(namespace['shuffled_order']) == 2

    # Label every player on the first frame, then advance -- resume_index()
    # skip-ahead should land exactly on the second (last) frame.
    first_clip, first_frame = namespace['shuffled_order'][0]
    for player_id in list(namespace['player_buttons']):
        namespace['player_buttons'][player_id]['unclear'].click()
    namespace['next_button'].click()
    assert namespace['position'] == 1
    assert namespace['status_label'].value == f'{namespace["shuffled_order"][1][0]} frame {namespace["shuffled_order"][1][1]}'

    # Label every player on the second (last) frame, then advance again --
    # nothing remains, so this must reach the completion state, not clamp
    # back onto the just-labelled last frame.
    for player_id in list(namespace['player_buttons']):
        namespace['player_buttons'][player_id]['unclear'].click()
    namespace['next_button'].click()

    assert namespace['position'] == len(namespace['shuffled_order'])
    assert 'fully labelled' in namespace['status_label'].value.lower()
    assert namespace['status_label'].value != f'{first_clip} frame {first_frame}'
    assert 'Frame' not in namespace['progress_label'].value  # not "Frame 2 of 2" -- a repeated-frame symptom
    assert namespace['player_buttons'] == {}  # render()'s completion branch clears the player rows


def test_previous_button_revisits_an_earlier_frame_for_a_correction(stub_ipywidgets, tmp_path):
    """
    Recovery path for a mis-click: label a frame, advance past it with Next,
    click Previous to land back on that exact frame (not "the previous
    UNLABELLED item" the way Next skips to the next unlabelled one -- the
    labeller may specifically want to revisit an already-labelled frame),
    confirm the existing label is shown pre-selected via highlight_selected,
    then click a different team button and confirm load_labelled_rows()
    resolves to the corrected value for that (clip, frame, player) key.
    """
    namespace = _base_namespace(tmp_path, position=0)
    _run_ui_cell(namespace)
    first_clip, first_frame = namespace['shuffled_order'][0]

    # Label player 1 on the first frame as team '1'.
    namespace['player_buttons'][1]['1'].click()
    assert (first_clip, first_frame, 1) in namespace['labelled']

    # Advance past it.
    namespace['next_button'].click()
    assert namespace['position'] == 1

    # Previous must land back on the first frame exactly (position 0), not
    # skip further looking for an unlabelled item.
    namespace['prev_button'].click()
    assert namespace['position'] == 0
    assert namespace['status_label'].value == f'{first_clip} frame {first_frame}'

    # render() must have pre-highlighted player 1's existing '1' label.
    assert namespace['player_buttons'][1]['1'].button_style == 'success'
    assert namespace['player_buttons'][1]['2'].button_style == ''

    # Click a different team -- the correction goes through the same
    # append_label() -> load_labelled_rows() dedup path as any other click.
    namespace['player_buttons'][1]['2'].click()

    rows = load_labelled_rows(namespace['CSV_PATH'])
    matching = [
        row for row in rows
        if row['clip'] == first_clip and int(row['frame_idx']) == first_frame and int(row['player_id']) == 1
    ]
    assert len(matching) == 1
    assert matching[0]['true_team'] == '2'


def test_previous_button_does_not_go_below_position_zero(stub_ipywidgets, tmp_path):
    namespace = _base_namespace(tmp_path, position=0)
    _run_ui_cell(namespace)

    namespace['prev_button'].click()

    assert namespace['position'] == 0


def test_next_button_does_not_declare_completion_when_the_last_frame_is_partially_labelled(stub_ipywidgets, tmp_path):
    """
    Regression test for the exact reported bug: on_next_clicked() used to
    check completion via shuffled_order[position + 1:], a slice that is
    always empty once position is the last index -- so clicking Next while
    sitting on the last frame with unlabelled players still on it used to
    declare "all fully labelled" instead of correctly identifying that the
    current (last) frame itself is still incomplete.
    """
    namespace = _multi_frame_namespace(tmp_path, num_frames=3, position=0)
    _run_ui_cell(namespace)
    last_index = len(namespace['shuffled_order']) - 1
    last_clip, last_frame = namespace['shuffled_order'][last_index]

    # Fully label every frame except the last one.
    for idx in range(last_index):
        namespace['position'] = idx
        namespace['render']()
        for player_id in list(namespace['player_buttons']):
            namespace['player_buttons'][player_id]['unclear'].click()

    # Land on the last frame, leave only ONE of its two players labelled.
    namespace['position'] = last_index
    namespace['render']()
    namespace['player_buttons'][1]['unclear'].click()

    namespace['next_button'].click()

    # Must NOT declare completion -- player 2 on the last frame is still
    # unlabelled, so the (correctly recomputed) position must stay on the
    # last frame, not jump to len(shuffled_order).
    assert namespace['position'] == last_index
    assert namespace['status_label'].value == f'{last_clip} frame {last_frame}'
    assert 'fully labelled' not in namespace['status_label'].value.lower()
    assert namespace['player_buttons']  # still showing the incomplete frame's players


def test_next_button_never_falsely_declares_completion_while_an_earlier_frame_revisited_via_previous_is_incomplete(
    stub_ipywidgets, tmp_path
):
    """
    Regression test: Previous makes navigation non-monotonic, so a
    forward-only completion check (the old shuffled_order[position + 1:]
    slice) cannot see an incomplete frame at or before the current
    position. Repeatedly clicking Next while an earlier frame remains
    incomplete must keep landing back on that earlier frame -- never
    advance past it and never falsely declare every frame labelled.
    """
    namespace = _multi_frame_namespace(tmp_path, num_frames=3, position=0)
    _run_ui_cell(namespace)

    # Fully label the first frame, advance to the second.
    for player_id in list(namespace['player_buttons']):
        namespace['player_buttons'][player_id]['unclear'].click()
    namespace['next_button'].click()
    assert namespace['position'] == 1

    # Use Previous to go back to the (already-complete) first frame --
    # exercising non-monotonic navigation -- then return to the second
    # frame and leave it only partially labelled (one of its two players).
    namespace['prev_button'].click()
    assert namespace['position'] == 0
    namespace['next_button'].click()
    assert namespace['position'] == 1
    namespace['player_buttons'][1]['unclear'].click()  # player 2 left unlabelled

    # Click Next repeatedly: the third (fully unlabelled) frame is later in
    # shuffled_order, but the second frame's remaining gap must keep being
    # found every time, never advancing past it or reaching completion.
    for _ in range(3):
        namespace['next_button'].click()
        assert namespace['position'] == 1
        assert namespace['position'] != len(namespace['shuffled_order'])
        assert 'fully labelled' not in namespace['status_label'].value.lower()


def test_load_cell_and_render_receive_a_real_frame_array_not_a_load_video_tuple(stub_ipywidgets, monkeypatch, tmp_path) -> None:
    """
    Companion check to the analogous regression test added on the sibling
    possession notebook (tests/test_possession_gt_labelling.py), after an
    equivalent load_video() misuse shipped there unnoticed: every other test
    in this file hand-builds frames_by_clip/tracks_by_clip directly (see
    _base_namespace above) and never runs THIS notebook's own load cell
    (code_cells[3]) against load_video -- the same gap that let the
    possession notebook's bug through despite it also being "wiring
    tested". This notebook's load cell already uses the correct
    {idx: frame for idx, frame in load_video(path) if idx in ...} form, so
    this is expected to pass; it exists so a future regression here would be
    caught the same way, rather than only by a human running it for real.
    """
    def fake_load_video(video_path: str):
        for idx in range(3):
            yield idx, np.full((20, 30, 3), idx + 1, dtype=np.uint8)

    def fake_load_cache(_path: str) -> list[dict]:
        return [
            {1: _track((10, 10, 40, 60))},
            {},
            {1: _track((12, 10, 42, 60))},
        ]

    monkeypatch.setattr('basketball.utils.io_utils.load_video', fake_load_video)
    monkeypatch.setattr('basketball.cache.cache_utils.load_cache', fake_load_cache)

    namespace: dict = {'__name__': '__main__'}
    _run_imports_cell(namespace)  # must bind load_video/load_cache to the fakes above
    namespace['CLIPS'] = ('clip_1',)
    # Only frames 0 and 2 of the 3 fake_load_video yields are sampled -- also
    # proves the cell's own "if idx in set(sample_frames[clip])" filter
    # still applies correctly to faked frames, not just real ones.
    namespace['sample_frames'] = {'clip_1': [0, 2]}

    code_cells = _code_cells()
    exec(compile(code_cells[3], '<load>', 'exec'), namespace)  # the REAL load cell, not a hand-built stand-in

    frames = namespace['frames_by_clip']['clip_1']
    assert isinstance(frames, dict)  # not a bare generator: len() and frames[idx] both need this
    assert set(frames) == {0, 2}
    for idx in (0, 2):
        assert isinstance(frames[idx], np.ndarray)
        assert not isinstance(frames[idx], tuple)
        assert np.array_equal(frames[idx], np.full((20, 30, 3), idx + 1, dtype=np.uint8))

    # Feed these REAL (not hand-built) frames_by_clip/tracks_by_clip into the
    # real UI cell, spying on draw_tracked_boxes -- this notebook's
    # equivalent of the possession notebook's render_labelling_frame spy.
    namespace['CSV_PATH'] = str(tmp_path / 'labels.csv')
    exec(
        compile(
            'shuffled_order = build_shuffled_order(sample_frames)\n'
            'frame_player_ids = frame_player_ids_from_tracks(sample_frames, tracks_by_clip)\n'
            'labelled = load_existing_labels(CSV_PATH)\n',
            '<resume>',
            'exec',
        ),
        namespace,
    )
    namespace['position'] = 0

    captured_frames: list[object] = []
    real_draw = namespace['draw_tracked_boxes']

    def spy_draw(frame: object, tracks: dict) -> np.ndarray:
        captured_frames.append(frame)
        return real_draw(frame, tracks)  # calls through: a tuple fails here too, inside frame.copy()

    namespace['draw_tracked_boxes'] = spy_draw

    _run_ui_cell(namespace)  # defines render() and calls it once at the cell's own tail

    assert len(captured_frames) == 1
    frame = captured_frames[0]
    assert isinstance(frame, np.ndarray), f'render() passed a {type(frame).__name__}, not a real frame array'
    assert not isinstance(frame, tuple)
