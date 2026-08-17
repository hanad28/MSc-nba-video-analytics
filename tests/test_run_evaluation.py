"""Unit tests for the MOT evaluation sweep's non-inference logic (scripts/run_evaluation.py).

The configuration matrix, the per-clip error handling and the table cells are
covered with synthetic data; the tracking runs themselves need weights, clips
and the GPU stack, so they are exercised on JupyterHub, not here.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import scripts.run_evaluation as run_evaluation
from basketball.detection.player_detector import PlayerTrack
from evaluation.ground_truth import GTAnnotation, save_mot_annotations
from evaluation.mot_metrics import MOTEvaluator, MOTResult
from scripts.run_evaluation import (
    CONFIGURATIONS,
    RESULT_HEADERS,
    SWITCHES_HEADER,
    evaluate_clip,
    load_scoreable_ground_truth,
    pooled_row,
    result_cells,
    track_clip,
    write_outputs,
)

BOX = [0.0, 0.0, 10.0, 10.0]


def test_the_configuration_matrix_matches_the_spec():
    assert [
        (config.label, config.model_path, config.conf_threshold, config.minimum_consecutive_frames)
        for config in CONFIGURATIONS
    ] == [
        ('production', 'models/ball.pt', 0.5, 2),
        ('players_pt', 'models/players.pt', 0.5, 2),
        ('lowscore_025', 'models/ball.pt', 0.25, 2),
        ('lowscore_010', 'models/ball.pt', 0.10, 2),
        ('mcf1', 'models/ball.pt', 0.5, 1),
    ]


def test_evaluate_clip_returns_a_result_for_valid_inputs():
    evaluator = MOTEvaluator()
    ground_truth = [GTAnnotation(frame=0, track_id=1, bbox=list(BOX))]
    tracks = [{1: PlayerTrack(track_id=1, bbox=list(BOX), confidence=0.9)}]

    result = evaluate_clip(evaluator, 'production', 'clip_1', ground_truth, tracks)

    assert isinstance(result, MOTResult)
    assert result.mota == 1.0


def test_evaluate_clip_skips_and_reports_a_mismatched_clip_instead_of_aborting(capsys):
    evaluator = MOTEvaluator()
    ground_truth = [GTAnnotation(frame=20, track_id=1, bbox=list(BOX))]
    tracks: list[dict[int, PlayerTrack]] = [{}] * 5

    assert evaluate_clip(evaluator, 'lowscore_010', 'clip_2', ground_truth, tracks) is None

    out = capsys.readouterr().out
    assert 'lowscore_010 / clip_2' in out            # names the configuration and clip
    assert 'highest GT frame index 20' in out        # forwards the full message
    assert '5 tracker frames' in out


def test_a_configuration_with_no_scoreable_clip_produces_no_pooled_row(capsys):
    evaluator = MOTEvaluator()

    assert pooled_row(evaluator, 'lowscore_010', []) is None

    out = capsys.readouterr().out
    assert 'lowscore_010' in out
    assert 'no clip could be scored' in out
    assert 'pooled row omitted' in out


def test_a_scoreable_configuration_gets_a_pooled_row(capsys):
    evaluator = MOTEvaluator()
    ground_truth = [GTAnnotation(frame=0, track_id=1, bbox=list(BOX))]
    tracks = [{1: PlayerTrack(track_id=1, bbox=list(BOX), confidence=0.9)}]

    row = pooled_row(evaluator, 'production', [(ground_truth, tracks)])

    assert row is not None
    assert row[:2] == ['production', 'all']
    assert row[2 + RESULT_HEADERS.index('mota')] == '1.0000'


def test_a_zero_annotation_ground_truth_file_is_skipped_not_scored(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_evaluation, 'GT_PATH_TEMPLATE', str(tmp_path / '{clip}_gt.txt'))
    (tmp_path / 'clip_1_gt.txt').write_text('')   # exists, holds no annotations

    assert load_scoreable_ground_truth('clip_1') is None

    out = capsys.readouterr().out
    assert 'clip_1' in out
    assert 'no annotations' in out and 'skipped' in out


def test_a_missing_gt_file_skips_the_clip_and_the_run_continues(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_evaluation, 'GT_PATH_TEMPLATE', str(tmp_path / '{clip}_gt.txt'))
    # clip_1 has no gt file at all, the state the labelling tool leaves for a
    # never-labelled clip, since it deliberately writes no empty files.
    save_mot_annotations(
        [GTAnnotation(frame=0, track_id=1, bbox=list(BOX))],
        str(tmp_path / 'clip_2_gt.txt'),
    )

    evaluator = MOTEvaluator()
    tracks = [{1: PlayerTrack(track_id=1, bbox=list(BOX), confidence=0.9)}]

    results = {}
    for clip in ('clip_1', 'clip_2'):                # the sweep's per-clip guard-and-continue shape
        ground_truth = load_scoreable_ground_truth(clip)
        if ground_truth is None:
            continue
        results[clip] = evaluate_clip(evaluator, 'production', clip, ground_truth, tracks)

    out = capsys.readouterr().out
    assert 'clip_1' in out and 'no ground-truth file' in out and 'skipped' in out
    assert list(results) == ['clip_2']               # the run continued past the missing clip
    assert results['clip_2'].mota == 1.0


def test_a_labelled_ground_truth_file_loads_for_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evaluation, 'GT_PATH_TEMPLATE', str(tmp_path / '{clip}_gt.txt'))
    save_mot_annotations(
        [GTAnnotation(frame=0, track_id=1, bbox=list(BOX))],
        str(tmp_path / 'clip_1_gt.txt'),
    )

    ground_truth = load_scoreable_ground_truth('clip_1')

    assert ground_truth is not None
    assert len(ground_truth) == 1


def test_an_unreadable_clip_is_skipped_with_a_message_not_fatal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_evaluation, 'CLIP_PATH_TEMPLATE', str(tmp_path / '{clip}.mp4'))
    # No clip file exists: load_video raises IOError, which previously
    # escaped every guard and killed the run before any CSV was written.
    tracks = track_clip(detector=None, config_label='production', clip_name='clip_1')

    assert tracks is None
    out = capsys.readouterr().out
    assert 'production / clip_1' in out
    assert 'could not be read' in out and 'skipped' in out


class _CacheFaultDetector:
    """Stands in for a detector whose cache layer fails after the clip was read fine."""

    def run_tracking(self, frames: list, video_path: str, cache_path: str) -> list:
        raise IOError('disk full while writing the tracks cache')


def test_an_ioerror_from_the_tracking_step_propagates_not_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evaluation, 'CLIP_PATH_TEMPLATE', str(tmp_path / '{clip}.mp4'))
    monkeypatch.setattr(run_evaluation, 'TRACKS_CACHE_TEMPLATE', str(tmp_path / '{clip}_{label}.pkl'))
    writer = cv2.VideoWriter(
        str(tmp_path / 'clip_1.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (16, 16)
    )
    for _ in range(2):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()

    # The clip itself reads fine; the IOError comes from the cache layer inside
    # run_tracking. Reporting it as "clip could not be read" would hide the
    # real fault and bin the finished tracking; it must surface as itself.
    with pytest.raises(IOError, match='disk full'):
        track_clip(_CacheFaultDetector(), 'production', 'clip_1')


def test_write_outputs_persists_both_csvs_with_partial_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evaluation, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(run_evaluation, 'RESULTS_CSV', tmp_path / 'results.csv')
    monkeypatch.setattr(run_evaluation, 'SWITCHES_CSV', tmp_path / 'switches.csv')

    # Called from main()'s finally: partial rows from an interrupted sweep
    # must land on disk exactly as complete ones would.
    write_outputs(
        per_clip_rows=[['production', 'clip_1'] + ['0.5'] * len(RESULT_HEADERS)],
        pooled_rows=[['production', 'all'] + ['0.5'] * len(RESULT_HEADERS)],
        switch_rows=[['production', 'clip_1', '30']],
    )

    results = (tmp_path / 'results.csv').read_text()
    assert 'per_clip,production,clip_1' in results
    assert 'pooled,production,all' in results
    assert 'lower bound' in results                     # the switch caveat reaches the CSV header
    assert 'production,clip_1,30' in (tmp_path / 'switches.csv').read_text()


def test_result_cells_align_with_the_headers_and_the_switch_caveat():
    result = MOTResult(
        mota=0.5, motp=0.75, idf1=0.25, num_switches=3, num_false_positives=2,
        num_misses=4, num_objects=10, num_matches=6, precision=0.8, recall=0.6,
    )

    cells = result_cells(result)

    assert len(cells) == len(RESULT_HEADERS)
    assert 'lower bound' in SWITCHES_HEADER and 'sampled GT' in SWITCHES_HEADER
    assert cells[RESULT_HEADERS.index(SWITCHES_HEADER)] == '3'
    assert cells[RESULT_HEADERS.index('mota')] == '0.5000'
