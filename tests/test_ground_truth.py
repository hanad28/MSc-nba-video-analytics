"""Unit tests for MOT Challenge ground-truth I/O, using synthetic in-memory data only."""

from __future__ import annotations

import pytest

from evaluation.ground_truth import GTAnnotation, load_mot_annotations, save_mot_annotations


def test_gt_annotation_dataclass_fields():
    annotation = GTAnnotation(frame=3, track_id=9, bbox=[1.0, 2.0, 3.0, 4.0])

    assert annotation.frame == 3
    assert annotation.track_id == 9
    assert annotation.bbox == [1.0, 2.0, 3.0, 4.0]


def test_round_trip_preserves_annotations_exactly(tmp_path):
    path = str(tmp_path / 'gt.txt')
    original = [
        GTAnnotation(frame=0, track_id=1, bbox=[12.0, 24.5, 100.25, 260.75]),
        GTAnnotation(frame=41, track_id=7, bbox=[0.0, 0.0, 15.5, 30.0]),
        GTAnnotation(frame=41, track_id=8, bbox=[640.0, 360.25, 700.5, 480.0]),
    ]

    save_mot_annotations(original, path)

    assert load_mot_annotations(path) == original


def test_load_converts_frame_1_and_tlwh_to_pipeline_conventions(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('1,5,10.0,20.0,30.0,40.0,1,-1,-1,-1\n')

    assert load_mot_annotations(str(path)) == [
        GTAnnotation(frame=0, track_id=5, bbox=[10.0, 20.0, 40.0, 60.0])
    ]


def test_save_writes_1_indexed_top_left_width_height_rows(tmp_path):
    path = tmp_path / 'gt.txt'

    save_mot_annotations([GTAnnotation(frame=0, track_id=5, bbox=[10.0, 20.0, 40.0, 60.0])], str(path))

    assert path.read_text().strip() == '1,5,10.0,20.0,30.0,40.0,1,-1,-1,-1'


def test_load_raises_naming_the_line_for_a_wrong_column_count(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('1,1,10,20,30,40,1,-1,-1,-1\n2,2,10,20,30\n')

    with pytest.raises(ValueError, match='line 2'):
        load_mot_annotations(str(path))


def test_load_raises_naming_the_line_for_a_non_numeric_field(tmp_path):
    # A header row is the realistic way this happens.
    path = tmp_path / 'gt.txt'
    path.write_text('frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z\n1,1,10,20,30,40,1,-1,-1,-1\n')

    with pytest.raises(ValueError, match='line 1'):
        load_mot_annotations(str(path))


def test_load_raises_for_frame_zero_in_a_1_indexed_file(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('0,1,10,20,30,40,1,-1,-1,-1\n')

    with pytest.raises(ValueError, match='1-indexed'):
        load_mot_annotations(str(path))


def test_load_raises_for_a_zero_width_box(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('1,1,10,20,0,40,1,-1,-1,-1\n')

    with pytest.raises(ValueError, match='line 1.*positive'):
        load_mot_annotations(str(path))


def test_load_raises_for_a_negative_height_box(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('1,1,10,20,30,-40,1,-1,-1,-1\n')

    with pytest.raises(ValueError, match='line 1.*positive'):
        load_mot_annotations(str(path))


def test_load_raises_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mot_annotations(str(tmp_path / 'absent.txt'))


def test_load_returns_an_empty_list_for_an_empty_file(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('')

    assert load_mot_annotations(str(path)) == []


def test_save_raises_for_a_non_positive_box_dimension_without_writing(tmp_path):
    path = tmp_path / 'gt.txt'
    degenerate = [GTAnnotation(frame=0, track_id=1, bbox=[10.0, 20.0, 10.0, 60.0])]

    with pytest.raises(ValueError, match='non-positive'):
        save_mot_annotations(degenerate, str(path))

    assert not path.exists()  # validation happens before anything touches disk


def test_save_raises_for_a_negative_frame():
    with pytest.raises(ValueError, match='cannot be negative'):
        save_mot_annotations([GTAnnotation(frame=-1, track_id=1, bbox=[0.0, 0.0, 5.0, 5.0])], 'unused.txt')
