"""Unit tests for basketball/labelling/team_gt_sampling.py: frame sampling,
the deterministic shuffled presentation order, CSV persistence/resume and
the post-hoc audit: the tested logic behind the (deliberately thin)
label_team_gt_per_frame.ipynb notebook.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from basketball.labelling.team_gt_sampling import (
    CSV_HEADER,
    PRODUCTION_CONFIG,
    append_label,
    backbone_frame_indices,
    build_shuffled_order,
    cross_reference_switches,
    flag_disagreements,
    frame_player_ids_from_tracks,
    load_existing_labels,
    load_labelled_rows,
    load_switch_configs,
    load_switch_frames,
    mot_gt_frame_indices,
    occlusion_window_frame_indices,
    resume_index,
    sample_all_clips,
    sample_clip_frame_indices,
    track_modal_team,
)
from evaluation.ground_truth import GTAnnotation, save_mot_annotations


def track(bbox: tuple[float, ...] = (0.0, 0.0, 1.0, 1.0)) -> SimpleNamespace:
    return SimpleNamespace(bbox=list(bbox))


# --- backbone -----------------------------------------------------------------

def test_backbone_takes_every_fifth_frame_from_zero():
    assert backbone_frame_indices(23) == [0, 5, 10, 15, 20]


def test_backbone_edge_cases():
    assert backbone_frame_indices(0) == []
    assert backbone_frame_indices(1) == [0]
    assert backbone_frame_indices(5) == [0]


def test_backbone_respects_a_custom_stride():
    assert backbone_frame_indices(12, stride=4) == [0, 4, 8]


# --- occlusion windows ----------------------------------------------------------

def test_occlusion_window_indices_are_inclusive_of_both_ends():
    assert occlusion_window_frame_indices([(85, 90)], frame_count=200) == [85, 86, 87, 88, 89, 90]


def test_occlusion_window_indices_union_multiple_windows_and_dedupe_overlap():
    result = occlusion_window_frame_indices([(0, 5), (3, 8)], frame_count=200)
    assert result == list(range(0, 9))


def test_occlusion_window_indices_clip_to_frame_count():
    # A window whose end reaches past the clip's actual length must not
    # request frames that do not exist.
    result = occlusion_window_frame_indices([(240, 260)], frame_count=243)
    assert result == list(range(240, 243))


def test_occlusion_window_entirely_beyond_frame_count_contributes_nothing():
    assert occlusion_window_frame_indices([(300, 320)], frame_count=100) == []


# --- MOT ground-truth frame extraction -----------------------------------------

def test_mot_gt_frame_indices_reads_the_actual_0_indexed_frames_present(tmp_path):
    path = str(tmp_path / 'clip_1_gt.txt')
    # 1-indexed MOT frames 1, 11, 21 -> 0-indexed 0, 10, 20.
    save_mot_annotations(
        [
            GTAnnotation(frame=0, track_id=1, bbox=[0.0, 0.0, 10.0, 10.0]),
            GTAnnotation(frame=10, track_id=1, bbox=[0.0, 0.0, 10.0, 10.0]),
            GTAnnotation(frame=10, track_id=2, bbox=[20.0, 20.0, 30.0, 30.0]),  # same frame, no duplicate entry
            GTAnnotation(frame=20, track_id=1, bbox=[0.0, 0.0, 10.0, 10.0]),
        ],
        path,
    )

    assert mot_gt_frame_indices(path) == [0, 10, 20]


def test_mot_gt_frame_indices_returns_empty_for_a_missing_file(tmp_path):
    assert mot_gt_frame_indices(str(tmp_path / 'absent_gt.txt')) == []


# --- per-clip union sampling: the superset property ----------------------------

def test_sample_clip_frame_indices_unions_backbone_and_stubbed_gt_that_ignores_every_10th():
    # A stubbed MOT-GT set that follows no stride at all (3, 47, 61; 61 is
    # also on the every-5th backbone, exercising dedup) must still all
    # appear in the sample: the superset property holds regardless of what
    # the real MOT stride turned out to be.
    result = sample_clip_frame_indices('clip_1', frame_count=100, gt_frames=[3, 47, 61])

    assert set(result) >= {3, 47, 61}
    assert set(result) >= set(backbone_frame_indices(100))
    assert result == sorted(set(result))  # sorted, deduped


def test_sample_clip_frame_indices_dedupes_when_gt_frame_is_already_on_the_backbone():
    result = sample_clip_frame_indices('clip_1', frame_count=50, gt_frames=[0, 5, 10])

    assert result.count(0) == 1
    assert result.count(5) == 1


def test_sample_clip_frame_indices_only_applies_occlusion_windows_to_clip_3():
    with_windows = sample_clip_frame_indices('clip_3', frame_count=243, gt_frames=[])
    without_windows = sample_clip_frame_indices('clip_1', frame_count=243, gt_frames=[])

    assert set(range(85, 116)) <= set(with_windows)
    assert set(range(200, 241)) <= set(with_windows)
    assert not (set(range(85, 116)) <= set(without_windows))


def test_sample_clip_frame_indices_reads_gt_path_when_gt_frames_not_given(tmp_path):
    path = str(tmp_path / 'clip_1_gt.txt')
    save_mot_annotations([GTAnnotation(frame=3, track_id=1, bbox=[0.0, 0.0, 10.0, 10.0])], path)

    result = sample_clip_frame_indices('clip_1', frame_count=50, gt_path=path)

    assert 3 in result


def test_sample_clip_frame_indices_clips_out_of_range_frames():
    # A stubbed GT frame beyond frame_count must not be sampled: it cannot
    # exist in the actual clip.
    result = sample_clip_frame_indices('clip_1', frame_count=20, gt_frames=[19, 25, -1])

    assert result == backbone_frame_indices(20) + [19] and max(result) < 20 and min(result) >= 0


def test_sample_all_clips_returns_one_list_per_clip():
    result = sample_all_clips({'clip_1': 20, 'clip_2': 30}, gt_paths=None)

    assert set(result) == {'clip_1', 'clip_2'}
    assert result['clip_1'] == backbone_frame_indices(20)
    assert result['clip_2'] == backbone_frame_indices(30)


# --- deterministic shuffled order -----------------------------------------------

def test_build_shuffled_order_is_reproducible_with_the_same_seed():
    per_clip = {'clip_1': [0, 5, 10], 'clip_2': [0, 5, 10], 'clip_3': [0, 5, 10]}

    first = build_shuffled_order(per_clip, seed=42)
    second = build_shuffled_order(per_clip, seed=42)

    assert first == second


def test_build_shuffled_order_differs_with_a_different_seed():
    per_clip = {'clip_1': list(range(10)), 'clip_2': list(range(10)), 'clip_3': list(range(10))}

    assert build_shuffled_order(per_clip, seed=42) != build_shuffled_order(per_clip, seed=1)


def test_build_shuffled_order_is_not_clip_chronological():
    per_clip = {'clip_1': list(range(10)), 'clip_2': list(range(10)), 'clip_3': list(range(10))}
    chronological = [(clip, frame) for clip in sorted(per_clip) for frame in per_clip[clip]]

    shuffled = build_shuffled_order(per_clip, seed=42)

    assert shuffled != chronological
    assert sorted(shuffled) == sorted(chronological)  # same items, different order


def test_build_shuffled_order_contains_every_pair_exactly_once():
    per_clip = {'clip_1': [0, 1], 'clip_2': [0]}

    shuffled = build_shuffled_order(per_clip)

    assert sorted(shuffled) == [('clip_1', 0), ('clip_1', 1), ('clip_2', 0)]


# --- frame_player_ids_from_tracks ------------------------------------------------

def test_frame_player_ids_from_tracks_builds_the_expected_mapping():
    sample_frames = {'clip_1': [0, 1]}
    tracks_by_clip = {'clip_1': [{1: track(), 2: track()}, {1: track()}]}

    result = frame_player_ids_from_tracks(sample_frames, tracks_by_clip)

    assert result == {('clip_1', 0): {1, 2}, ('clip_1', 1): {1}}


# --- CSV persistence and resume --------------------------------------------------

def test_load_existing_labels_returns_empty_set_for_a_missing_file(tmp_path):
    assert load_existing_labels(str(tmp_path / 'absent.csv')) == set()


def test_append_label_creates_the_file_with_a_header(tmp_path):
    path = str(tmp_path / 'labels.csv')

    append_label('clip_1', 0, 1, '1', path=path, labelled_at='2026-01-01T00:00:00+00:00')

    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0] == ','.join(CSV_HEADER)
    assert lines[1] == 'clip_1,0,1,1,2026-01-01T00:00:00+00:00'


def test_append_label_writes_the_header_for_a_pre_existing_zero_byte_file(tmp_path):
    # open(path, 'a') can itself create a zero-byte file (e.g. a kernel
    # crash between file creation and the first write) -- Path.exists()
    # alone would see that empty file and skip the header, silently turning
    # the first real label into a corrupted header row.
    path = tmp_path / 'labels.csv'
    path.touch()
    assert path.stat().st_size == 0

    append_label('clip_1', 0, 1, '1', path=str(path), labelled_at='2026-01-01T00:00:00+00:00')

    lines = path.read_text().splitlines()
    assert lines[0] == ','.join(CSV_HEADER)
    assert lines[1] == 'clip_1,0,1,1,2026-01-01T00:00:00+00:00'

    rows = load_labelled_rows(str(path))
    assert len(rows) == 1
    assert rows[0]['clip'] == 'clip_1'
    assert rows[0]['true_team'] == '1'


def test_append_label_appends_immediately_without_touching_prior_rows(tmp_path):
    path = str(tmp_path / 'labels.csv')

    append_label('clip_1', 0, 1, '1', path=path, labelled_at='t0')
    append_label('clip_1', 0, 2, '2', path=path, labelled_at='t1')
    append_label('clip_1', 5, 1, 'unclear', path=path, labelled_at='t2')

    rows = load_labelled_rows(path)
    assert len(rows) == 3
    assert rows[0]['true_team'] == '1'
    assert rows[1]['player_id'] == '2'
    assert rows[2]['true_team'] == 'unclear'


def test_append_label_rejects_an_invalid_true_team(tmp_path):
    path = str(tmp_path / 'labels.csv')

    with pytest.raises(ValueError, match='true_team'):
        append_label('clip_1', 0, 1, 'team_a', path=path)

    assert not load_labelled_rows(path)  # nothing written on rejection


def test_load_existing_labels_reflects_a_partially_written_file(tmp_path):
    path = str(tmp_path / 'labels.csv')
    append_label('clip_1', 0, 1, '1', path=path)
    append_label('clip_1', 0, 2, 'unclear', path=path)
    append_label('clip_2', 5, 1, '2', path=path)

    labelled = load_existing_labels(path)

    assert labelled == {('clip_1', 0, 1), ('clip_1', 0, 2), ('clip_2', 5, 1)}


# --- re-labelling (correction) dedup ---------------------------------------

def test_load_labelled_rows_keeps_only_the_last_row_for_a_re_labelled_key(tmp_path):
    path = str(tmp_path / 'labels.csv')
    # A mis-click corrected: (clip_1, 0, 4) labelled team 1, then team 2.
    append_label('clip_1', 0, 4, '1', path=path, labelled_at='t0')
    append_label('clip_1', 0, 4, '2', path=path, labelled_at='t1')

    rows = load_labelled_rows(path)

    assert len(rows) == 1
    assert rows[0]['true_team'] == '2'
    assert rows[0]['labelled_at'] == 't1'


def test_load_labelled_rows_dedup_does_not_affect_other_keys(tmp_path):
    path = str(tmp_path / 'labels.csv')
    append_label('clip_1', 0, 4, '1', path=path, labelled_at='t0')
    append_label('clip_1', 0, 5, '2', path=path, labelled_at='t1')  # different player, untouched
    append_label('clip_1', 0, 4, '2', path=path, labelled_at='t2')  # correction of the first row

    rows = load_labelled_rows(path)

    assert len(rows) == 2
    by_player = {int(row['player_id']): row['true_team'] for row in rows}
    assert by_player == {4: '2', 5: '2'}


def test_load_existing_labels_is_unaffected_by_a_superseded_row(tmp_path):
    path = str(tmp_path / 'labels.csv')
    append_label('clip_1', 0, 4, '1', path=path)
    append_label('clip_1', 0, 4, '2', path=path)  # correction, same key

    # A corrected key is still exactly one (clip, frame_idx, player_id)
    # tuple in the set -- the correction does not create a phantom second
    # membership or otherwise change resume/skip-check behaviour.
    assert load_existing_labels(path) == {('clip_1', 0, 4)}


def test_track_modal_team_never_sees_a_superseded_row():
    # frame 0 was mis-clicked '1' then corrected to '2'; frame 5 is '1'.
    # The undeduplicated raw history counts '1' twice (the stale frame-0
    # row plus frame 5) against '2' once, giving mode '1' -- wrong, since
    # the frame-0 '1' was superseded. The deduplicated view (what
    # load_labelled_rows() actually returns) has one row per key: frame 0
    # is '2', frame 5 is '1' -- an exact 1-1 tie, mode None. This is the
    # *routing* contract: track_modal_team() itself does not dedup, so only
    # the deduplicated rows must ever reach it.
    raw_rows = [
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': '1', 'labelled_at': 't0'},
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': '2', 'labelled_at': 't1'},
        {'clip': 'clip_1', 'frame_idx': '5', 'player_id': '4', 'true_team': '1', 'labelled_at': 't2'},
    ]
    deduped = [raw_rows[1], raw_rows[2]]  # what load_labelled_rows() would return for this file

    assert track_modal_team(raw_rows) == {('clip_1', 4): '1'}      # wrong: sees the superseded row
    assert track_modal_team(deduped) == {('clip_1', 4): None}      # correct: exact tie once deduped
    assert track_modal_team(raw_rows) != track_modal_team(deduped)


def test_end_to_end_relabelling_is_reflected_in_the_audit(tmp_path):
    path = str(tmp_path / 'labels.csv')
    # Every other frame for player 4 says team 1; one frame was mis-clicked
    # as team 2 and then corrected back to team 1.
    append_label('clip_1', 0, 4, '1', path=path, labelled_at='t0')
    append_label('clip_1', 5, 4, '1', path=path, labelled_at='t1')
    append_label('clip_1', 10, 4, '2', path=path, labelled_at='t2')   # mis-click
    append_label('clip_1', 10, 4, '1', path=path, labelled_at='t3')   # corrected

    rows = load_labelled_rows(path)

    assert len(rows) == 3  # the superseded team-2 row at frame 10 is gone
    assert track_modal_team(rows) == {('clip_1', 4): '1'}
    assert flag_disagreements(rows) == []  # the correction means nothing disagrees


def test_resume_index_starts_at_zero_with_no_existing_labels():
    order = [('clip_1', 0), ('clip_1', 5)]
    frame_player_ids = {('clip_1', 0): {1, 2}, ('clip_1', 5): {1}}

    assert resume_index(order, labelled=set(), frame_player_ids=frame_player_ids) == 0


def test_resume_index_skips_fully_labelled_frames():
    order = [('clip_1', 0), ('clip_1', 5), ('clip_1', 10)]
    frame_player_ids = {('clip_1', 0): {1, 2}, ('clip_1', 5): {1}, ('clip_1', 10): {1, 2}}
    labelled = {('clip_1', 0, 1), ('clip_1', 0, 2)}  # frame 0 fully done

    assert resume_index(order, labelled, frame_player_ids) == 1


def test_resume_index_lands_on_a_partially_labelled_frame_not_past_it():
    # Next Frame does not force full completion, so a frame with one
    # of two players labelled is still the correct resume point, not skipped.
    order = [('clip_1', 0), ('clip_1', 5)]
    frame_player_ids = {('clip_1', 0): {1, 2}, ('clip_1', 5): {1}}
    labelled = {('clip_1', 0, 1)}  # player 2 still unlabelled on frame 0

    assert resume_index(order, labelled, frame_player_ids) == 0


def test_resume_index_returns_length_when_everything_is_labelled():
    order = [('clip_1', 0), ('clip_1', 5)]
    frame_player_ids = {('clip_1', 0): {1}, ('clip_1', 5): {1}}
    labelled = {('clip_1', 0, 1), ('clip_1', 5, 1)}

    assert resume_index(order, labelled, frame_player_ids) == len(order)


def test_resume_index_treats_a_frame_with_no_tracked_players_as_already_done():
    order = [('clip_1', 0), ('clip_1', 5)]
    frame_player_ids = {('clip_1', 0): set(), ('clip_1', 5): {1}}

    assert resume_index(order, labelled={('clip_1', 5, 1)}, frame_player_ids=frame_player_ids) == len(order)


# --- post-hoc audit ---------------------------------------------------------------

def test_track_modal_team_is_the_majority_label_excluding_unclear():
    rows = [
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '5', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '10', 'player_id': '4', 'true_team': '2'},
        {'clip': 'clip_1', 'frame_idx': '15', 'player_id': '4', 'true_team': 'unclear'},
    ]

    assert track_modal_team(rows) == {('clip_1', 4): '1'}


def test_track_modal_team_is_per_clip_not_global_across_track_ids():
    rows = [
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '1', 'true_team': '1'},
        {'clip': 'clip_2', 'frame_idx': '0', 'player_id': '1', 'true_team': '2'},
    ]

    assert track_modal_team(rows) == {('clip_1', 1): '1', ('clip_2', 1): '2'}


def test_track_modal_team_ignores_a_track_with_only_unclear_rows():
    rows = [{'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': 'unclear'}]

    assert track_modal_team(rows) == {}


def test_flag_disagreements_finds_the_minority_label():
    rows = [
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '5', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '10', 'player_id': '4', 'true_team': '2'},  # disagrees with the mode
    ]

    flagged = flag_disagreements(rows)

    assert len(flagged) == 1
    assert flagged[0]['frame_idx'] == '10'


def test_flag_disagreements_never_flags_an_unclear_row():
    rows = [
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '5', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '10', 'player_id': '4', 'true_team': 'unclear'},
    ]

    assert flag_disagreements(rows) == []


def test_flag_disagreements_returns_nothing_for_a_perfectly_consistent_track():
    rows = [
        {'clip': 'clip_1', 'frame_idx': str(i), 'player_id': '4', 'true_team': '1'}
        for i in range(5)
    ]

    assert flag_disagreements(rows) == []


# --- exact-tie modal team --------------------------------------------------

def test_track_modal_team_resolves_a_50_50_split_to_none():
    rows = [
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '5', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '10', 'player_id': '4', 'true_team': '2'},
        {'clip': 'clip_1', 'frame_idx': '15', 'player_id': '4', 'true_team': '2'},
    ]

    assert track_modal_team(rows) == {('clip_1', 4): None}


def test_track_modal_team_tie_is_independent_of_row_order():
    # A perfect split must resolve to None regardless of which team's rows
    # happen to come first, not a row-order-dependent pick of either side.
    forward = [
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '5', 'player_id': '4', 'true_team': '2'},
    ]
    reversed_rows = list(reversed(forward))

    assert track_modal_team(forward) == track_modal_team(reversed_rows) == {('clip_1', 4): None}


def test_flag_disagreements_flags_every_row_of_a_tied_track_not_a_row_order_dependent_half():
    rows = [
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '5', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '10', 'player_id': '4', 'true_team': '2'},
        {'clip': 'clip_1', 'frame_idx': '15', 'player_id': '4', 'true_team': '2'},
    ]

    flagged = flag_disagreements(rows)

    # A perfect split is precisely the ID-switch signature this audit exists
    # to detect: every row for the tied track must be surfaced, not half.
    assert len(flagged) == 4
    assert flagged == rows


def test_flag_disagreements_tie_does_not_affect_other_tracks():
    rows = [
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '4', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '5', 'player_id': '4', 'true_team': '2'},   # tied track
        {'clip': 'clip_1', 'frame_idx': '0', 'player_id': '9', 'true_team': '1'},
        {'clip': 'clip_1', 'frame_idx': '5', 'player_id': '9', 'true_team': '1'},   # consistent track
    ]

    flagged = flag_disagreements(rows)

    assert {(row['player_id']) for row in flagged} == {'4'}


SWITCHES_CSV_TEXT = (
    'config,clip,switch_frame\n'
    'production,clip_3,90\n'
    'production,clip_3,210\n'
    'lowscore_025,clip_3,90\n'   # same frame as a production switch, different config
    'lowscore_025,clip_3,150\n'  # a non-production-only switch, no production switch at this frame
    'production,clip_1,110\n'
)


def test_load_switch_configs_returns_empty_dict_when_the_file_is_absent(tmp_path):
    assert load_switch_configs(str(tmp_path / 'absent_switches.csv')) == {}


def test_load_switch_configs_maps_every_matching_config_per_clip_and_frame(tmp_path):
    path = tmp_path / 'switches.csv'
    path.write_text(SWITCHES_CSV_TEXT)

    result = load_switch_configs(str(path))

    assert result == {
        ('clip_3', 90): {'production', 'lowscore_025'},
        ('clip_3', 210): {'production'},
        ('clip_3', 150): {'lowscore_025'},
        ('clip_1', 110): {'production'},
    }


def test_load_switch_frames_returns_empty_dict_when_the_file_is_absent(tmp_path):
    assert load_switch_frames(str(tmp_path / 'absent_switches.csv')) == {}


def test_load_switch_frames_defaults_to_the_production_configuration_only(tmp_path):
    path = tmp_path / 'switches.csv'
    path.write_text(SWITCHES_CSV_TEXT)

    result = load_switch_frames(str(path))

    # clip_3 frame 150 is a lowscore_025-only switch and must not appear:
    # unioning across every sweep configuration would let a non-adopted
    # configuration's switch inflate the production-relevant claim.
    assert result == {'clip_3': {90, 210}, 'clip_1': {110}}
    assert PRODUCTION_CONFIG == 'production'


def test_load_switch_frames_can_select_a_different_configuration(tmp_path):
    path = tmp_path / 'switches.csv'
    path.write_text(SWITCHES_CSV_TEXT)

    result = load_switch_frames(str(path), config='lowscore_025')

    assert result == {'clip_3': {90, 150}}


def test_cross_reference_switches_flags_production_coincidence_correctly():
    flagged_rows = [
        {'clip': 'clip_3', 'frame_idx': '90', 'player_id': '4', 'true_team': '2'},
        {'clip': 'clip_3', 'frame_idx': '55', 'player_id': '4', 'true_team': '2'},
    ]
    switch_frames_by_clip = {'clip_3': {90, 210}}  # production only

    result = cross_reference_switches(flagged_rows, switch_frames_by_clip)

    assert result[0]['production_switch'] is True
    assert result[1]['production_switch'] is False
    # Original row keys are preserved alongside the new ones.
    assert result[0]['player_id'] == '4'


def test_cross_reference_switches_records_matched_configs_without_inflating_production_switch():
    # clip_3 frame 150 switches under lowscore_025 but NOT under production --
    # production_switch must stay False while matched_configs still surfaces
    # the non-production match, rather than discarding the information.
    flagged_rows = [{'clip': 'clip_3', 'frame_idx': '150', 'player_id': '7', 'true_team': '1'}]
    switch_frames_by_clip = {'clip_3': {90, 210}}
    switch_configs_by_clip_frame = {('clip_3', 150): {'lowscore_025'}}

    result = cross_reference_switches(flagged_rows, switch_frames_by_clip, switch_configs_by_clip_frame)

    assert result[0]['production_switch'] is False
    assert result[0]['matched_configs'] == ['lowscore_025']


def test_cross_reference_switches_matched_configs_defaults_to_empty_list_without_the_mapping():
    flagged_rows = [{'clip': 'clip_3', 'frame_idx': '90', 'player_id': '4', 'true_team': '2'}]

    result = cross_reference_switches(flagged_rows, switch_frames_by_clip={'clip_3': {90}})

    assert result[0]['production_switch'] is True
    assert result[0]['matched_configs'] == []


def test_cross_reference_switches_handles_a_clip_with_no_recorded_switches():
    flagged_rows = [{'clip': 'clip_2', 'frame_idx': '10', 'player_id': '1', 'true_team': '1'}]

    result = cross_reference_switches(flagged_rows, switch_frames_by_clip={})

    assert result[0]['production_switch'] is False
    assert result[0]['matched_configs'] == []
