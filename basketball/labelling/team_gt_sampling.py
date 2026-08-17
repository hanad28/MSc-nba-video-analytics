"""
team_gt_sampling.py

Frame sampling, deterministic presentation order, CSV persistence/resume
and post-hoc audit for the per-frame team-classification ground-truth
redo; the labelling notebook only drives widgets and calls into these
functions, so the notebook stays thin.
"""
# Why per-frame, not per-track: asking for one team per track_id for the
# whole clip lets a human labeller build a memory of a track's previous
# label and silently reuse it across an ID switch, a human analogue of the
# sticky-cache problem this ground truth exists to catch. Labelling per
# (frame, player) instead, in a shuffled order that breaks
# clip-chronological adjacency, removes that memory effect by
# construction (see build_shuffled_order).
#
# Frame sampling combines three deliberately overlapping sources per clip,
# unioned and deduplicated:
#   (a) an every-BACKBONE_STRIDE-th frame backbone (dense enough coverage
#       everywhere without labelling every frame),
#   (b) every frame inside clip_3's two occlusion windows
#       (CLIP_3_OCCLUSION_WINDOWS): clip_3 carries the pipeline's one
#       confirmed, visually verified tracking failure mode, so team
#       assignment there needs denser ground truth to separate a genuine
#       ID switch from a labelling error,
#   (c) the real MOT ground-truth frame indices already labelled for
#       tracking evaluation (data/annotations/{clip}_gt.txt via
#       evaluation.ground_truth), which lets a later analysis
#       cross-reference team labels against known tracking identity
#       directly on the same frames.
# Taking the union guarantees every MOT-GT frame is also a team-GT frame
# regardless of what stride the MOT sampling used:
# sample_clip_frame_indices does not assume a stride for (c), it reads the
# actual frames present.
from __future__ import annotations

import csv
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from basketball.cache.cache_utils import load_cache
from evaluation.ground_truth import load_mot_annotations

CLIPS = ('clip_1', 'clip_2', 'clip_3')

# Every-5th-frame backbone, 0-indexed, starting at frame 0.
BACKBONE_STRIDE = 5

# clip_3's two occlusion windows, from the switch-frame analysis behind
# scripts/extract_switch_stills.py's
# SWITCH_FRAMES = [40, 90, 150, 210, 230, 240]: switches begin at frame 90
# onward (the loose-ball recovery / fast break) and a three-switch run at
# 210/230/240 spans the closing dunk sequence. That analysis records
# individual switch frames and a qualitative description, not a precise
# start/end for either window, so approximate ranges are used: ~85-115
# around the frame-90 switch, ~200-240 around the 210/230/240 run.
CLIP_3_OCCLUSION_WINDOWS = [(85, 115), (200, 240)]

# Reproducible presentation order: same seed -> same shuffle -> resuming a
# partial session lands on the correct next item.
SHUFFLE_SEED = 42

CSV_PATH = 'data/annotations/team_assignment_gt_per_frame.csv'
CSV_HEADER = ['clip', 'frame_idx', 'player_id', 'true_team', 'labelled_at']
VALID_TEAMS = ('1', '2', 'unclear')

PLAYER_DETECTIONS_CACHE_TEMPLATE = 'data/processed/{clip}/player_detections.pkl'
MOT_GT_PATH_TEMPLATE = 'data/annotations/{clip}_gt.txt'
SWITCHES_CSV_PATH = 'data/outputs/mot_evaluation/switches.csv'

# The shipped tracker configuration (see scripts/run_evaluation.py's
# CONFIGURATIONS) -- the only one the dissertation's cross-reference claim
# is about. switches.csv also records four non-adopted sweep configurations
# (lowscore_025, lowscore_010, mcf1, players_pt); unioning across all five
# would let a disagreement read as "coincides with a known switch" purely
# because a configuration nobody shipped switched there.
PRODUCTION_CONFIG = 'production'


# --- frame sampling ----------------------------------------------------------

def backbone_frame_indices(frame_count: int, stride: int = BACKBONE_STRIDE) -> list[int]:
    """Every stride-th 0-indexed frame starting at 0, up to (excluding) frame_count."""
    return list(range(0, frame_count, stride))


def occlusion_window_frame_indices(windows: list[tuple[int, int]], frame_count: int) -> list[int]:
    """Every 0-indexed frame inside any of the given inclusive [start, end] windows, clipped to [0, frame_count)."""
    indices: set[int] = set()
    for start, end in windows:
        clipped_end = min(end, frame_count - 1)
        if clipped_end < start:
            continue
        indices.update(range(max(start, 0), clipped_end + 1))
    return sorted(indices)


def mot_gt_frame_indices(gt_path: str) -> list[int]:
    """0-indexed frames actually present in an MOT ground-truth file, or [] if it does not exist yet."""
    if not Path(gt_path).exists():
        return []
    return sorted({annotation.frame for annotation in load_mot_annotations(gt_path)})


def sample_clip_frame_indices(
    clip_name: str,
    frame_count: int,
    gt_path: str | None = None,
    gt_frames: list[int] | None = None,
) -> list[int]:
    """Union of the backbone, clip_3's occlusion windows and the real MOT ground-truth frames for one clip, deduplicated, sorted and clipped to [0, frame_count)."""
    frames = set(backbone_frame_indices(frame_count))

    if clip_name == 'clip_3':
        frames.update(occlusion_window_frame_indices(CLIP_3_OCCLUSION_WINDOWS, frame_count))

    # The superset property holds by construction, as the module docstring
    # says: whatever frames are present are read, never a stride. gt_frames
    # overrides gt_path when both are given, so the guarantee can be exercised
    # with a stubbed frame list in tests, without a real MOT file on disk.
    if gt_frames is not None:
        frames.update(gt_frames)
    elif gt_path is not None:
        frames.update(mot_gt_frame_indices(gt_path))

    return sorted(frame for frame in frames if 0 <= frame < frame_count)


def sample_all_clips(
    frame_counts: dict[str, int],
    gt_paths: dict[str, str] | None = None,
) -> dict[str, list[int]]:
    """sample_clip_frame_indices() for every clip in frame_counts, keyed by clip name."""
    gt_paths = gt_paths or {}
    return {
        clip: sample_clip_frame_indices(clip, frame_count, gt_path=gt_paths.get(clip))
        for clip, frame_count in frame_counts.items()
    }


def default_frame_counts(clips: tuple[str, ...] = CLIPS) -> dict[str, int]:
    """Real per-clip frame counts from the cached player-track length, never a hardcoded number."""
    return {clip: len(load_cache(PLAYER_DETECTIONS_CACHE_TEMPLATE.format(clip=clip))) for clip in clips}


def default_gt_paths(clips: tuple[str, ...] = CLIPS) -> dict[str, str]:
    """The standard MOT ground-truth file path for every clip."""
    return {clip: MOT_GT_PATH_TEMPLATE.format(clip=clip) for clip in clips}


# --- deterministic shuffled presentation order --------------------------------

def build_shuffled_order(per_clip_frames: dict[str, list[int]], seed: int = SHUFFLE_SEED) -> list[tuple[str, int]]:
    """Every (clip, frame_idx) pair across all clips, shuffled with a fixed seed rather than presented in clip-chronological order."""
    # The memory effect this breaks is described in the module docstring.
    # Clips are iterated in sorted order before shuffling so the pre-shuffle
    # sequence (and therefore the shuffled one) does not depend on dict
    # insertion order, only on which frames were sampled.
    ordered = [(clip, frame) for clip in sorted(per_clip_frames) for frame in per_clip_frames[clip]]
    random.Random(seed).shuffle(ordered)
    return ordered


def frame_player_ids_from_tracks(
    sample_frames: dict[str, list[int]],
    tracks_by_clip: dict[str, list],
) -> dict[tuple[str, int], set[int]]:
    """{(clip, frame_idx): {player_id, ...}} for every sampled frame, from each clip's per-frame track dicts."""
    result: dict[tuple[str, int], set[int]] = {}
    for clip, frames in sample_frames.items():
        tracks = tracks_by_clip[clip]
        for frame_idx in frames:
            result[(clip, frame_idx)] = set(tracks[frame_idx].keys())
    return result


# --- CSV persistence and resume ----------------------------------------------

def load_labelled_rows(path: str = CSV_PATH) -> list[dict]:
    """Every label row from the CSV, deduplicated to the LAST row on disk for each (clip, frame_idx, player_id) key, or [] if the file does not exist yet."""
    # append_label() stays append-only for crash safety -- a correction
    # (re-labelling a mis-click) is a new row, never a rewrite of the old
    # one -- so the file itself can hold more than one row per key. This is
    # the single place that history is resolved to a current view: every
    # other consumer (resume/skip-check, the audit functions, the
    # notebook's label_lookup and progress count) MUST read through this
    # function rather than the raw CSV, so duplicates are resolved in
    # exactly one place, not reinvented per caller. Row order below is the
    # original first-occurrence order of each key, with that key's latest
    # content.
    if not Path(path).exists():
        return []

    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))

    latest_by_key: dict[tuple[str, int, int], dict] = {}
    first_seen_order: list[tuple[str, int, int]] = []
    for row in rows:
        key = (row['clip'], int(row['frame_idx']), int(row['player_id']))
        if key not in latest_by_key:
            first_seen_order.append(key)
        latest_by_key[key] = row  # a later row for the same key supersedes the earlier one

    return [latest_by_key[key] for key in first_seen_order]


def load_existing_labels(path: str = CSV_PATH) -> set[tuple[str, int, int]]:
    """The set of (clip, frame_idx, player_id) tuples already labelled (post-dedup), or empty set if none exist yet."""
    return {
        (row['clip'], int(row['frame_idx']), int(row['player_id']))
        for row in load_labelled_rows(path)
    }


def append_label(
    clip: str,
    frame_idx: int,
    player_id: int,
    true_team: str,
    path: str = CSV_PATH,
    labelled_at: str | None = None,
) -> None:
    """Append one label row to the CSV immediately (a single flushed, fsync'd write, never batched), so a kernel restart mid-session can lose at most the label in flight, never a previously completed one."""
    if true_team not in VALID_TEAMS:
        raise ValueError(f'true_team must be one of {VALID_TEAMS}, got {true_team!r}.')

    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)

    # A header is needed when the file doesn't exist OR exists but is zero
    # bytes -- open(path, 'a') can itself create an empty file (e.g. on a
    # kernel crash between file creation and the first write), so
    # Path(path).exists() alone is true for a header-less empty file. Missing
    # that would make the first real label become the header row, silently
    # dropping it and breaking every later read (resume, load_existing_labels,
    # the audit, the notebook's progress count) on the corrupted header.
    needs_header = not Path(path).exists() or Path(path).stat().st_size == 0
    if labelled_at is None:
        labelled_at = datetime.now(timezone.utc).isoformat()

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(CSV_HEADER)
        writer.writerow([clip, frame_idx, player_id, true_team, labelled_at])
        f.flush()
        os.fsync(f.fileno())


def resume_index(
    shuffled_order: list[tuple[str, int]],
    labelled: set[tuple[str, int, int]],
    frame_player_ids: dict[tuple[str, int], set[int]],
) -> int:
    """Index into shuffled_order of the first frame with at least one currently-tracked player not yet labelled, or len(shuffled_order) when every sampled frame's players are all labelled."""
    # A frame with some but not all players labelled still counts as the
    # resume point -- Next Frame does not force full completion, so a
    # partially labelled frame is exactly where a resumed session should land.
    for index, (clip, frame_idx) in enumerate(shuffled_order):
        players = frame_player_ids.get((clip, frame_idx), set())
        if any((clip, frame_idx, player_id) not in labelled for player_id in players):
            return index
    return len(shuffled_order)


# --- post-hoc audit ------------------------------------------------------------

def track_modal_team(rows: list[dict]) -> dict[tuple[str, int], str | None]:
    """The modal true_team (excluding 'unclear') for every (clip, player_id) across its labelled rows; an exact tie (e.g. a 50/50 split) resolves to None rather than picking a side by row order."""
    # A perfect split is precisely the ID-switch signature this audit exists
    # to detect, so breaking it arbitrarily would hide the very thing being
    # looked for. Every (clip, player_id) that has at least one
    # non-'unclear' row appears in the result, tied or not --
    # flag_disagreements() relies on that to tell "no mode" apart from
    # "never labelled".
    counters: dict[tuple[str, int], Counter] = {}
    for row in rows:
        if row['true_team'] == 'unclear':
            continue
        key = (row['clip'], int(row['player_id']))
        counters.setdefault(key, Counter())[row['true_team']] += 1

    modal: dict[tuple[str, int], str | None] = {}
    for key, counter in counters.items():
        ranked = counter.most_common()
        # A tie for first place means more than one team shares the top
        # count; most_common() lists ties adjacently at the same count.
        is_tie = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
        modal[key] = None if is_tie else ranked[0][0]
    return modal


def flag_disagreements(rows: list[dict]) -> list[dict]:
    """Every row that is a labelling-error/ID-switch candidate: its true_team disagrees with its track's modal team, or that track has no modal team at all (an exact tie), in which case every one of that track's rows is flagged."""
    # 'unclear' rows are never flagged and never used to compute the mode.
    modal = track_modal_team(rows)
    flagged = []
    for row in rows:
        if row['true_team'] == 'unclear':
            continue
        key = (row['clip'], int(row['player_id']))
        if key not in modal:
            continue
        track_modal = modal[key]
        if track_modal is None or row['true_team'] != track_modal:
            flagged.append(row)
    return flagged


def load_switch_configs(switches_csv_path: str = SWITCHES_CSV_PATH) -> dict[tuple[str, int], set[str]]:
    """{(clip, 0-indexed switch frame): {config, ...}} for every switch event in scripts/run_evaluation.py's switches.csv, across all tracker configurations, or {} if it is absent."""
    if not Path(switches_csv_path).exists():
        return {}

    switch_configs: dict[tuple[str, int], set[str]] = {}
    with open(switches_csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['clip'], int(row['switch_frame']))
            switch_configs.setdefault(key, set()).add(row['config'])
    return switch_configs


def load_switch_frames(switches_csv_path: str = SWITCHES_CSV_PATH, config: str = PRODUCTION_CONFIG) -> dict[str, set[int]]:
    """{clip: {0-indexed switch frame, ...}} for one tracker configuration (default PRODUCTION_CONFIG), or {} if switches.csv is absent."""
    switch_configs = load_switch_configs(switches_csv_path)
    switch_frames: dict[str, set[int]] = {}
    for (clip, frame), configs in switch_configs.items():
        if config in configs:
            switch_frames.setdefault(clip, set()).add(frame)
    return switch_frames


def cross_reference_switches(
    flagged_rows: list[dict],
    switch_frames_by_clip: dict[str, set[int]],
    switch_configs_by_clip_frame: dict[tuple[str, int], set[str]] | None = None,
) -> list[dict]:
    """Annotate every flagged disagreement row with production_switch (whether its frame is a switch under switch_frames_by_clip) and matched_configs (every configuration, if any, that recorded a switch there)."""
    switch_configs_by_clip_frame = switch_configs_by_clip_frame or {}
    annotated = []
    for row in flagged_rows:
        clip, frame_idx = row['clip'], int(row['frame_idx'])
        annotated.append({
            **row,
            'production_switch': frame_idx in switch_frames_by_clip.get(clip, set()),
            'matched_configs': sorted(switch_configs_by_clip_frame.get((clip, frame_idx), set())),
        })
    return annotated
