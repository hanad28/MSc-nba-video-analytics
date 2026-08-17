"""
possession_gt.py

Frame ordering, answer parsing, append-only CSV persistence and resume for
the per-frame possession ground-truth labelling tool, plus the per-clip lazy
frame/track/ball-detection loader that keeps at most one clip's data resident
at a time.
"""
from __future__ import annotations

import csv
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from basketball.cache.cache_utils import load_cache
from basketball.detection.ball_detector import BallDetection
from basketball.detection.player_detector import PlayerTrack
from basketball.possession.possession_io import BALL_DETECTIONS_CACHE_TEMPLATE, possession_ball_input
from basketball.utils.io_utils import load_video

CLIPS = ('clip_1', 'clip_2', 'clip_3')

CSV_PATH = 'data/annotations/possession_gt_per_frame.csv'
CSV_HEADER = ['clip', 'frame_idx', 'holder', 'labelled_at']

# holder is ONE string column: a track id rendered as a string, or one of
# these two sentinels. Mixing ints and sentinels in a single column would
# force every consumer to parse two types out of one field and guess which
# it had, so the track id is stringified on the way in instead.
NOBODY = 'nobody'
UNCLEAR = 'unclear'

# Answer keys accepted alongside a bare track id.
NOBODY_KEY = 'n'
UNCLEAR_KEY = 'u'
STOP_KEY = 's'

# parse_answer() outcomes.
LABEL = 'label'
STOP = 'stop'
INVALID = 'invalid'

# Same cache the team GT tool reads its frame counts from; duplicated here as
# a one-line template rather than imported, to keep this module independent
# of the team GT module.
PLAYER_DETECTIONS_CACHE_TEMPLATE = 'data/processed/{clip}/player_detections.pkl'

RAW_VIDEO_TEMPLATE = 'data/raw/{clip}.mp4'


class ParsedAnswer(NamedTuple):
    """One parsed labeller answer: an outcome of LABEL, STOP or INVALID, carrying the holder value or the rejection message."""

    outcome: str
    value: str


# --- frame ordering -----------------------------------------------------------

def build_frame_order(frame_counts: dict[str, int], clips: tuple[str, ...] = CLIPS) -> list[tuple[str, int]]:
    """Every (clip, frame_idx) pair, chronological within each clip and clips in fixed CLIPS order."""
    # CHRONOLOGICAL ON PURPOSE, and the opposite of the team GT tool's
    # deterministic shuffle. There, shuffling stopped a labeller propagating
    # one remembered team across a track's whole life through an ID switch.
    # Here the label IS a temporal reading of the play -- who is holding the
    # ball is judged from the possession developing across neighbouring
    # frames -- so presenting frames out of order would remove the very
    # context the answer depends on. Do not "improve" this by shuffling.
    return [(clip, frame_idx) for clip in clips for frame_idx in range(frame_counts[clip])]


def default_frame_counts(clips: tuple[str, ...] = CLIPS) -> dict[str, int]:
    """Real per-clip frame counts from the cached player-track length, never a hardcoded number."""
    return {clip: len(load_cache(PLAYER_DETECTIONS_CACHE_TEMPLATE.format(clip=clip))) for clip in clips}


def frame_player_ids_from_tracks(tracks_by_clip: dict[str, list[dict[int, object]]]) -> dict[tuple[str, int], set[int]]:
    """The set of track_ids visible in every (clip, frame_idx), used to reject a holder that is not on screen in that frame."""
    return {
        (clip, frame_idx): set(tracks)
        for clip, frames in tracks_by_clip.items()
        for frame_idx, tracks in enumerate(frames)
    }


def frame_player_ids_for_all_clips(clips: tuple[str, ...] = CLIPS) -> dict[tuple[str, int], set[int]]:
    """frame_player_ids_from_tracks() built one clip's cached track list at a time, so no more than one clip's list is ever held while building it."""
    # answer validation needs every (clip, frame_idx)'s valid ids up front,
    # since run_session() can be asked to validate a frame in any clip at
    # any point -- but the RESULT (per-frame int SETS) is tiny regardless of
    # how many clips it covers, unlike a whole clip's decoded video frames.
    # clip_tracks is reassigned, not accumulated, each iteration, so only one
    # clip's raw track list is reachable at a time while this runs.
    result: dict[tuple[str, int], set[int]] = {}
    for clip in clips:
        clip_tracks = load_cache(PLAYER_DETECTIONS_CACHE_TEMPLATE.format(clip=clip))
        result.update(frame_player_ids_from_tracks({clip: clip_tracks}))
    return result


# --- per-clip lazy loading ----------------------------------------------------

class ClipData(NamedTuple):
    """One clip's decoded frames, cached player tracks and gated ball detections, all indexed by frame_idx."""

    frames: dict[int, np.ndarray]
    tracks: list[dict[int, PlayerTrack]]
    gated: list[dict[int, BallDetection]]


def load_clip_data(clip: str) -> ClipData:
    """Decode one clip's video and load its cached tracks and gated ball detections -- the only function that ever decodes a clip's video into frames."""
    # possession_ball_input() is the single source of truth for possession's
    # ball input: it runs the adopted gate BEFORE interpolation, in main.py's
    # order. .gated is what the labeller sees (real detections only); .filled
    # exists for the sweep and is deliberately never read here.
    frames = {idx: frame for idx, frame in load_video(RAW_VIDEO_TEMPLATE.format(clip=clip))}
    tracks = load_cache(PLAYER_DETECTIONS_CACHE_TEMPLATE.format(clip=clip))
    gated = possession_ball_input(load_cache(BALL_DETECTIONS_CACHE_TEMPLATE.format(clip=clip))).gated
    megabytes = sum(frame.nbytes for frame in frames.values()) / 1e6
    print(
        f'{clip}: loaded {len(frames)} frames into memory ({megabytes:.0f} MB), '
        f'{len(tracks)} track frames, '
        f'{sum(1 for f in gated if f)} frames with a gated ball detection'
    )
    return ClipData(frames=frames, tracks=tracks, gated=gated)


def make_clip_loader() -> Callable[[str], ClipData]:
    """Return a stateful getter holding at most ONE clip's ClipData at a time, reloading (and discarding whatever was cached before) whenever a different clip is asked for."""
    # Safe to evict unconditionally on a clip change, not just an
    # optimisation that might occasionally miss a later re-request:
    # build_frame_order()'s chronological, clip-at-a-time progression (every
    # clip_1 frame before any clip_2 frame, every clip_2 before any clip_3)
    # means a clip is never asked for again once labelling has moved past it.
    current_clip: str | None = None
    current_data: ClipData | None = None

    def get(clip: str) -> ClipData:
        nonlocal current_clip, current_data
        if clip != current_clip:
            if current_clip is not None:
                print(f'  (discarding {current_clip}\'s frames from memory)')
            # Released BEFORE the next clip is decoded: assigning the load's
            # result straight to current_data would keep the previous clip
            # alive throughout load_clip_data()'s decode, peaking at two
            # clips' frames at once -- the exact doubling this loader exists
            # to avoid.
            current_clip = None
            current_data = None
            current_data = load_clip_data(clip)
            current_clip = clip
        return current_data

    return get


# --- answer parsing -----------------------------------------------------------

def parse_answer(answer: str, valid_track_ids: set[int]) -> ParsedAnswer:
    """Parse one labeller answer into a holder string, a stop request, or an INVALID outcome carrying the message to show before re-presenting the same frame."""
    cleaned = answer.strip().lower()
    if cleaned == STOP_KEY:
        return ParsedAnswer(STOP, '')
    if cleaned == NOBODY_KEY:
        return ParsedAnswer(LABEL, NOBODY)
    if cleaned == UNCLEAR_KEY:
        return ParsedAnswer(LABEL, UNCLEAR)
    if not cleaned.lstrip('-').isdigit():
        return ParsedAnswer(
            INVALID,
            f'Unrecognised answer {answer.strip()!r} — enter a track id, '
            f'{NOBODY_KEY!r} (nobody), {UNCLEAR_KEY!r} (unclear) or {STOP_KEY!r} (stop).',
        )
    if int(cleaned) not in valid_track_ids:
        return ParsedAnswer(
            INVALID,
            f'Track id {int(cleaned)} is not in this frame. Valid ids: {sorted(valid_track_ids)}.',
        )
    # Stringified, so the holder column stays one type end to end.
    return ParsedAnswer(LABEL, str(int(cleaned)))


# --- CSV persistence and resume ----------------------------------------------

def load_labelled_rows(path: str = CSV_PATH) -> list[dict]:
    """Every label row from the CSV, deduplicated to the LAST row on disk for each (clip, frame_idx) key, or [] if the file does not exist yet."""
    # append_label() stays append-only for crash safety, so a correction is a
    # new row rather than a rewrite. Last-wins dedup therefore lives here, in
    # the one function EVERY consumer (resume, progress count, the sweep's
    # scoring) reads through -- a correction supersedes its earlier row in
    # exactly one place, instead of each caller reimplementing the rule.
    if not Path(path).exists():
        return []

    latest_by_key: dict[tuple[str, int], dict] = {}
    first_seen_order: list[tuple[str, int]] = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['clip'], int(row['frame_idx']))
            if key not in latest_by_key:
                first_seen_order.append(key)
            latest_by_key[key] = row
    # Returned in each key's first-appearance order carrying that key's latest
    # content, the same convention as the team GT tool. first_seen_order is
    # tracked explicitly rather than leaning on latest_by_key's own key order;
    # the two coincide today, since reassigning an existing dict key keeps its
    # original position, so this makes the intended ordering checkable rather
    # than resting on that property staying true of how the dict is built.
    return [latest_by_key[key] for key in first_seen_order]


def load_existing_labels(path: str = CSV_PATH) -> set[tuple[str, int]]:
    """The set of (clip, frame_idx) pairs already labelled (post-dedup), or an empty set if none exist yet."""
    return {(row['clip'], int(row['frame_idx'])) for row in load_labelled_rows(path)}


def append_label(
    clip: str,
    frame_idx: int,
    holder: str,
    path: str = CSV_PATH,
    labelled_at: str | None = None,
) -> None:
    """Append one label row, flushing and fsyncing so a kernel death loses at most the frame in progress."""
    # The holder column's one-type invariant is enforced at this, the only
    # write boundary, rather than merely intended: a caller passing an int or
    # None would otherwise persist a value every reader (resume, the progress
    # count, the sweep's scoring) would have to guess the type of, which is
    # the whole reason track ids are stringified rather than stored alongside
    # the sentinels.
    if not isinstance(holder, str) or (holder not in (NOBODY, UNCLEAR) and not holder.isdigit()):
        raise ValueError(
            f'holder must be a track id as a digit string, {NOBODY!r} or {UNCLEAR!r}, got {holder!r}.'
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Zero bytes counts as "needs header" as well as absent: a crash between
    # open and the first write leaves an empty file behind, and checking
    # existence alone would then make the first real label the CSV header --
    # silently dropping it and breaking every later read on a corrupt header.
    needs_header = not Path(path).exists() or Path(path).stat().st_size == 0
    if labelled_at is None:
        labelled_at = datetime.now(timezone.utc).isoformat()

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(CSV_HEADER)
        writer.writerow([clip, frame_idx, holder, labelled_at])
        f.flush()
        os.fsync(f.fileno())


def next_unlabelled_index(
    frame_order: list[tuple[str, int]],
    labelled: set[tuple[str, int]],
) -> int | None:
    """The index in frame_order of the first frame with no label yet, or None when every frame is labelled."""
    # Scans the FULL order every time rather than continuing from wherever
    # the session happens to be: a labeller who goes back to correct an
    # earlier frame, or who runs two sessions, must not leave a hole that a
    # forward-only cursor would skip past forever. Returns None rather than
    # len(frame_order) so callers cannot index past the end.
    for index, key in enumerate(frame_order):
        if key not in labelled:
            return index
    return None


def run_session(
    frame_order: list[tuple[str, int]],
    frame_player_ids: dict[tuple[str, int], set[int]],
    show: Callable[[str, int], None],
    prompt: Callable[[str], str],
    path: str = CSV_PATH,
) -> int:
    """Present unlabelled frames in order and record one answer each until the labeller stops or every frame is labelled, returning how many labels this session wrote."""
    written = 0
    while True:
        # Re-derived from disk every iteration, so a correction appended
        # mid-session is honoured and no frame can be skipped.
        index = next_unlabelled_index(frame_order, load_existing_labels(path))
        if index is None:
            print(f'All {len(frame_order)} frames are labelled — nothing left to do.')
            return written

        clip, frame_idx = frame_order[index]
        show(clip, frame_idx)
        answer = prompt(
            f'[{index + 1}/{len(frame_order)}] {clip} frame {frame_idx} — '
            f'holder track id, {NOBODY_KEY!r}=nobody, {UNCLEAR_KEY!r}=unclear, {STOP_KEY!r}=stop: '
        )
        parsed = parse_answer(answer, frame_player_ids[(clip, frame_idx)])

        if parsed.outcome == STOP:
            print(f'Stopped. {written} label(s) written this session.')
            return written
        if parsed.outcome == INVALID:
            # Nothing is written, so the loop re-presents this same frame.
            print(parsed.value)
            continue

        append_label(clip, frame_idx, parsed.value, path=path)
        written += 1
