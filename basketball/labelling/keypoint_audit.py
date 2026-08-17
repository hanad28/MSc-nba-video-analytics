"""
keypoint_audit.py

Frame sampling, verdict parsing, append-only CSV persistence and resume for
the court-keypoint identification audit, which asks a human whether keypoint
index k sits on the landmark that index is defined to mean.
"""
from __future__ import annotations

import csv
import os
import random
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from basketball.keypoints.court_keypoints import CourtKeypoints, Keypoint
from basketball.keypoints.court_template import NUM_KEYPOINTS
from basketball.utils.io_utils import load_video

CLIPS = ('clip_1', 'clip_2', 'clip_3')

CSV_PATH = 'data/annotations/keypoint_audit.csv'
CSV_HEADER = [
    'clip', 'frame_idx', 'keypoint_index', 'verdict', 'actual_index', 'confidence', 'labelled_at',
]

RAW_VIDEO_TEMPLATE = 'data/raw/{clip}.mp4'
KEYPOINTS_CACHE_TEMPLATE = 'data/processed/{clip}/keypoints.pkl'
MODEL_PATH = 'models/keypoints.pt'

# The per-keypoint confidence threshold, matching the measurement script's.
# A keypoint below it is not drawn and is not verdicted.
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5

# 25 frames, split evenly rather than in proportion to clip length
# (117/174/243). The three clips are deliberately different scenario types and
# per-clip reporting is the unit throughout this project, so proportional
# allocation would leave clip_1 with five frames.
FRAMES_PER_CLIP = {'clip_1': 8, 'clip_2': 8, 'clip_3': 9}

# Reproducible sampling and presentation order: same seed -> same frames in the
# same order -> resuming a partial session lands on the correct next item.
SAMPLE_SEED = 42

# verdict is ONE string column, and actual_index another. Mixing ints and
# sentinels in a single field would force every consumer to parse two types out
# of one column and guess which it had.
CORRECT = 'correct'
WRONG_LANDMARK = 'wrong_landmark'
NOT_ON_COURT = 'not_on_court'
UNCLEAR = 'unclear'

CORRECT_KEY = 'c'
WRONG_KEY = 'w'
NOT_ON_COURT_KEY = 'n'
UNCLEAR_KEY = 'u'
STOP_KEY = 's'

VERDICT_KEYS = {
    CORRECT_KEY: CORRECT,
    WRONG_KEY: WRONG_LANDMARK,
    NOT_ON_COURT_KEY: NOT_ON_COURT,
    UNCLEAR_KEY: UNCLEAR,
}

# Recorded when the labeller judges a keypoint wrong but cannot say which
# landmark it actually sits on.
CANNOT_TELL = '?'

# parse_verdict() and parse_actual_index() outcomes.
LABEL = 'label'
STOP = 'stop'
INVALID = 'invalid'


class ParsedAnswer(NamedTuple):
    """One parsed labeller answer: an outcome of LABEL, STOP or INVALID, carrying the value or the rejection message."""

    outcome: str
    value: str


class ClipData(NamedTuple):
    """One clip's sampled frames and their detected keypoints, both indexed by frame_idx."""

    frames: dict[int, np.ndarray]
    keypoints: dict[int, list[Keypoint]]


# --- frame sampling -----------------------------------------------------------

def confident_indices(frame_keypoints: list[Keypoint]) -> list[int]:
    """The keypoint indices in one frame at or above the confidence threshold, in index order."""
    return [
        keypoint.index for keypoint in frame_keypoints
        if keypoint.confidence >= KEYPOINT_CONFIDENCE_THRESHOLD
    ]


def eligible_frames(keypoints_per_frame: list[list[Keypoint]]) -> list[int]:
    """Every frame index carrying at least one confident keypoint, which is what makes a frame verdictable."""
    # At least ONE, deliberately not the four a homography needs. Restricting
    # to well-populated frames would exclude exactly where errors are
    # suspected: a clip_3 frame carried a single confident keypoint placed
    # visibly wrong at confidence 1.0. Hard frames
    # must appear in proportion rather than being designed out of the sample.
    return [
        frame_idx for frame_idx, frame_keypoints in enumerate(keypoints_per_frame)
        if confident_indices(frame_keypoints)
    ]


def sample_frames(
    eligible_by_clip: dict[str, list[int]],
    frames_per_clip: dict[str, int] = FRAMES_PER_CLIP,
    seed: int = SAMPLE_SEED,
) -> dict[str, list[int]]:
    """Draw the per-clip quota of frames from each clip's eligible set, reproducibly under the fixed seed."""
    sampled: dict[str, list[int]] = {}
    for clip in sorted(frames_per_clip):
        eligible = eligible_by_clip.get(clip, [])
        quota = frames_per_clip[clip]
        # A clip with fewer eligible frames than its quota contributes all of
        # them rather than raising: a short or poorly-detected clip should
        # shrink the audit, not stop it.
        chosen = sorted(random.Random(f'{seed}:{clip}').sample(eligible, min(quota, len(eligible))))
        sampled[clip] = chosen
    return sampled


def build_verdict_order(
    sampled_by_clip: dict[str, list[int]],
    keypoints_by_clip: dict[str, list[list[Keypoint]]],
    seed: int = SAMPLE_SEED,
) -> list[tuple[str, int, int]]:
    """Every (clip, frame_idx, keypoint_index) to verdict, frames in a fixed shuffle across clips and keypoints in index order within a frame."""
    # SHUFFLED ON PURPOSE, matching the team GT tool rather than the possession
    # tool's chronological order. Possession is a temporal judgement needing
    # neighbouring-frame context; a keypoint's correctness is judged from the
    # single frame, so shuffling costs nothing and stops a verdict propagating
    # across visually near-identical neighbours.
    #
    # Frames are shuffled as whole units and their keypoints kept together, so
    # one render serves every verdict on that frame: at the observed 6-7
    # confident keypoints per frame, re-rendering per verdict would be roughly
    # eight times the work for the same information.
    frames = [(clip, frame_idx) for clip in sorted(sampled_by_clip) for frame_idx in sampled_by_clip[clip]]
    random.Random(seed).shuffle(frames)

    return [
        (clip, frame_idx, keypoint_index)
        for clip, frame_idx in frames
        for keypoint_index in confident_indices(keypoints_by_clip[clip][frame_idx])
    ]


# --- clip loading -------------------------------------------------------------

def load_clip_data(clip: str, sampled_frames: list[int]) -> ClipData:
    """Detect keypoints across a clip's whole length so the production cache validates, keeping only the sampled frames' images and keypoints."""
    # Detection runs over EVERY frame, not just the sampled ones, and that is
    # load-bearing rather than wasteful. CourtKeypoints' cache fingerprint
    # includes n_frames, so passing 8 frames against a 117-frame cache can
    # never validate: the stage would re-infer and then overwrite the
    # production cache with an 8-entry one, destroying the artefact
    # measure_court_keypoints.py produced and breaking this tool's own resume
    # on the next run. Passing the full clip makes the fingerprint match, so
    # the cache is SERVED and never rewritten.
    all_frames = [frame for _, frame in load_video(RAW_VIDEO_TEMPLATE.format(clip=clip))]
    detector = CourtKeypoints(
        MODEL_PATH, keypoint_confidence_threshold=KEYPOINT_CONFIDENCE_THRESHOLD,
    )
    all_keypoints = detector.run_detection(
        frames=all_frames,
        video_path=RAW_VIDEO_TEMPLATE.format(clip=clip),
        cache_path=KEYPOINTS_CACHE_TEMPLATE.format(clip=clip),
    )

    if len(all_keypoints) != len(all_frames):
        raise ValueError(
            f'Got {len(all_keypoints)} keypoint frames for {len(all_frames)} video frames in '
            f'{clip} — the two must be aligned frame-for-frame.'
        )

    # Indexed absolutely by frame_idx, never positionally: the sampled frames
    # are a scattered subset, so a positional index would silently pair a
    # frame's image with another frame's keypoints.
    frames = {idx: all_frames[idx] for idx in sampled_frames}
    keypoints = {idx: all_keypoints[idx] for idx in sampled_frames}
    megabytes = sum(frame.nbytes for frame in frames.values()) / 1e6
    print(f'{clip}: kept {len(frames)} sampled frames in memory ({megabytes:.0f} MB)')
    return ClipData(frames=frames, keypoints=keypoints)


def make_sampled_loader(sampled_by_clip: dict[str, list[int]]) -> Callable[[str], ClipData]:
    """Return a getter holding every sampled frame across all clips, loaded once on first use."""
    # Holds all three clips' SAMPLED frames rather than evicting per clip, and
    # that is the deliberate difference from possession_gt's make_clip_loader().
    # This tool's presentation order is shuffled ACROSS clips, so an evicting
    # loader would thrash -- re-decoding a whole clip every few frames. Only 25
    # frames are ever held here, against the ~1.5GB that loading all three
    # clips in full measured at, so holding them outright is the cheaper and
    # simpler choice.
    loaded: dict[str, ClipData] = {}

    def get(clip: str) -> ClipData:
        if clip not in loaded:
            loaded[clip] = load_clip_data(clip, sampled_by_clip[clip])
        return loaded[clip]

    return get


# --- answer parsing -----------------------------------------------------------

def parse_verdict(answer: str) -> ParsedAnswer:
    """Parse one primary verdict into a verdict string, a stop request, or an INVALID outcome carrying the message to show before re-presenting."""
    cleaned = answer.strip().lower()
    if cleaned == STOP_KEY:
        return ParsedAnswer(STOP, '')
    if cleaned in VERDICT_KEYS:
        return ParsedAnswer(LABEL, VERDICT_KEYS[cleaned])
    return ParsedAnswer(
        INVALID,
        f'Unrecognised verdict {answer.strip()!r} — enter '
        f'{CORRECT_KEY!r} (correct), {WRONG_KEY!r} (wrong landmark), '
        f'{NOT_ON_COURT_KEY!r} (not on court), {UNCLEAR_KEY!r} (unclear) '
        f'or {STOP_KEY!r} (stop).',
    )


def parse_actual_index(answer: str) -> ParsedAnswer:
    """Parse the follow-up naming which landmark a wrongly-placed keypoint actually sits on, accepting an index or the cannot-tell marker."""
    # Asked only after a wrong_landmark verdict. K5 asks what fraction of
    # errors are mirror confusion specifically, which is answerable only if the
    # landmark the keypoint actually sits on is recorded alongside the verdict.
    cleaned = answer.strip()
    if cleaned == CANNOT_TELL:
        return ParsedAnswer(LABEL, CANNOT_TELL)
    # isdigit() on the raw string, with no sign stripping: lstrip('-') would
    # turn '--3' into '3', pass the digit check, and then abort the whole
    # session on int('--3'). A valid index is a plain non-negative integer, so
    # anything carrying a sign is rejected here rather than parsed.
    if not cleaned.isdigit() or not 0 <= int(cleaned) < NUM_KEYPOINTS:
        return ParsedAnswer(
            INVALID,
            f'Unrecognised index {answer.strip()!r} — enter a keypoint index '
            f'0 to {NUM_KEYPOINTS - 1}, or {CANNOT_TELL!r} if you cannot tell.',
        )
    return ParsedAnswer(LABEL, str(int(cleaned)))


# --- CSV persistence and resume ----------------------------------------------

def load_labelled_rows(path: str = CSV_PATH) -> list[dict]:
    """Every verdict row from the CSV, deduplicated to the LAST row on disk for each (clip, frame_idx, keypoint_index) key, or [] if the file does not exist yet."""
    # append_verdict() stays append-only for crash safety, so a correction is a
    # new row rather than a rewrite. Last-wins dedup therefore lives here, in
    # the one function EVERY consumer (resume, progress count, scoring) reads
    # through -- a correction supersedes its earlier row in exactly one place,
    # instead of each caller reimplementing the rule.
    if not Path(path).exists():
        return []

    latest_by_key: dict[tuple[str, int, int], dict] = {}
    first_seen_order: list[tuple[str, int, int]] = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['clip'], int(row['frame_idx']), int(row['keypoint_index']))
            if key not in latest_by_key:
                first_seen_order.append(key)
            latest_by_key[key] = row
    return [latest_by_key[key] for key in first_seen_order]


def load_existing_verdicts(path: str = CSV_PATH) -> set[tuple[str, int, int]]:
    """The set of (clip, frame_idx, keypoint_index) keys already verdicted (post-dedup), or an empty set if none exist yet."""
    return {
        (row['clip'], int(row['frame_idx']), int(row['keypoint_index']))
        for row in load_labelled_rows(path)
    }


def append_verdict(
    clip: str,
    frame_idx: int,
    keypoint_index: int,
    verdict: str,
    actual_index: str,
    confidence: float,
    path: str = CSV_PATH,
    labelled_at: str | None = None,
) -> None:
    """Append one verdict row, flushing and fsyncing so a kernel death loses at most the verdict in progress."""
    # Both string columns' one-type invariant is enforced at this, the only
    # write boundary, rather than merely intended.
    if verdict not in (CORRECT, WRONG_LANDMARK, NOT_ON_COURT, UNCLEAR):
        raise ValueError(
            f'verdict must be one of {CORRECT!r}, {WRONG_LANDMARK!r}, '
            f'{NOT_ON_COURT!r} or {UNCLEAR!r}, got {verdict!r}.'
        )
    if not isinstance(actual_index, str):
        raise ValueError(f'actual_index must be a string, got {actual_index!r}.')
    # Empty for every verdict except wrong_landmark: an index recorded against
    # a correct verdict would be meaningless, and a reader cannot tell a
    # meaningless value from a real one after the fact.
    if verdict != WRONG_LANDMARK and actual_index != '':
        raise ValueError(
            f'actual_index must be empty for a {verdict!r} verdict, got {actual_index!r}.'
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Zero bytes counts as needing a header as well as absent: a crash between
    # open and the first write leaves an empty file behind, and checking
    # existence alone would then make the first real verdict the CSV header.
    needs_header = not Path(path).exists() or Path(path).stat().st_size == 0
    if labelled_at is None:
        labelled_at = datetime.now(timezone.utc).isoformat()

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(CSV_HEADER)
        writer.writerow([
            clip, frame_idx, keypoint_index, verdict, actual_index, f'{confidence:.6f}', labelled_at,
        ])
        f.flush()
        os.fsync(f.fileno())


def next_unverdicted_index(
    verdict_order: list[tuple[str, int, int]],
    verdicted: set[tuple[str, int, int]],
) -> int | None:
    """The index in verdict_order of the first keypoint with no verdict yet, or None when every one is verdicted."""
    # Scans the FULL order every time rather than continuing from wherever the
    # session happens to be: a labeller who goes back to correct an earlier
    # verdict, or who runs two sessions, must not leave a hole a forward-only
    # cursor would skip past forever. Returns None rather than
    # len(verdict_order) so callers cannot index past the end.
    for index, key in enumerate(verdict_order):
        if key not in verdicted:
            return index
    return None


def keypoint_confidence(frame_keypoints: list[Keypoint], keypoint_index: int) -> float:
    """The confidence the detector gave one keypoint index in one frame."""
    for keypoint in frame_keypoints:
        if keypoint.index == keypoint_index:
            return keypoint.confidence
    return 0.0


def run_session(
    verdict_order: list[tuple[str, int, int]],
    keypoints_by_clip: dict[str, list[list[Keypoint]]],
    show: Callable[[str, int, int], None],
    prompt: Callable[[str], str],
    path: str = CSV_PATH,
) -> int:
    """Present each unverdicted keypoint in order, rendering its frame with that keypoint highlighted, until the labeller stops or all are verdicted."""
    written = 0

    while True:
        # Re-derived from disk every iteration, so a correction appended
        # mid-session is honoured and no keypoint can be skipped.
        index = next_unverdicted_index(verdict_order, load_existing_verdicts(path))
        if index is None:
            print(f'All {len(verdict_order)} keypoints are verdicted — nothing left to do.')
            return written

        clip, frame_idx, keypoint_index = verdict_order[index]
        # Re-rendered per verdict rather than once per frame, so the point
        # under judgement can be highlighted among its neighbours. Hunting
        # for the prompted index among six or seven identical dots is slow
        # across ~148 verdicts and invites verdicting the wrong point, which
        # is invisible afterwards in ground truth that scores K3 and K5.
        show(clip, frame_idx, keypoint_index)

        confidence = keypoint_confidence(keypoints_by_clip[clip][frame_idx], keypoint_index)
        parsed = parse_verdict(prompt(
            f'[{index + 1}/{len(verdict_order)}] {clip} frame {frame_idx} '
            f'keypoint {keypoint_index} (confidence {confidence:.2f}) — '
            f'{CORRECT_KEY!r}=correct, {WRONG_KEY!r}=wrong landmark, '
            f'{NOT_ON_COURT_KEY!r}=not on court, {UNCLEAR_KEY!r}=unclear, '
            f'{STOP_KEY!r}=stop: '
        ))

        if parsed.outcome == STOP:
            print(f'Stopped. {written} verdict(s) written this session.')
            return written
        if parsed.outcome == INVALID:
            # Nothing is written, so the loop re-presents this same keypoint.
            print(parsed.value)
            continue

        actual_index = ''
        if parsed.value == WRONG_LANDMARK:
            follow_up = parse_actual_index(prompt(
                f'    Which index does keypoint {keypoint_index} actually sit on? '
                f'0-{NUM_KEYPOINTS - 1}, or {CANNOT_TELL!r} if you cannot tell: '
            ))
            if follow_up.outcome == INVALID:
                print(follow_up.value)
                continue
            actual_index = follow_up.value

        append_verdict(
            clip, frame_idx, keypoint_index, parsed.value, actual_index, confidence, path=path,
        )
        written += 1
