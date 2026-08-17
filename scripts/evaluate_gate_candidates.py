"""
evaluate_gate_candidates.py

Read-only evaluation harness for the two gate-fix candidates:

  Option A -- recalibrated budget: the baseline gate
    (BallInterpolator.filter_detections()) with its one constant swept via
    the max_travel parameter. Mechanism, metric, gap scaling,
    frozen-anchor rule and rejection semantics are untouched.

  Option B -- global trajectory selection
    (BallInterpolator.filter_detections_global_trajectory()): offline,
    non-causal dynamic program choosing the least-implausible route through
    the whole clip's detections at once.

Option B (filter_detections_global_trajectory) was adopted as the
production gate; Option A remains evaluated here as the comparator.

They are alternatives, not layers, and are never combined here. Loads the
three real caches, never runs YOLO, never modifies production code, and
writes only to data/outputs/gate_candidate_evaluation/.

Reports, in order:
  1. Calibration: the empirical distribution of consecutive
     raw-detection displacements at frame gaps 1, 2 and 3, per clip --
     replacing the assumed 25 px/frame with a measurement. A supplementary
     column converts each measured displacement into the Option B skip
     penalty that would be needed to keep motion of that size (see the
     LAMBDA note below).
  2. Per-window decision strips for W1-W5, every variant.
  3. Full-clip regression for all three clips, every variant:
     accepted counts, every decision change outside the five windows, and
     the longest run of NEW-REJECT frames anywhere in the clip.
  4. Filmstrip stills for every window and variant, production
     marker and variant marker on the same cropped, frame-labelled tile.
  5. Out-of-window sampled stills: up to
     SAMPLED_OUT_OF_WINDOW_STILLS of the decision-change frames from step 3,
     per clip per variant, rendered through the exact same tile pipeline as
     step 4 (crop_and_mark() / label_tile() / build_montage()) and saved
     individually under filmstrips/out_of_window/ so they are never confused
     with the windowed filmstrips. Sampling is deterministic (evenly spaced
     through the sorted frame list) so re-runs are comparable. This section
     always prints how many it rendered out of how many available changes,
     including the SAMPLED_OUT_OF_WINDOW_STILLS=0 case explicitly, so a
     silently-disconnected setting is impossible without a visible line
     saying so.

Full-length verification videos are deliberately NOT produced here: they
are rendered only for the shortlist, after these tables are read.

SUPPLEMENTARY LAMBDA VALUES -- a deliberate, flagged addition to the locked
configuration set, not a silent deviation. The rule that a pair of consecutive detections
d px apart is kept only while d^2 < LAMBDA holds where abandoning the path
is an option: an isolated run, or one at a clip boundary. It does NOT hold
for a bracketed run. Crossing a constant-velocity flight costs the same in
the motion term whether the in-flight points are included or jumped over
(path invariance), so when detections exist on both sides the path must
traverse the span anyway and including the flight is cheaper by LAMBDA per
point. Verified against this implementation: on a bracketed 37 px/frame
flight, LAMBDA=500 accepts 7 of 8 in-flight frames and LAMBDA=1000 accepts
all 8, against zero under the baseline gate; the same test rejects a
static off-path false-positive run at every LAMBDA from 500 up. LAMBDA=250
collapses the flight, which is the predicted over-skipping lower
bound. The locked values are therefore sound and are run unchanged
and first. {2000, 4000, 10000} are run alongside them, labelled
SUPPLEMENTARY in every table, as an upper sensitivity arm only: high LAMBDA
makes skipping expensive and pushes B towards accepting everything, so
these are expected to be worse, not better. Boundary windows (W5 sits at
the end of clip_3) are the one place the isolated-run rule genuinely binds.

Read-only with respect to every clip's cache. Does not modify
ball_interpolation.py or any other production code.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basketball.cache.cache_utils import load_cache
from basketball.detection.ball_interpolation import (
    GLOBAL_MAX_EDGE_GAP,
    MAX_BALL_TRAVEL,
    BallInterpolator,
)
from basketball.utils.box_utils import bbox_center
from basketball.utils.geometry import euclidean_distance
from basketball.utils.io_utils import load_video

CLIPS = ['clip_1', 'clip_2', 'clip_3']

CLIP_PATHS = {clip_name: f'data/raw/{clip_name}.mp4' for clip_name in CLIPS}
CACHE_PATHS = {clip_name: f'data/processed/{clip_name}/ball_detections.pkl' for clip_name in CLIPS}

OUTPUT_DIR = Path('data/outputs/gate_candidate_evaluation')

# Option A sweep. Baseline P is filter_detections() at its own
# default; A25 would duplicate it, so it is asserted equal instead of listed.
A_SWEEP = [40.0, 55.0, 70.0, 90.0]

# Option B configs as (S_MAX, LAMBDA). The first four are the locked set;
# the rest are the flagged supplementary values -- see the module docstring.
B_CONFIGS_SPEC = [(100.0, 500.0), (100.0, 250.0), (100.0, 1000.0), (150.0, 500.0)]
B_CONFIGS_SUPPLEMENTARY = [(100.0, 2000.0), (100.0, 4000.0), (100.0, 10000.0),  (150.0, 1000.0), (200.0, 1000.0)]

# (label, clip, first frame, last frame, event). Indices are
# cache indices, identical to the 30fps render frame numbers.
WINDOWS = [
    ('W1', 'clip_2', 12, 45, 'second pass and aftermath'),
    ('W2', 'clip_2', 106, 127, 'fourth pass and aftermath'),
    ('W3', 'clip_3', 78, 102, 'interception; static head FP currently accepted'),
    ('W4', 'clip_3', 163, 188, 'forward pass (~37px/frame flight, currently rejected)'),
    ('W5', 'clip_3', 220, 242, 'post-dunk drop; largest divergence, not yet verified against footage'),
]

CALIBRATION_GAPS = [1, 2, 3]
CALIBRATION_PERCENTILES = [50, 90, 95, 99]

CROP_SIZE = 260  # px square cut from each frame, centred on that frame's raw detection
MONTAGE_COLUMNS = 6
TILE_LABEL_HEIGHT = 22
MONTAGE_HEADER_HEIGHT = 86

RAW_COLOUR = (0, 0, 255)        # BGR red -- raw (pre-gate) detection box
PRODUCTION_COLOUR = (255, 160, 0)  # BGR blue -- accepted by production P
VARIANT_COLOUR = (0, 255, 0)    # BGR green -- accepted by the variant

# Decision changes OUTSIDE the windows also require sampled visual checks.
# render_out_of_window_stills() samples up to this many per clip per
# variant from the changes already computed by the full-clip regression;
# 0 skips rendering (but still prints that it was skipped, see that function).
SAMPLED_OUT_OF_WINDOW_STILLS = 25


# Inlined from the evaluation harness this script originally shared a folder with.
def find_consecutive_runs(frame_indices: list[int]) -> list[tuple[int, int]]:
    """Groups a sorted list of frame indices into (start, end) runs of consecutive frames."""
    if not frame_indices:
        return []
    runs: list[tuple[int, int]] = []
    run_start = frame_indices[0]
    run_end = frame_indices[0]
    for idx in frame_indices[1:]:
        if idx == run_end + 1:
            run_end = idx
        else:
            runs.append((run_start, run_end))
            run_start = run_end = idx
    runs.append((run_start, run_end))
    return runs


def variant_file_slug(display_name: str) -> str:
    """Turn a variant's display name into something safe for a filename."""
    return (
        display_name.replace('(', '_').replace(')', '')
        .replace(',', '_').replace('.', '').replace(' ', '')
    )


def build_variants(interpolator: BallInterpolator) -> list[dict]:
    """Assemble every variant to evaluate as a list of records carrying its display name, whether it is from the locked set, and a runner mapping raw detections to gated output."""
    variants: list[dict] = [{
        'name': 'P',
        'kind': 'baseline',
        'spec': True,
        'note': f'production baseline (max_travel={MAX_BALL_TRAVEL:g})',
        'run': lambda raw: interpolator.filter_detections(raw),
    }]

    for value in A_SWEEP:
        variants.append({
            'name': f'A{value:g}',
            'kind': 'A',
            'spec': True,
            'note': f'recalibrated budget, max_travel={value:g}',
            'run': (lambda raw, v=value: interpolator.filter_detections(raw, max_travel=v)),
        })

    for configs, is_spec in ((B_CONFIGS_SPEC, True), (B_CONFIGS_SUPPLEMENTARY, False)):
        for s_max, lam in configs:
            variants.append({
                'name': f'B({s_max:g},{lam:g})',
                'kind': 'B',
                'spec': is_spec,
                'note': (
                    f'global trajectory, S_MAX={s_max:g}, LAMBDA={lam:g} '
                    f'(isolated-run free speed {lam ** 0.5:.1f} px/frame)'
                ),
                's_max': s_max,
                'lambda': lam,
                'run': (
                    lambda raw, s=s_max, l=lam: interpolator.filter_detections_global_trajectory(
                        raw, max_speed=s, skip_penalty=l,
                    )
                ),
            })

    return variants


def accepted_set(gated: list[dict]) -> set[int]:
    """Frame indices accepted by a gated detection list."""
    return {i for i, frame in enumerate(gated) if 1 in frame}


def classify(raw_present: bool, in_p: bool, in_v: bool) -> str:
    """Classify one frame's production-vs-variant decision."""
    if not raw_present or in_p == in_v:
        return 'SAME'

    return 'NEW-ACCEPT' if in_v else 'NEW-REJECT'


def measure_displacements(raw: list[dict]) -> dict[int, list[float]]:
    """Collect displacements between consecutive raw detections, bucketed by the frame gap separating them."""
    nodes: list[tuple[int, float, float]] = []
    for i, frame in enumerate(raw):
        if 1 in frame:
            cx, cy = bbox_center(*frame[1].bbox)
            nodes.append((i, cx, cy))

    buckets: dict[int, list[float]] = {gap: [] for gap in CALIBRATION_GAPS}

    for (f_a, x_a, y_a), (f_b, x_b, y_b) in zip(nodes, nodes[1:]):
        gap = f_b - f_a
        if gap in buckets:
            buckets[gap].append(euclidean_distance((x_a, y_a), (x_b, y_b)))

    return buckets


def report_calibration(caches: dict[str, list[dict]]) -> list[dict]:
    """Print and return the per-clip, per-gap displacement distribution that replaces the assumed 25 px/frame with a measurement."""
    print('=' * 100)
    print('1. CALIBRATION -- measured consecutive raw-detection displacements')
    print('=' * 100)
    print(
        '\nDisplacement is in px between two consecutive raw detections; per-frame speed is that\n'
        'divided by the gap, i.e. the quantity the production gate compares against max_travel.\n'
        'LAMBDA-to-keep is supplementary: the Option B skip penalty needed to keep a pair that far\n'
        'apart (speed^2), included because it converts this measurement directly into a B setting.\n'
    )

    rows: list[dict] = []
    header = (
        f'{"clip":>8} | {"gap":>3} | {"n":>5} | {"p50":>7} {"p90":>7} {"p95":>7} {"p99":>7} {"max":>7} | '
        f'{"p95 speed":>9} {"max speed":>9} | {"LAMBDA to keep p95":>18}'
    )
    print(header)
    print('-' * len(header))

    for clip_name in CLIPS:
        buckets = measure_displacements(caches[clip_name])

        for gap in CALIBRATION_GAPS:
            values = buckets[gap]
            if not values:
                print(f'{clip_name:>8} | {gap:>3} | {0:>5} | (no consecutive pairs at this gap)')
                rows.append({'clip': clip_name, 'gap': gap, 'n': 0})
                continue

            array = np.array(values, dtype=float)
            p50, p90, p95, p99 = (float(np.percentile(array, q)) for q in CALIBRATION_PERCENTILES)
            maximum = float(array.max())
            p95_speed = p95 / gap
            max_speed = maximum / gap

            print(
                f'{clip_name:>8} | {gap:>3} | {len(values):>5} | '
                f'{p50:>7.1f} {p90:>7.1f} {p95:>7.1f} {p99:>7.1f} {maximum:>7.1f} | '
                f'{p95_speed:>9.1f} {max_speed:>9.1f} | {p95_speed ** 2:>18.0f}'
            )

            rows.append({
                'clip': clip_name, 'gap': gap, 'n': len(values),
                'p50': round(p50, 2), 'p90': round(p90, 2), 'p95': round(p95, 2),
                'p99': round(p99, 2), 'max': round(maximum, 2),
                'p95_speed_px_per_frame': round(p95_speed, 2),
                'max_speed_px_per_frame': round(max_speed, 2),
                'lambda_to_keep_p95': round(p95_speed ** 2, 1),
            })

    print(
        f'\nFor reference: the production cap is max_travel={MAX_BALL_TRAVEL:g} px/frame '
        f'(equivalent LAMBDA {MAX_BALL_TRAVEL ** 2:g}).'
    )
    print(
        'If the measured gap-1 p95/max speeds above sit well above 25, the inherited constant is\n'
        'confirmed as noise suppression rather than a model of ball motion, which is the\n'
        "parameter-provenance point this table exists to settle for the dissertation.\n"
    )

    return rows


def report_windows(
    caches: dict[str, list[dict]],
    gated: dict[str, dict[str, list[dict]]],
    variants: list[dict],
    active_windows: list[tuple],
) -> list[dict]:
    """Print and return the per-frame decision strip for every window and variant."""
    print('=' * 100)
    print('2. PER-WINDOW DECISION STRIPS')
    print('=' * 100)

    rows: list[dict] = []

    for label, clip_name, start, end, event in active_windows:
        raw = caches[clip_name]
        p_accepted = accepted_set(gated[clip_name]['P'])

        print(f'\n{"=" * 92}')
        print(f'{label}  {clip_name} frames {start}-{end}  --  {event}')
        print('=' * 92)

        for variant in variants:
            if variant['name'] == 'P':
                continue

            v_accepted = accepted_set(gated[clip_name][variant['name']])
            tag = '' if variant['spec'] else '  [SUPPLEMENTARY]'
            print(f'\n  {variant["name"]}  ({variant["note"]}){tag}')

            changes = 0
            strip_parts: list[str] = []

            for frame_num in range(start, end + 1):
                raw_present = 1 in raw[frame_num]
                in_p = frame_num in p_accepted
                in_v = frame_num in v_accepted
                classification = classify(raw_present, in_p, in_v)

                if classification != 'SAME':
                    changes += 1

                if not raw_present:
                    symbol = '.'
                elif classification == 'NEW-ACCEPT':
                    symbol = '+'
                elif classification == 'NEW-REJECT':
                    symbol = '-'
                elif in_p:
                    symbol = '#'
                else:
                    symbol = 'o'

                strip_parts.append(symbol)

                rows.append({
                    'window': label,
                    'clip': clip_name,
                    'variant': variant['name'],
                    'spec_variant': variant['spec'],
                    'frame': frame_num,
                    'raw_present': raw_present,
                    'accepted_by_P': in_p,
                    'accepted_by_variant': in_v,
                    'classification': classification,
                })

            print(f'    frames {start}-{end}: {"".join(strip_parts)}')
            print(f'    legend: # both accept | o both reject | + NEW-ACCEPT | - NEW-REJECT | . no raw detection')
            print(f'    decision changes in window: {changes}')

            new_accepts = [r['frame'] for r in rows if r['window'] == label and r['variant'] == variant['name'] and r['classification'] == 'NEW-ACCEPT']
            new_rejects = [r['frame'] for r in rows if r['window'] == label and r['variant'] == variant['name'] and r['classification'] == 'NEW-REJECT']
            if new_accepts:
                print(f'    NEW-ACCEPT frames: {new_accepts}')
            if new_rejects:
                print(f'    NEW-REJECT frames: {new_rejects}')

    return rows


def report_fullclip(
    caches: dict[str, list[dict]],
    gated: dict[str, dict[str, list[dict]]],
    variants: list[dict],
    active_windows: list[tuple],
) -> tuple[list[dict], list[dict]]:
    """Print and return the full-clip regression summary and the list of decision changes outside the five windows."""
    print('\n' + '=' * 100)
    print('3. FULL-CLIP REGRESSION')
    print('=' * 100)

    window_frames: dict[str, set[int]] = {clip_name: set() for clip_name in CLIPS}
    for _, clip_name, start, end, _ in active_windows:
        window_frames[clip_name].update(range(start, end + 1))

    summary_rows: list[dict] = []
    change_rows: list[dict] = []

    for clip_name in CLIPS:
        raw = caches[clip_name]
        raw_present_count = sum(1 for frame in raw if 1 in frame)
        p_accepted = accepted_set(gated[clip_name]['P'])

        print(f'\n{clip_name}: {len(raw)} frames, {raw_present_count} raw detections present, '
              f'{len(p_accepted)} accepted by P')
        header = (
            f'  {"variant":>14} | {"accepted":>8} | {"vs P":>6} | {"NEW-ACC":>7} | {"NEW-REJ":>7} | '
            f'{"out-of-window changes":>21} | {"longest NEW-REJ run":>19}'
        )
        print(header)
        print('  ' + '-' * (len(header) - 2))

        for variant in variants:
            v_accepted = accepted_set(gated[clip_name][variant['name']])

            new_accepts = sorted(v_accepted - p_accepted)
            new_rejects = sorted(p_accepted - v_accepted)

            reject_runs = find_consecutive_runs(new_rejects)
            longest_run = max((end - start + 1 for start, end in reject_runs), default=0)
            longest_run_desc = 'none'
            if reject_runs:
                start, end = max(reject_runs, key=lambda r: r[1] - r[0])
                longest_run_desc = f'{longest_run} @ {start}-{end}'

            out_of_window = [
                f for f in sorted(set(new_accepts) | set(new_rejects))
                if f not in window_frames[clip_name]
            ]

            for frame_num in out_of_window:
                change_rows.append({
                    'clip': clip_name,
                    'variant': variant['name'],
                    'spec_variant': variant['spec'],
                    'frame': frame_num,
                    'classification': 'NEW-ACCEPT' if frame_num in v_accepted else 'NEW-REJECT',
                })

            tag = '' if variant['spec'] else ' *'
            print(
                f'  {variant["name"] + tag:>14} | {len(v_accepted):>8} | '
                f'{len(v_accepted) - len(p_accepted):>+6} | {len(new_accepts):>7} | {len(new_rejects):>7} | '
                f'{len(out_of_window):>21} | {longest_run_desc:>19}'
            )

            summary_rows.append({
                'clip': clip_name,
                'variant': variant['name'],
                'spec_variant': variant['spec'],
                'frames': len(raw),
                'raw_present': raw_present_count,
                'accepted': len(v_accepted),
                'accepted_minus_P': len(v_accepted) - len(p_accepted),
                'new_accept': len(new_accepts),
                'new_reject': len(new_rejects),
                'out_of_window_changes': len(out_of_window),
                'longest_new_reject_run': longest_run,
            })

        print('  (* = supplementary variant, not in the locked set)')

    print('\n  Out-of-window decision changes are listed in full in out_of_window_changes.csv.')
    print('  These also require sampled visual checks -- see section 5 below for the')
    print(f'  {SAMPLED_OUT_OF_WINDOW_STILLS}-per-clip-per-variant sample actually rendered this run.')

    return summary_rows, change_rows


def crop_and_mark(
    frame: np.ndarray,
    raw_detection,
    accepted_by_p: bool,
    accepted_by_variant: bool,
    size: int = CROP_SIZE,
) -> np.ndarray:
    """Cut a fixed-size square around this frame's raw detection and draw the raw box plus whichever of the production and variant markers accepted it."""
    height, width = frame.shape[:2]

    if raw_detection is not None:
        focus_x, focus_y = bbox_center(*raw_detection.bbox)
    else:
        focus_x, focus_y = width / 2.0, height / 2.0

    half = size // 2
    x0 = max(0, min(int(round(focus_x)) - half, max(0, width - size)))
    y0 = max(0, min(int(round(focus_y)) - half, max(0, height - size)))

    tile = frame[y0:y0 + size, x0:x0 + size].copy()
    if tile.shape[0] != size or tile.shape[1] != size:
        padded = np.zeros((size, size, 3), dtype=frame.dtype)
        padded[:tile.shape[0], :tile.shape[1]] = tile
        tile = padded

    if raw_detection is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in raw_detection.bbox)
        cv2.rectangle(tile, (x1 - x0, y1 - y0), (x2 - x0, y2 - y0), RAW_COLOUR, 1)

        cx, cy = int(round(focus_x)) - x0, int(round(focus_y)) - y0
        if accepted_by_p:
            cv2.circle(tile, (cx, cy), 11, PRODUCTION_COLOUR, 2)
        if accepted_by_variant:
            cv2.rectangle(tile, (cx - 17, cy - 17), (cx + 17, cy + 17), VARIANT_COLOUR, 2)

    return tile


def label_tile(tile: np.ndarray, frame_num: int, classification: str, raw_present: bool) -> np.ndarray:
    """Stack a caption bar under one filmstrip tile carrying its frame number and decision classification."""
    size = tile.shape[1]
    bar = np.zeros((TILE_LABEL_HEIGHT, size, 3), dtype=tile.dtype)

    if classification == 'NEW-ACCEPT':
        colour = VARIANT_COLOUR
    elif classification == 'NEW-REJECT':
        colour = RAW_COLOUR
    else:
        colour = (200, 200, 200)

    text = f'f{frame_num} {classification}' if raw_present else f'f{frame_num} no-det'
    cv2.putText(bar, text, (4, TILE_LABEL_HEIGHT - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    return np.vstack([tile, bar])


def build_montage(
    tiles: list[np.ndarray],
    header_lines: list[str],
    columns: int = MONTAGE_COLUMNS,
    min_width: int = 0,
) -> np.ndarray:
    """Tile labelled crops into a grid under a header band describing the window, variant and marker legend. min_width widens the canvas beyond columns * tile_w when there are too few columns for the header/legend text to fit (e.g. the single-tile out-of-window stills), without affecting the normal multi-column montages that already have room."""
    if not tiles:
        raise ValueError('build_montage() requires at least one tile.')

    tile_h, tile_w = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    width = max(columns * tile_w, min_width)

    canvas = np.zeros((MONTAGE_HEADER_HEIGHT + rows * tile_h, width, 3), dtype=tiles[0].dtype)

    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        y = MONTAGE_HEADER_HEIGHT + r * tile_h
        x = c * tile_w
        canvas[y:y + tile_h, x:x + tile_w] = tile

    for line_index, line in enumerate(header_lines[:2]):
        y = 24 + line_index * 24
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    legend_y = MONTAGE_HEADER_HEIGHT - 12
    cv2.putText(canvas, 'red box = raw detection', (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RAW_COLOUR, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'blue circle = P accepted', (240, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PRODUCTION_COLOUR, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'green square = variant accepted', (480, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, VARIANT_COLOUR, 1, cv2.LINE_AA)

    return canvas


def render_filmstrips(
    caches: dict[str, list[dict]],
    gated: dict[str, dict[str, list[dict]]],
    variants: list[dict],
    active_windows: list[tuple],
    frames_by_clip: dict[str, list[np.ndarray]],
) -> int:
    """Render one filmstrip montage per window per variant."""
    print('\n' + '=' * 100)
    print('4. FILMSTRIP STILLS')
    print('=' * 100)

    strips_dir = OUTPUT_DIR / 'filmstrips'
    strips_dir.mkdir(parents=True, exist_ok=True)

    written = 0

    for label, clip_name, start, end, event in active_windows:
        raw = caches[clip_name]
        frames = frames_by_clip[clip_name]
        p_accepted = accepted_set(gated[clip_name]['P'])

        for variant in variants:
            v_accepted = accepted_set(gated[clip_name][variant['name']])

            tiles = []
            for frame_num in range(start, end + 1):
                raw_detection = raw[frame_num].get(1)
                raw_present = raw_detection is not None
                in_p = frame_num in p_accepted
                in_v = frame_num in v_accepted

                tile = crop_and_mark(frames[frame_num], raw_detection, in_p, in_v)
                tiles.append(label_tile(tile, frame_num, classify(raw_present, in_p, in_v), raw_present))

            tag = '' if variant['spec'] else '  [SUPPLEMENTARY]'
            montage = build_montage(tiles, [
                f'{label} {clip_name} frames {start}-{end} -- {event}',
                f'variant {variant["name"]}: {variant["note"]}{tag}',
            ])

            out_path = strips_dir / f'{label}_{clip_name}_{variant_file_slug(variant["name"])}.png'
            cv2.imwrite(str(out_path), montage)
            written += 1

        print(f'  {label} ({clip_name} {start}-{end}): {len(variants)} filmstrips written')

    print(f'\n  {written} filmstrips saved to {strips_dir}/')

    return written


def sample_out_of_window_frames(frame_numbers: list[int], sample_size: int) -> list[int]:
    """Deterministically pick up to sample_size frames, evenly spaced through the sorted list, so re-runs are comparable. Returns every frame if fewer exist than sample_size, and an empty list if sample_size is 0 or there are no frames."""
    if sample_size <= 0 or not frame_numbers:
        return []

    if len(frame_numbers) <= sample_size:
        return list(frame_numbers)

    step = len(frame_numbers) / sample_size
    indices = sorted({int(i * step) for i in range(sample_size)})

    return [frame_numbers[i] for i in indices]


def render_out_of_window_stills(
    caches: dict[str, list[dict]],
    gated: dict[str, dict[str, list[dict]]],
    variants: list[dict],
    change_rows: list[dict],
    frames_by_clip: dict[str, list[np.ndarray]],
    sample_size: int,
) -> tuple[int, int]:
    """
    Renders up to sample_size of the out-of-window decision-change frames
    (the full-clip regression's change_rows) per clip per variant, via the
    exact same crop_and_mark() / label_tile() / build_montage() pipeline the
    windowed filmstrips use -- no second rendering path -- saved individually
    under filmstrips/out_of_window/ so they are never confused with the
    windowed stills. Returns (stills_written, total_changes_available), and
    always prints that count, including the sample_size=0 case, so this
    section cannot silently do nothing.
    """
    print('\n' + '=' * 100)
    print('5. OUT-OF-WINDOW SAMPLED STILLS')
    print('=' * 100)

    total_available = len(change_rows)

    if sample_size <= 0:
        print(f'\n  SAMPLED_OUT_OF_WINDOW_STILLS = 0 -- 0 requested, none rendered '
              f'({total_available} out-of-window changes available in out_of_window_changes.csv).')
        return 0, total_available

    out_dir = OUTPUT_DIR / 'filmstrips' / 'out_of_window'
    out_dir.mkdir(parents=True, exist_ok=True)

    by_clip_variant: dict[tuple[str, str], list[int]] = {}
    for row in change_rows:
        by_clip_variant.setdefault((row['clip'], row['variant']), []).append(row['frame'])

    total_written = 0

    for clip_name in CLIPS:
        if clip_name not in frames_by_clip:
            # Out-of-window changes can occur in a clip that has no active
            # window at all (clip_1 has none) -- load its frames
            # here rather than assuming render_filmstrips() already did.
            if not Path(CLIP_PATHS[clip_name]).exists():
                raise FileNotFoundError(
                    f'Expected video not found at {CLIP_PATHS[clip_name]}, needed to render out-of-window stills.'
                )
            frames_by_clip[clip_name] = [frame for _, frame in load_video(CLIP_PATHS[clip_name])]

        frames = frames_by_clip[clip_name]
        raw = caches[clip_name]
        p_accepted = accepted_set(gated[clip_name]['P'])

        for variant in variants:
            if variant['name'] == 'P':
                continue

            frame_numbers = sorted(by_clip_variant.get((clip_name, variant['name']), []))
            sampled = sample_out_of_window_frames(frame_numbers, sample_size)
            v_accepted = accepted_set(gated[clip_name][variant['name']])
            tag = '' if variant['spec'] else '  [SUPPLEMENTARY]'

            for frame_num in sampled:
                raw_detection = raw[frame_num].get(1)
                raw_present = raw_detection is not None
                in_p = frame_num in p_accepted
                in_v = frame_num in v_accepted

                tile = crop_and_mark(frames[frame_num], raw_detection, in_p, in_v)
                tile = label_tile(tile, frame_num, classify(raw_present, in_p, in_v), raw_present)

                montage = build_montage(
                    [tile],
                    [f'OUT-OF-WINDOW  {clip_name} frame {frame_num}',
                     f'variant {variant["name"]}: {variant["note"]}{tag}'],
                    columns=1,
                    min_width=760,
                )

                out_path = out_dir / f'{clip_name}_{variant_file_slug(variant["name"])}_f{frame_num}.png'
                cv2.imwrite(str(out_path), montage)
                total_written += 1

            print(f'  {clip_name} {variant["name"]}: {len(sampled)} of {len(frame_numbers)} out-of-window changes sampled')

    print(
        f'\n  out-of-window: {total_written} stills sampled from {total_available} total changes, '
        f'saved to {out_dir}/'
    )

    return total_written, total_available


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of uniform dict rows to CSV, skipping empty result sets."""
    if not rows:
        print(f'  (nothing to write for {path.name})')
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'  wrote {len(rows)} rows to {path}')


def main() -> None:
    started = time.time()

    for clip_name in CLIPS:
        if not Path(CACHE_PATHS[clip_name]).exists():
            raise FileNotFoundError(
                f'Expected raw ball cache not found at {CACHE_PATHS[clip_name]}. '
                'This script must run where the real cached detections exist (JupyterHub).'
            )

    caches = {clip_name: load_cache(CACHE_PATHS[clip_name]) for clip_name in CLIPS}
    for clip_name in CLIPS:
        print(f'loaded {clip_name}: {len(caches[clip_name])} frames cached')

    # Drop any window whose indices exceed its clip, loudly, rather than
    # losing the whole run to one bad index.
    active_windows = []
    for window in WINDOWS:
        label, clip_name, start, end, _ = window
        if end >= len(caches[clip_name]):
            print(
                f'\nWARNING: {label} requests {clip_name} frames {start}-{end} but that clip has only '
                f'{len(caches[clip_name])} cached frames -- SKIPPING this window. Every other section '
                'still runs; re-check the index against the cache before trusting the window table.'
            )
            continue
        active_windows.append(window)

    interpolator = BallInterpolator()
    variants = build_variants(interpolator)

    print(f'\n{len(variants)} variants: ' + ', '.join(
        v['name'] + ('' if v['spec'] else '*') for v in variants
    ) + '   (* = supplementary, not in the locked set)')

    gated: dict[str, dict[str, list[dict]]] = {}
    for clip_name in CLIPS:
        raw = caches[clip_name]
        gated[clip_name] = {variant['name']: variant['run'](raw) for variant in variants}

        # Option A's parameterisation must not have changed
        # filter_detections()'s default behaviour -- the one production-code
        # edit made for this evaluation, verified here against every real
        # clip rather than assumed.
        explicit_default = interpolator.filter_detections(raw, max_travel=MAX_BALL_TRAVEL)
        if accepted_set(explicit_default) != accepted_set(gated[clip_name]['P']):
            raise AssertionError(
                f'{clip_name}: filter_detections(raw, max_travel=MAX_BALL_TRAVEL) disagrees with '
                'filter_detections(raw) -- the Option A parameterisation has changed the production '
                'gate\'s default behaviour, nothing below can be trusted.'
            )

        # Option B's trace shares the gate's solver, so it cannot drift; the
        # established discipline is to verify anyway before using its numbers.
        for variant in variants:
            if variant['kind'] != 'B':
                continue

            trace = interpolator.trace_global_trajectory_decisions(
                raw, max_speed=variant['s_max'], skip_penalty=variant['lambda'],
            )
            accepted = accepted_set(gated[clip_name][variant['name']])
            for i, frame in enumerate(raw):
                if 1 not in frame:
                    continue
                if i not in trace or trace[i]['accepted'] != (i in accepted):
                    raise AssertionError(
                        f'{clip_name} frame {i}: trace_global_trajectory_decisions() disagrees with '
                        f'filter_detections_global_trajectory() for {variant["name"]} -- '
                        'instrumentation is out of sync.'
                    )

    print('verification: PASS -- Option A default behaviour unchanged; every Option B trace agrees with its gate.\n')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration_rows = report_calibration(caches)
    window_rows = report_windows(caches, gated, variants, active_windows)
    summary_rows, change_rows = report_fullclip(caches, gated, variants, active_windows)

    clips_needing_frames = sorted({clip_name for _, clip_name, _, _, _ in active_windows})
    frames_by_clip: dict[str, list[np.ndarray]] = {}
    for clip_name in clips_needing_frames:
        if not Path(CLIP_PATHS[clip_name]).exists():
            raise FileNotFoundError(
                f'Expected video not found at {CLIP_PATHS[clip_name]}, needed to render filmstrips.'
            )
        frames_by_clip[clip_name] = [frame for _, frame in load_video(CLIP_PATHS[clip_name])]
        if len(frames_by_clip[clip_name]) != len(caches[clip_name]):
            print(
                f'WARNING: {clip_name} video has {len(frames_by_clip[clip_name])} frames but its cache has '
                f'{len(caches[clip_name])} -- cache may be stale relative to the clip on disk.'
            )

    render_filmstrips(caches, gated, variants, active_windows, frames_by_clip)
    render_out_of_window_stills(
        caches, gated, variants, change_rows, frames_by_clip, SAMPLED_OUT_OF_WINDOW_STILLS,
    )

    print('\n' + '=' * 100)
    print('CSV OUTPUT')
    print('=' * 100)
    write_csv(OUTPUT_DIR / 'calibration_displacements.csv', calibration_rows)
    write_csv(OUTPUT_DIR / 'window_decisions.csv', window_rows)
    write_csv(OUTPUT_DIR / 'fullclip_summary.csv', summary_rows)
    write_csv(OUTPUT_DIR / 'out_of_window_changes.csv', change_rows)

    print(f'\nCompleted in {time.time() - started:.1f}s. All output under {OUTPUT_DIR}/')
    print(
        '\nREMINDER: these tables narrow the field; they do not decide it. The\n'
        'shortlist still needs full-length verification videos and footage checks of every in-window\n'
        'NEW-ACCEPT before any variant is adoptable.'
    )


if __name__ == '__main__':
    main()
