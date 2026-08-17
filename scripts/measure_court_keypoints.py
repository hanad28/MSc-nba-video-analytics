"""
measure_court_keypoints.py

Runs Stage 7's three label-free measurements (court keypoint spec, sections
4.2, 4.3 and 4.5) over the three evaluation clips: sufficiency, leave-one-out
reprojection residuals and temporal stability. Runs on JupyterHub; it needs
the clips, the checkpoint and the GPU stack.
"""
from __future__ import annotations

import csv
import itertools
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

# Running this file directly (`python scripts/measure_court_keypoints.py`) puts
# scripts/ on sys.path[0], not the repo root, so `basketball` would not be
# importable. Insert the repo root explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basketball.keypoints.court_keypoints import CourtKeypoints, Keypoint
from basketball.keypoints.court_template import NUM_KEYPOINTS, TEMPLATE_POINTS_M
from basketball.utils.io_utils import load_video

CLIPS = ('clip_1', 'clip_2', 'clip_3')

CLIP_PATH_TEMPLATE = 'data/raw/{clip}.mp4'
KEYPOINTS_CACHE_TEMPLATE = 'data/processed/{clip}/keypoints.pkl'
MODEL_PATH = 'models/keypoints.pt'

OUTPUT_DIR = Path('data/outputs/keypoint_evaluation')
SUFFICIENCY_CSV = OUTPUT_DIR / 'sufficiency.csv'
REPROJECTION_CSV = OUTPUT_DIR / 'reprojection.csv'
STABILITY_CSV = OUTPUT_DIR / 'stability.csv'
SUMMARY_TXT = OUTPUT_DIR / 'summary.txt'

# The PER-KEYPOINT confidence threshold, gating individual landmarks. Distinct
# from CourtKeypoints' instance threshold, which gates whether a court is
# detected at all. Defined once so it cannot drift between the three measures.
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5

# Homography needs 4 correspondences, so a held-out fit needs 5 confident
# points: 4 to fit plus 1 to hold out.
MIN_SUFFICIENT_KEYPOINTS = 4
MIN_REPROJECTION_KEYPOINTS = 5

# Buckets for the confident-keypoint count, reported separately.
REPROJECTION_BUCKETS = (5, 6, 7, 8)
OVERFLOW_BUCKET = '9+'

# Same constant and rationale as basketball/homography/transform.py's
# MIN_TRIANGLE_AREA_M2.
MIN_TRIANGLE_AREA_M2 = 0.01


def print_table(title: str, headers: list[str], rows: list[list[str]], lines: list[str]) -> None:
    """Print one aligned results table and append it to the summary lines."""
    widths = (
        [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)] if rows
        else [len(header) for header in headers]
    )
    rendered = [f'\n{title}', '  '.join(header.ljust(width) for header, width in zip(headers, widths))]
    for row in rows:
        rendered.append('  '.join(str(cell).ljust(width) for cell, width in zip(row, widths)))

    for line in rendered:
        print(line)
    lines.extend(rendered)


def confident_keypoints(frame_keypoints: list[Keypoint]) -> list[Keypoint]:
    """Return the keypoints in one frame at or above the per-keypoint confidence threshold."""
    return [
        keypoint for keypoint in frame_keypoints
        if keypoint.confidence >= KEYPOINT_CONFIDENCE_THRESHOLD
    ]


def below_threshold_runs(confident_counts: list[int]) -> list[int]:
    """Return the lengths of each maximal run of consecutive frames holding fewer than four confident keypoints."""
    # Runs rather than a bare count: first inspection found the sub-threshold
    # frames arrive in contiguous windows during camera movement rather than
    # scattered, and that shape is what decides whether carrying the last valid
    # homography is viable. A count alone hides it.
    runs: list[int] = []
    current = 0
    for count in confident_counts:
        if count < MIN_SUFFICIENT_KEYPOINTS:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def bucket_for(confident_count: int) -> str:
    """Return the reporting bucket label for a frame's confident-keypoint count."""
    # Buckets exist so residuals are never pooled into one headline: a
    # 5-keypoint frame fits 4 points to an 8-degree-of-freedom transform, an
    # exact fit whose held-out residual is highly sensitive to error in those
    # four, while an 8-keypoint frame is over-determined and far more stable.
    # Pooling the two would mix different measurements.
    if confident_count >= REPROJECTION_BUCKETS[-1] + 1:
        return OVERFLOW_BUCKET
    return str(confident_count)


def is_collinear(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    """Return whether three template positions are collinear, by triangle area."""
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0 < MIN_TRIANGLE_AREA_M2


def has_general_position_quadruple(template_points: list[tuple[float, float]]) -> bool:
    """Return whether some four of the given template positions have no three collinear, which is what determines the homography."""
    # Same rule as the identically-named function in
    # basketball/homography/transform.py. Verified here on exact synthetic
    # data: {0, 1, 2, 8, 9} contains the collinear triple (0, 1, 2) and still
    # recovers the transform to within 6e-7 m, while [0,1,2,8] yields
    # determinant 0.046 -- nowhere near singular, so neither the finite check
    # nor the determinant threshold catches it -- and a 1.9 m residual where
    # the true answer is zero.
    for quad in itertools.combinations(template_points, 4):
        if not any(is_collinear(*triple) for triple in itertools.combinations(quad, 3)):
            return True
    return False


def fit_homography(
    image_points: list[tuple[float, float]],
    template_points: list[tuple[float, float]],
) -> np.ndarray | None:
    """Fit an image-to-template homography, returning None when the fit fails or is degenerate."""
    source = np.array(image_points, dtype=np.float64).reshape(-1, 1, 2)
    target = np.array(template_points, dtype=np.float64).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(source, target, method=0)

    # findHomography returns None outright on failure, and a near-singular
    # matrix when the fitting points are close to collinear -- a real
    # possibility here, since several keypoints lie along one baseline. A
    # degenerate fit must never produce a residual, so it is rejected rather
    # than used.
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        return None
    return matrix


def hartley_normalise(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Translate points so their centroid is at the origin and scale them so the mean distance from it is root two."""
    count = len(points)
    centroid_x = sum(x for x, _ in points) / count
    centroid_y = sum(y for _, y in points) / count
    centred = [(x - centroid_x, y - centroid_y) for x, y in points]

    mean_distance = sum(np.hypot(x, y) for x, y in centred) / count
    if mean_distance <= 0.0:
        # Every point coincident: no scale to recover, and dividing would
        # produce NaN rather than a usable measure.
        return centred
    scale = np.sqrt(2.0) / mean_distance
    return [(x * scale, y * scale) for x, y in centred]


def dlt_conditioning(
    image_points: list[tuple[float, float]],
    template_points: list[tuple[float, float]],
) -> float:
    """Return the ratio of the smallest to the largest non-null singular value of the Hartley-normalised DLT matrix for one fit."""
    # Recorded, never used to reject. The determinant guard only fires at float
    # precision, and the template-side collinearity check cannot see image-side
    # near-collinearity: points in general position on the court can project to
    # nearly a line under an oblique camera. Measuring the conditioning makes
    # that question answerable from evidence -- whether ill-conditioned fits
    # occur in this footage at all, and whether they track large residuals --
    # so any future threshold can be chosen against data rather than guessed.
    #
    # The DLT solution is the null vector, so the smallest singular value is
    # the residual of the homogeneous system rather than a conditioning
    # measure. The ratio reported is sigma[7] / sigma[0], the last non-null
    # value over the largest: near zero means the second-smallest direction is
    # also nearly null, so the solution is barely distinguished from a family.
    # Hartley normalisation first, on BOTH sets. Raw image coordinates run to
    # ~10^3 pixels while template coordinates run to ~10^1 metres, so an
    # unnormalised DLT matrix has wildly unequal column magnitudes and its
    # singular value ratio is depressed by that disparity rather than by
    # geometry. cv2.findHomography normalises internally, so without this the
    # reported number would describe a differently-scaled system from the one
    # actually solved. Measured: identical geometry scored 2.8e-5 at one image
    # scale and 7.6e-7 at 1000x that, a 37-fold swing from units alone.
    # Normalising also makes the column comparable across clips of differing
    # resolution, which is what lets a rejection threshold be chosen against
    # it later.
    normalised_image = hartley_normalise(image_points)
    normalised_template = hartley_normalise(template_points)

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(normalised_image, normalised_template):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    singular_values = np.linalg.svd(np.array(rows, dtype=np.float64), compute_uv=False)
    if len(singular_values) < 8 or singular_values[0] <= 0.0:
        return float('nan')

    # sigma[7] over sigma[0], but WHICH quantity that is depends on the matrix
    # shape, which is why the reported column must be read within a
    # confident-count bucket rather than pooled across them. With 4 fitting
    # points the matrix is 8x9, so sigma[7] is its smallest singular value.
    # With 5 or more it is 2n x 9 with nine values, sigma[8] is the near-null
    # one (~1e-16, the solution itself) and sigma[7] is the SECOND smallest.
    # Measured: 0.835 at 4 points against 1.073 at 5 and 1.896 at 8 for the
    # same court geometry -- not comparable across that boundary. In both cases
    # sigma[7] is the last non-null direction, so near zero means the solution
    # is barely distinguished from a family; only the scale differs.
    return float(singular_values[7] / singular_values[0])


def apply_homography(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float] | None:
    """Apply a homography to one point, returning None when the result falls on the line at infinity."""
    vector = matrix @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if not np.isfinite(vector[2]) or abs(float(vector[2])) < 1e-12:
        return None
    return (float(vector[0] / vector[2]), float(vector[1] / vector[2]))


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the Euclidean distance between two points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def leave_one_out_residuals(
    frame_keypoints: list[Keypoint],
) -> tuple[list[tuple[int, float, float, float]], int, int]:
    """Return per-keypoint (index, metres, pixels, conditioning) residuals for one frame, plus the counts of degenerate and under-determined rejections."""
    # A count rather than a boolean: one degenerate fit among eight does not
    # invalidate the other seven, and reporting it as a lost FRAME would both
    # discard good observations and stop the coverage columns reconciling
    # against the frame count.
    confident = confident_keypoints(frame_keypoints)
    residuals: list[tuple[int, float, float, float]] = []
    degenerate_fits = 0
    collinear_fits = 0

    for held_out in confident:
        others = [keypoint for keypoint in confident if keypoint.index != held_out.index]
        if len(others) < MIN_SUFFICIENT_KEYPOINTS:
            continue

        image_points = [(keypoint.x, keypoint.y) for keypoint in others]
        template_points = [TEMPLATE_POINTS_M[keypoint.index] for keypoint in others]

        # Counted apart from the determinant rejections below: an
        # under-determined fitting set is a property of WHICH landmarks were
        # confident, while a degenerate determinant is a property of the fit
        # itself. The write-up needs to distinguish the two causes.
        if not has_general_position_quadruple(template_points):
            collinear_fits += 1
            continue

        matrix = fit_homography(image_points, template_points)
        if matrix is None:
            degenerate_fits += 1
            continue

        predicted_m = apply_homography(matrix, (held_out.x, held_out.y))
        if predicted_m is None:
            degenerate_fits += 1
            continue

        truth_m = TEMPLATE_POINTS_M[held_out.index]
        metres = euclidean(predicted_m, truth_m)

        # Back into pixels via the inverse: metres is what Stage 9 inherits,
        # pixels is what is visible in a rendered frame. Reported together
        # because neither alone answers both questions.
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            degenerate_fits += 1
            continue

        predicted_px = apply_homography(inverse, truth_m)
        if predicted_px is None:
            degenerate_fits += 1
            continue

        pixels = euclidean(predicted_px, (held_out.x, held_out.y))
        conditioning = dlt_conditioning(image_points, template_points)
        residuals.append((held_out.index, metres, pixels, conditioning))

    return residuals, degenerate_fits, collinear_fits


def frame_displacements(
    previous: list[Keypoint],
    current: list[Keypoint],
) -> list[tuple[int, float]]:
    """Return per-index pixel displacements between two consecutive frames, only where the index is confident in both."""
    previous_by_index = {
        keypoint.index: keypoint for keypoint in previous
        if keypoint.confidence >= KEYPOINT_CONFIDENCE_THRESHOLD
    }
    displacements: list[tuple[int, float]] = []
    for keypoint in current:
        if keypoint.confidence < KEYPOINT_CONFIDENCE_THRESHOLD:
            continue
        earlier = previous_by_index.get(keypoint.index)
        if earlier is None:
            continue
        displacements.append((keypoint.index, euclidean((earlier.x, earlier.y), (keypoint.x, keypoint.y))))
    return displacements


def median_and_iqr(values: list[float]) -> tuple[float, float, float]:
    """Return the median and the 25th and 75th percentiles of a list of values."""
    # Median and IQR rather than mean and standard deviation: the residual
    # distribution is expected to be heavy-tailed, and a mean would be
    # dominated by a handful of bad frames.
    if not values:
        return (float('nan'), float('nan'), float('nan'))
    ordered = sorted(values)
    if len(ordered) == 1:
        only = ordered[0]
        return (only, only, only)
    quartiles = statistics.quantiles(ordered, n=4, method='inclusive')
    return (statistics.median(ordered), quartiles[0], quartiles[2])


def percentile(values: list[float], fraction: float) -> float:
    """Return the value at a given fraction through the sorted list, by nearest rank."""
    if not values:
        return float('nan')
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[rank]


def fmt(value: float, places: int = 3) -> str:
    """Format a float for a results table, rendering an absent value as a dash."""
    return '-' if value != value else f'{value:.{places}f}'


def measure_clip(clip: str) -> dict:
    """Run all three measurements over one clip, returning its rows and per-clip counts."""
    frames = [frame for _, frame in load_video(CLIP_PATH_TEMPLATE.format(clip=clip))]

    detector = CourtKeypoints(
        MODEL_PATH, keypoint_confidence_threshold=KEYPOINT_CONFIDENCE_THRESHOLD,
    )
    # Through run_detection with the production cache path rather than by
    # loading the pickle directly, so the fingerprint check runs and a stale
    # cache regenerates instead of being silently reused.
    keypoints_per_frame = detector.run_detection(
        frames=frames,
        video_path=CLIP_PATH_TEMPLATE.format(clip=clip),
        cache_path=KEYPOINTS_CACHE_TEMPLATE.format(clip=clip),
    )

    sufficiency_rows: list[list[str]] = []
    reprojection_rows: list[list[str]] = []
    stability_rows: list[list[str]] = []

    confident_counts: list[int] = []
    # Four mutually exclusive frame categories that must sum to the frame
    # count: never attempted (too few confident points), attempted and lost
    # entirely to degenerate fits, attempted and lost entirely to
    # under-determined fitting sets, and scored. The two loss causes are kept
    # apart because they mean different things: a degenerate fit is a
    # property of the numerics, an under-determined set a property of WHICH
    # landmarks were confident. degenerate_fits and collinear_fits count
    # individual held-out reprojections and are NOT frame categories: a
    # frame can contribute both good residuals and rejected fits.
    coverage_skipped = 0
    lost_degenerate_frames = 0
    lost_collinear_frames = 0
    scored_frames = 0
    degenerate_fits = 0
    collinear_fits = 0

    for frame_idx, frame_keypoints in enumerate(keypoints_per_frame):
        count = len(confident_keypoints(frame_keypoints))
        confident_counts.append(count)
        sufficiency_rows.append([
            clip, str(frame_idx), str(count),
            str(count >= MIN_SUFFICIENT_KEYPOINTS), str(count >= MIN_REPROJECTION_KEYPOINTS),
        ])

        if count < MIN_REPROJECTION_KEYPOINTS:
            # Localisation is measured only where the model already has
            # several confident points, so the frames where errors are most
            # suspected are precisely the ones excluded. Counted and reported.
            coverage_skipped += 1
            continue

        residuals, frame_degenerate, frame_collinear = leave_one_out_residuals(frame_keypoints)
        degenerate_fits += frame_degenerate
        collinear_fits += frame_collinear
        if residuals:
            scored_frames += 1
        elif frame_collinear > frame_degenerate:
            # A frame losing fits to both causes counts once, under whichever
            # accounts for more of its rejections; an exact tie is attributed
            # to the degenerate branch below, since a degenerate fit is the
            # more specific numerical failure. Both are distinct from a
            # coverage skip, which was never attempted at all.
            lost_collinear_frames += 1
        else:
            lost_degenerate_frames += 1
        for index, metres, pixels, conditioning in residuals:
            reprojection_rows.append([
                clip, str(frame_idx), str(index), str(count),
                f'{metres:.6f}', f'{pixels:.6f}', f'{conditioning:.6e}',
            ])

    for frame_idx in range(1, len(keypoints_per_frame)):
        for index, displacement in frame_displacements(
            keypoints_per_frame[frame_idx - 1], keypoints_per_frame[frame_idx],
        ):
            stability_rows.append([clip, str(frame_idx), str(index), f'{displacement:.6f}'])

    return {
        'clip': clip,
        'n_frames': len(keypoints_per_frame),
        'confident_counts': confident_counts,
        'runs': below_threshold_runs(confident_counts),
        'coverage_skipped': coverage_skipped,
        'lost_degenerate_frames': lost_degenerate_frames,
        'lost_collinear_frames': lost_collinear_frames,
        'scored_frames': scored_frames,
        'degenerate_fits': degenerate_fits,
        'collinear_fits': collinear_fits,
        'sufficiency_rows': sufficiency_rows,
        'reprojection_rows': reprojection_rows,
        'stability_rows': stability_rows,
    }


def sufficiency_table(measurements: list[dict]) -> list[list[str]]:
    """Return the per-clip sufficiency rows: reach rates, and the runs of consecutive sub-threshold frames."""
    rows: list[list[str]] = []
    for measurement in measurements:
        counts = measurement['confident_counts']
        n_frames = len(counts)
        reaching_four = sum(1 for count in counts if count >= MIN_SUFFICIENT_KEYPOINTS)
        reaching_five = sum(1 for count in counts if count >= MIN_REPROJECTION_KEYPOINTS)
        runs = measurement['runs']
        rows.append([
            measurement['clip'], str(n_frames),
            f'{reaching_four}/{n_frames}', fmt(reaching_four / n_frames if n_frames else float('nan')),
            f'{reaching_five}/{n_frames}', fmt(reaching_five / n_frames if n_frames else float('nan')),
            str(len(runs)), ','.join(str(length) for length in runs) or '-',
        ])
    return rows


def reprojection_bucket_table(measurements: list[dict]) -> list[list[str]]:
    """Return residual medians and IQRs per clip per confident-count bucket, never pooled across buckets."""
    rows: list[list[str]] = []
    bucket_labels = [str(size) for size in REPROJECTION_BUCKETS] + [OVERFLOW_BUCKET]

    for scope, rows_source in (
        [(measurement['clip'], measurement['reprojection_rows']) for measurement in measurements]
        + [('pooled', [row for measurement in measurements for row in measurement['reprojection_rows']])]
    ):
        for label in bucket_labels:
            metres = [float(row[4]) for row in rows_source if bucket_for(int(row[3])) == label]
            pixels = [float(row[5]) for row in rows_source if bucket_for(int(row[3])) == label]
            if not metres:
                continue
            median_m, q1_m, q3_m = median_and_iqr(metres)
            median_px, q1_px, q3_px = median_and_iqr(pixels)
            rows.append([
                scope, label, str(len(metres)),
                fmt(median_m), f'{fmt(q1_m)}-{fmt(q3_m)}',
                fmt(median_px, 2), f'{fmt(q1_px, 2)}-{fmt(q3_px, 2)}',
            ])
    return rows


def reprojection_index_table(measurements: list[dict]) -> list[list[str]]:
    """Return the median and IQR residual per keypoint index, per clip and pooled."""
    # Per clip as well as pooled, because section 4.2 asks for both and a
    # clip-specific index failure -- clip_3 being the obvious candidate -- is
    # invisible in a pooled median. The pooled row is kept alongside rather
    # than instead, so the summary answers both questions.
    scopes = [
        (measurement['clip'], measurement['reprojection_rows']) for measurement in measurements
    ]
    scopes.append(
        ('pooled', [row for measurement in measurements for row in measurement['reprojection_rows']])
    )

    rows: list[list[str]] = []
    for scope, source in scopes:
        for index in range(NUM_KEYPOINTS):
            metres = [float(row[4]) for row in source if int(row[2]) == index]
            pixels = [float(row[5]) for row in source if int(row[2]) == index]
            if not metres:
                rows.append([scope, str(index), '0', '-', '-', '-', '-'])
                continue
            median_m, q1_m, q3_m = median_and_iqr(metres)
            median_px, q1_px, q3_px = median_and_iqr(pixels)
            rows.append([
                scope, str(index), str(len(metres)),
                fmt(median_m), f'{fmt(q1_m)}-{fmt(q3_m)}',
                fmt(median_px, 2), f'{fmt(q1_px, 2)}-{fmt(q3_px, 2)}',
            ])
    return rows


def conditioning_table(measurements: list[dict]) -> list[list[str]]:
    """Return the distribution of DLT conditioning per clip and pooled, bucketed by confident-keypoint count so the values stay comparable."""
    # Bucketed for the same reason the residual table is, and more sharply:
    # the DLT matrix shape changes with the fitting-point count, so sigma[7]
    # is the smallest singular value at 4 points and the second smallest at 5
    # or more (see dlt_conditioning). Pooling would average quantities that
    # are not the same measurement. Bucketing also lets this table be read
    # directly against the residual table, which uses the same buckets.
    scopes = [
        (measurement['clip'], measurement['reprojection_rows']) for measurement in measurements
    ]
    scopes.append(
        ('pooled', [row for measurement in measurements for row in measurement['reprojection_rows']])
    )
    bucket_labels = [str(size) for size in REPROJECTION_BUCKETS] + [OVERFLOW_BUCKET]

    rows: list[list[str]] = []
    for scope, source in scopes:
        for label in bucket_labels:
            values = [
                float(row[6]) for row in source
                if bucket_for(int(row[3])) == label and float(row[6]) == float(row[6])
            ]
            if not values:
                continue
            median_c, q1_c, q3_c = median_and_iqr(values)
            rows.append([
                scope, label, str(len(values)), f'{min(values):.3e}', f'{q1_c:.3e}',
                f'{median_c:.3e}', f'{q3_c:.3e}', f'{max(values):.3e}',
            ])
    return rows


def coverage_table(measurements: list[dict]) -> list[list[str]]:
    """Return the per-clip frame accounting for the reprojection measurement, with the four frame categories reconciling against the frame count."""
    rows: list[list[str]] = []
    for measurement in measurements:
        # The four frame categories are mutually exclusive and exhaustive, so
        # their sum is printed beside the frame count: if the two ever differ,
        # the accounting is wrong and the residuals cannot be trusted.
        accounted = (
            measurement['coverage_skipped']
            + measurement['lost_degenerate_frames']
            + measurement['lost_collinear_frames']
            + measurement['scored_frames']
        )
        rows.append([
            measurement['clip'], str(measurement['n_frames']),
            str(measurement['coverage_skipped']),
            str(measurement['lost_degenerate_frames']), str(measurement['lost_collinear_frames']),
            str(measurement['scored_frames']), str(accounted),
            'yes' if accounted == measurement['n_frames'] else 'NO',
            str(measurement['degenerate_fits']), str(measurement['collinear_fits']),
            str(len(measurement['reprojection_rows'])),
        ])
    return rows


def stability_table(measurements: list[dict]) -> list[list[str]]:
    """Return per-clip per-index displacement medians and 95th percentiles, with the contributing pair count."""
    rows: list[list[str]] = []
    for measurement in measurements:
        for index in range(NUM_KEYPOINTS):
            displacements = [
                float(row[3]) for row in measurement['stability_rows'] if int(row[2]) == index
            ]
            if not displacements:
                continue
            rows.append([
                measurement['clip'], str(index), str(len(displacements)),
                fmt(statistics.median(displacements), 2), fmt(percentile(displacements, 0.95), 2),
            ])
    return rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write one results CSV, creating the output directory if needed."""
    # Row-level rather than summary-only: ablations A2 and A3 re-score these
    # same detections under a different filtering policy, and re-running
    # inference per policy would be wasteful and would risk the policies
    # seeing different detections.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(measurements: list[dict], skipped: list[tuple[str, str]]) -> None:
    """Print every table and persist all four artefacts, writing nothing at all when no clip was scored."""
    # Partial results are worth keeping when at least one clip succeeded, but a
    # run where every clip failed has nothing to save and would overwrite four
    # existing files with headers only -- silently, since the run still reports
    # skipped clips and leaves files that appear to exist. Those outputs cost a
    # full inference pass to regenerate. Same rule as run_evaluation.py omitting
    # an all-zero pooled row, and load_cache refusing to let a bad read
    # overwrite evidence.
    # Declined deliberately: a PARTIALLY failed run does rewrite complete
    # previous outputs with a subset, and that is intended. Partial writes
    # are why the finally exists -- a partial write is a real measurement of
    # what actually ran, with the skipped clips recorded in the summary,
    # whereas the all-failed case below is an empty file masquerading as a
    # result. Per-clip output files would fragment the artefact the
    # ablations read, and refusing to rewrite would leave the script unable
    # to update after a transient failure.
    if not measurements:
        print('\n[measure] No clip was scored, so nothing was written.')
        if skipped:
            for clip, reason in skipped:
                print(f'[measure]   {clip}: {reason}')
        print(
            f'[measure] Existing files in {OUTPUT_DIR} are left untouched rather than '
            f'overwritten with empty results.'
        )
        return

    lines: list[str] = []

    print_table(
        'Sufficiency (section 4.3)',
        ['clip', 'frames', 'reach_4', 'rate_4', 'reach_5', 'rate_5', 'runs_below_4', 'run_lengths'],
        sufficiency_table(measurements), lines,
    )
    print_table(
        'Reprojection frame accounting — every frame is skipped, lost (to either cause) or scored '
        '(degenerate_fits counts individual held-out reprojections, not frames)',
        [
            'clip', 'frames', 'skipped_below_5', 'lost_all_fits_degenerate',
            'lost_all_fits_collinear', 'scored',
            'accounted', 'reconciles', 'degenerate_fits', 'collinear_fits', 'observations',
        ],
        coverage_table(measurements), lines,
    )
    print_table(
        'Leave-one-out reprojection by confident-count bucket (section 4.2)',
        ['scope', 'bucket', 'n', 'median_m', 'iqr_m', 'median_px', 'iqr_px'],
        reprojection_bucket_table(measurements), lines,
    )
    print_table(
        'Leave-one-out reprojection by keypoint index, per clip and pooled (section 4.2)',
        ['scope', 'index', 'n', 'median_m', 'iqr_m', 'median_px', 'iqr_px'],
        reprojection_index_table(measurements), lines,
    )
    print_table(
        'DLT conditioning of each fit by confident-count bucket (recorded, never used to reject) — '
        'sigma[7]/sigma[0], near zero meaning the solution is barely distinguished from a family; '
        'never pooled across buckets, since the matrix shape and so the meaning of sigma[7] changes with the count',
        ['scope', 'bucket', 'n', 'min', 'q1', 'median', 'q3', 'max'],
        conditioning_table(measurements), lines,
    )
    print_table(
        'Temporal stability (section 4.5)',
        ['clip', 'index', 'pairs', 'median_px', 'p95_px'],
        stability_table(measurements), lines,
    )

    if skipped:
        print_table('Skipped clips', ['clip', 'reason'], [list(entry) for entry in skipped], lines)

    limitation = (
        '\nLimitation: reprojection is measured only on frames already carrying at least '
        f'{MIN_REPROJECTION_KEYPOINTS} confident keypoints, so the frames where localisation '
        'error is most suspected are precisely the ones excluded. Read the coverage table '
        'beside the residuals. Residuals are reported per confident-count bucket and never '
        'pooled across buckets: a 5-keypoint frame is an exact 4-point fit and is not '
        'comparable to an over-determined 8-keypoint frame.'
    )
    print(limitation)
    lines.append(limitation)

    write_csv(
        SUFFICIENCY_CSV, ['clip', 'frame', 'confident_count', 'reaches_4', 'reaches_5'],
        [row for measurement in measurements for row in measurement['sufficiency_rows']],
    )
    write_csv(
        REPROJECTION_CSV,
        [
            'clip', 'frame', 'keypoint_index', 'confident_count',
            'residual_m', 'residual_px', 'dlt_conditioning',
        ],
        [row for measurement in measurements for row in measurement['reprojection_rows']],
    )
    write_csv(
        STABILITY_CSV, ['clip', 'frame', 'keypoint_index', 'displacement_px'],
        [row for measurement in measurements for row in measurement['stability_rows']],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nWrote {SUFFICIENCY_CSV}, {REPROJECTION_CSV}, {STABILITY_CSV} and {SUMMARY_TXT}')


def main() -> None:
    """Measure every clip and write the results, keeping partial output if one clip fails."""
    measurements: list[dict] = []
    skipped: list[tuple[str, str]] = []

    try:
        for clip in CLIPS:
            try:
                print(f'[measure] {clip}...')
                measurements.append(measure_clip(clip))
            except (OSError, ValueError) as error:
                # One unreadable clip or cache must not abort the whole
                # measurement; the remaining clips still produce results.
                print(f'[measure] Skipping {clip}: {type(error).__name__}: {error}')
                skipped.append((clip, f'{type(error).__name__}: {error}'))
    finally:
        write_outputs(measurements, skipped)


if __name__ == '__main__':
    main()
