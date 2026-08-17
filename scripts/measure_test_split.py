"""
measure_test_split.py runs the court keypoint spec's section 4.2 leave-one-out
reprojection on the grouped test split so it is directly comparable with the same
measurement on the evaluation clips, and measures direct pixel error against the
split's labels. Runs on JupyterHub; it needs the checkpoint and the dataset.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Running this file directly (`python scripts/measure_test_split.py`) puts
# scripts/ on sys.path[0], not the repo root, so `basketball` would not be
# importable. Insert the repo root explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basketball.keypoints.court_keypoints import CourtKeypoints, Keypoint
from basketball.keypoints.court_template import NUM_KEYPOINTS
from scripts.measure_court_keypoints import (
    KEYPOINT_CONFIDENCE_THRESHOLD,
    MIN_REPROJECTION_KEYPOINTS,
    OVERFLOW_BUCKET,
    REPROJECTION_BUCKETS,
    bucket_for,
    fmt,
    leave_one_out_residuals,
    median_and_iqr,
    print_table,
)

TEST_IMAGE_DIR = Path('training/court-keypoints-grouped/test/images')
TEST_LABEL_DIR = Path('training/court-keypoints-grouped/test/labels')
MODEL_PATH = 'models/keypoints.pt'

OUTPUT_DIR = Path('data/outputs/keypoint_evaluation')
TEST_REPROJECTION_CSV = OUTPUT_DIR / 'test_split_reprojection.csv'
TEST_DIRECT_ERROR_CSV = OUTPUT_DIR / 'test_split_direct_error.csv'
TEST_SUMMARY_TXT = OUTPUT_DIR / 'test_split_summary.txt'

# The PER-KEYPOINT confidence threshold, gating individual landmarks. Distinct
# from CourtKeypoints' instance threshold, which gates whether a court is
# detected at all. Bound to the clip measurement's constant rather than
# redeclared at the same value: the imported leave_one_out_residuals() filters
# with THAT constant, so a local copy would let the two drift apart silently,
# desynchronising the reprojection residuals from the recorded confident_count
# and from direct-error matching. Aliased rather than used directly so the name
# still says which of the two thresholds this is.
TEST_KEYPOINT_CONFIDENCE_THRESHOLD = KEYPOINT_CONFIDENCE_THRESHOLD

# Ultralytics' three-state visibility flag: 0 absent, 1 occluded, 2 visible.
# Both 1 and 2 are trained on and both mark a landmark that exists in the
# image, so 'labelled present' is v > 0 rather than v == 2.
VISIBLE_FLAG_MINIMUM = 1

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')

# Bounding-box values preceding the keypoint triples in a YOLO pose label line,
# counted after the class index has already been dropped.
LABEL_BOX_VALUES = 4

# The stable part of the message CourtKeypoints raises when a checkpoint emits
# no per-keypoint confidences. Pinned by a test against the real class, so a
# reworded message fails there rather than silently degrading this script back
# to reporting the same fault once per image.
NO_CONFIDENCE_MARKER = 'provides no per-keypoint confidence'


def is_checkpoint_level_error(error: Exception) -> bool:
    """Return whether an error raised while measuring one image is really a property of the checkpoint, so every image would raise it too."""
    # Matched on the message because CourtKeypoints raises a bare ValueError
    # for this rather than a dedicated type, and introducing one there is a
    # change to a Stage 7 module that this measurement script has no business
    # making. The substring is the stable part of that message.
    return isinstance(error, ValueError) and NO_CONFIDENCE_MARKER in str(error)


def image_paths() -> list[Path]:
    """Return every test-split image path, sorted, so a run is reproducible and its output ordering is stable."""
    if not TEST_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f'Test split images not found at {TEST_IMAGE_DIR}. '
            f'Run training/split_court_keypoints.py to build the grouped split.'
        )
    return sorted(
        path for path in TEST_IMAGE_DIR.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_path_for(image_path: Path) -> Path:
    """Return the YOLO label path matching one image, by stem."""
    return TEST_LABEL_DIR / f'{image_path.stem}.txt'


def parse_label(
    text: str,
    width: int,
    height: int,
) -> dict[int, tuple[float, float]]:
    """Return the labelled pixel position of every present keypoint in one YOLO pose label, keyed by index."""
    # Only the first instance is read: the dataset labels one court per image,
    # and a second row would be a different object rather than more keypoints
    # of the same one.
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {}

    values = [float(value) for value in lines[0].split()[1:]]
    triples = values[LABEL_BOX_VALUES:]
    if len(triples) < NUM_KEYPOINTS * 3:
        raise ValueError(
            f'Label holds {len(triples) // 3} keypoints, expected {NUM_KEYPOINTS}.'
        )

    present: dict[int, tuple[float, float]] = {}
    for index in range(NUM_KEYPOINTS):
        x, y, visibility = triples[index * 3:index * 3 + 3]
        if visibility >= VISIBLE_FLAG_MINIMUM:
            # Coordinates are normalised in the label and are compared against
            # predictions in pixels, so they are scaled by this image's own
            # size rather than a shared constant: the split spans many source
            # videos at differing resolutions.
            present[index] = (x * width, y * height)
    return present


def confident_by_index(frame_keypoints: list[Keypoint]) -> dict[int, Keypoint]:
    """Return the keypoints at or above this script's per-keypoint threshold, keyed by index."""
    return {
        keypoint.index: keypoint for keypoint in frame_keypoints
        if keypoint.confidence >= TEST_KEYPOINT_CONFIDENCE_THRESHOLD
    }


def direct_errors(
    frame_keypoints: list[Keypoint],
    labelled: dict[int, tuple[float, float]],
    width: int,
    height: int,
) -> tuple[list[tuple[int, float, float, float, float, float, float]], int, int]:
    """Return per-index matched errors for one image plus the counts of confident-but-absent and labelled-but-unconfident keypoints."""
    # The two counts are identity-level rather than localisation-level: a
    # keypoint the model places confidently where the label says nothing exists
    # has no error to measure, and counting it as a zero-error match -- or
    # omitting it silently -- would flatter the model in opposite directions.
    predicted = confident_by_index(frame_keypoints)
    diagonal = float(np.hypot(width, height))

    matched: list[tuple[int, float, float, float, float, float, float]] = []
    false_positives = 0
    false_negatives = 0

    for index in range(NUM_KEYPOINTS):
        keypoint = predicted.get(index)
        truth = labelled.get(index)

        if keypoint is not None and truth is None:
            false_positives += 1
            continue
        if keypoint is None and truth is not None:
            false_negatives += 1
            continue
        if keypoint is None or truth is None:
            continue

        error_px = float(np.hypot(keypoint.x - truth[0], keypoint.y - truth[1]))
        # Normalised by the image diagonal as well as raw: the split's images
        # come from many source videos at differing resolutions, so a raw pixel
        # error is not comparable across them and a pooled median of raw pixels
        # would be dominated by whichever resolution is most common.
        normalised = error_px / diagonal if diagonal > 0.0 else float('nan')
        matched.append(
            (index, keypoint.x, keypoint.y, truth[0], truth[1], error_px, normalised)
        )

    return matched, false_positives, false_negatives


def measure_image(
    image_path: Path,
    detector: CourtKeypoints,
) -> dict:
    """Measure reprojection residuals and direct label error for one test image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f'Could not read image {image_path}')

    label_file = label_path_for(image_path)
    if not label_file.exists():
        raise OSError(f'Image {image_path.name} has no label at {label_file}')

    height, width = image.shape[:2]
    labelled = parse_label(label_file.read_text(encoding='utf-8'), width, height)

    frame_keypoints = detector.detect_keypoints(image)
    confident_count = len(confident_by_index(frame_keypoints))

    # The same four mutually exclusive categories the clip measurement uses,
    # so the two accounting tables reconcile the same way: never attempted
    # (too few confident points), attempted and lost entirely to degenerate
    # fits, attempted and lost entirely to under-determined fitting sets, and
    # scored. Without this the reprojection n cannot be reconciled against the
    # images and keypoints actually attempted.
    residuals: list[tuple[int, float, float, float]] = []
    degenerate_fits = 0
    collinear_fits = 0
    category = 'skipped'

    if confident_count >= MIN_REPROJECTION_KEYPOINTS:
        residuals, degenerate_fits, collinear_fits = leave_one_out_residuals(frame_keypoints)
        if residuals:
            category = 'scored'
        elif collinear_fits > degenerate_fits:
            # A tie is attributed to the degenerate branch, matching the clip
            # measurement, since a degenerate fit is the more specific failure.
            category = 'lost_collinear'
        else:
            category = 'lost_degenerate'

    matched, false_positives, false_negatives = direct_errors(
        frame_keypoints, labelled, width, height,
    )

    reprojection_rows = [
        [
            image_path.name, str(index), str(confident_count),
            f'{metres:.6f}', f'{pixels:.6f}', f'{conditioning:.6e}',
        ]
        for index, metres, pixels, conditioning in residuals
    ]
    direct_rows = [
        [
            image_path.name, str(index), f'{px:.3f}', f'{py:.3f}',
            f'{tx:.3f}', f'{ty:.3f}', f'{error:.6f}', f'{normalised:.8f}',
            str(width), str(height),
        ]
        for index, px, py, tx, ty, error, normalised in matched
    ]

    return {
        'image': image_path.name,
        'width': width,
        'height': height,
        'confident_count': confident_count,
        'labelled_present': len(labelled),
        'category': category,
        'degenerate_fits': degenerate_fits,
        'collinear_fits': collinear_fits,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'matched': matched,
        'reprojection_rows': reprojection_rows,
        'direct_rows': direct_rows,
    }


def coverage_table(measurements: list[dict]) -> list[list[str]]:
    """Return the image accounting for the reprojection measurement, with the four categories reconciling against the image count."""
    # Matches coverage_table in the clip measurement, so the two sit side by
    # side. The four categories are mutually exclusive and exhaustive, so their
    # sum is printed beside the image count: if the two ever differ, the
    # accounting is wrong and the residuals cannot be trusted. degenerate_fits
    # and collinear_fits count individual held-out reprojections, NOT images:
    # one image can contribute both good residuals and rejected fits.
    counts = {'skipped': 0, 'lost_degenerate': 0, 'lost_collinear': 0, 'scored': 0}
    unknown = 0
    for measurement in measurements:
        if measurement['category'] in counts:
            counts[measurement['category']] += 1
        else:
            # Counted rather than raising, and deliberately excluded from
            # `accounted`: an unrecognised category is what makes the
            # reconciliation a real check instead of a tautology. Summing the
            # four buckets alone would always equal the image count no matter
            # what measure_image produced, so the column would print 'yes'
            # unconditionally and verify nothing.
            unknown += 1

    accounted = sum(counts.values())
    return [[
        str(len(measurements)),
        str(counts['skipped']), str(counts['lost_degenerate']),
        str(counts['lost_collinear']), str(counts['scored']),
        str(accounted), 'yes' if accounted == len(measurements) else f'NO ({unknown} unknown)',
        str(sum(measurement['degenerate_fits'] for measurement in measurements)),
        str(sum(measurement['collinear_fits'] for measurement in measurements)),
        str(sum(len(measurement['reprojection_rows']) for measurement in measurements)),
    ]]


def resolution_table(measurements: list[dict]) -> list[list[str]]:
    """Return the images and residual observations contributing to the pooled pixel column, per distinct resolution."""
    # Printed because it is the reason the reprojection pixel column is only
    # indicative: the split spans six resolutions whose diagonals differ by
    # roughly 6.8x, so a pooled pixel median is dominated by whichever
    # resolutions are present rather than by localisation quality.
    #
    # Only images that actually produced residuals are counted. A skipped or
    # lost image contributes no row to the pooled column, so including it
    # would describe the composition of a set the pooled median is not drawn
    # from. Shares are weighted by OBSERVATION rather than by image, because
    # one image contributes several held-out residuals and it is the
    # observation count that determines the median. Both counts are reported:
    # images answers how much of the split is represented, observations
    # answers what the pooled figure is actually made of.
    images: dict[tuple[int, int], int] = {}
    observations: dict[tuple[int, int], int] = {}
    for measurement in measurements:
        residual_count = len(measurement['reprojection_rows'])
        if not residual_count:
            continue
        key = (measurement['width'], measurement['height'])
        images[key] = images.get(key, 0) + 1
        observations[key] = observations.get(key, 0) + residual_count

    total_observations = sum(observations.values())
    rows: list[list[str]] = []
    for key in sorted(observations, key=lambda item: (-observations[item], item)):
        width, height = key
        rows.append([
            f'{width}x{height}', str(images[key]), str(observations[key]),
            fmt(observations[key] / total_observations if total_observations else float('nan')),
            f'{np.hypot(width, height):.1f}',
        ])
    return rows


def reprojection_bucket_table(measurements: list[dict]) -> list[list[str]]:
    """Return test-split residual medians and IQRs per confident-count bucket, never pooled across buckets."""
    # Same buckets and same reporting shape as the clip measurement's own
    # table, so the two sit side by side. Pooling across buckets would mix an
    # exact 4-point fit with an over-determined one, which is why neither
    # table does it.
    rows_source = [row for measurement in measurements for row in measurement['reprojection_rows']]
    bucket_labels = [str(size) for size in REPROJECTION_BUCKETS] + [OVERFLOW_BUCKET]

    rows: list[list[str]] = []
    for label in bucket_labels:
        metres = [float(row[3]) for row in rows_source if bucket_for(int(row[2])) == label]
        pixels = [float(row[4]) for row in rows_source if bucket_for(int(row[2])) == label]
        if not metres:
            continue
        median_m, q1_m, q3_m = median_and_iqr(metres)
        # Reported but NOT normalised by image diagonal, unlike direct error.
        # Normalising would break comparability with the clip table, which
        # reports raw pixels, and comparability is the reason this script
        # exists. The column heading and the limitations mark it indicative.
        median_px, q1_px, q3_px = median_and_iqr(pixels)
        rows.append([
            'test', label, str(len(metres)),
            fmt(median_m), f'{fmt(q1_m)}-{fmt(q3_m)}',
            fmt(median_px, 2), f'{fmt(q1_px, 2)}-{fmt(q3_px, 2)}',
        ])
    return rows


def conditioning_table(measurements: list[dict]) -> list[list[str]]:
    """Return the distribution of DLT conditioning on the test split, bucketed by confident-keypoint count so the values stay comparable."""
    # Same shape and same buckets as the clip measurement's own conditioning
    # table, so the two artefacts are column-comparable. Never pooled across
    # buckets: the DLT matrix shape changes with the fitting-point count, so
    # sigma[7] is the smallest singular value at 4 points and the second
    # smallest at 5 or more, and averaging those is averaging two different
    # quantities. NaN values are excluded rather than propagated.
    source = [row for measurement in measurements for row in measurement['reprojection_rows']]
    bucket_labels = [str(size) for size in REPROJECTION_BUCKETS] + [OVERFLOW_BUCKET]

    rows: list[list[str]] = []
    for label in bucket_labels:
        values = [
            float(row[5]) for row in source
            if bucket_for(int(row[2])) == label and float(row[5]) == float(row[5])
        ]
        if not values:
            continue
        median_c, q1_c, q3_c = median_and_iqr(values)
        rows.append([
            'test', label, str(len(values)), f'{min(values):.3e}', f'{q1_c:.3e}',
            f'{median_c:.3e}', f'{q3_c:.3e}', f'{max(values):.3e}',
        ])
    return rows


def direct_error_index_table(measurements: list[dict]) -> list[list[str]]:
    """Return the median and IQR direct error per keypoint index and pooled, in pixels and normalised by image diagonal."""
    matched = [entry for measurement in measurements for entry in measurement['matched']]

    rows: list[list[str]] = []
    for index in range(NUM_KEYPOINTS):
        errors = [entry[5] for entry in matched if entry[0] == index]
        normalised = [entry[6] for entry in matched if entry[0] == index]
        if not errors:
            rows.append([str(index), '0', '-', '-', '-', '-'])
            continue
        median_px, q1_px, q3_px = median_and_iqr(errors)
        median_n, q1_n, q3_n = median_and_iqr(normalised)
        rows.append([
            str(index), str(len(errors)),
            fmt(median_px, 2), f'{fmt(q1_px, 2)}-{fmt(q3_px, 2)}',
            fmt(median_n, 4), f'{fmt(q1_n, 4)}-{fmt(q3_n, 4)}',
        ])

    if matched:
        errors = [entry[5] for entry in matched]
        normalised = [entry[6] for entry in matched]
        median_px, q1_px, q3_px = median_and_iqr(errors)
        median_n, q1_n, q3_n = median_and_iqr(normalised)
        rows.append([
            'pooled', str(len(errors)),
            fmt(median_px, 2), f'{fmt(q1_px, 2)}-{fmt(q3_px, 2)}',
            fmt(median_n, 4), f'{fmt(q1_n, 4)}-{fmt(q3_n, 4)}',
        ])
    return rows


def identity_table(measurements: list[dict]) -> list[list[str]]:
    """Return the identity-level accounting: matched keypoints against confident-but-absent and labelled-but-unconfident counts."""
    matched = sum(len(measurement['matched']) for measurement in measurements)
    false_positives = sum(measurement['false_positives'] for measurement in measurements)
    false_negatives = sum(measurement['false_negatives'] for measurement in measurements)
    labelled = sum(measurement['labelled_present'] for measurement in measurements)
    confident = sum(measurement['confident_count'] for measurement in measurements)

    # Printed beside the totals they must reconcile against: a matched count
    # plus a false-negative count that does not equal the labelled total means
    # the accounting is wrong and the error medians cannot be trusted.
    return [[
        str(len(measurements)), str(matched), str(false_positives), str(false_negatives),
        str(labelled), str(matched + false_negatives),
        'yes' if matched + false_negatives == labelled else 'NO',
        str(confident), str(matched + false_positives),
        'yes' if matched + false_positives == confident else 'NO',
    ]]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write one results CSV, creating this script's own output directory if needed."""
    # Deliberately not measure_court_keypoints' write_csv, despite the
    # identical body: that one creates ITS module's OUTPUT_DIR, so reusing it
    # would make this script's output location depend on another module's
    # constant. Both point at the same directory today, which is exactly what
    # would keep the coupling invisible until one of them moved.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(measurements: list[dict], skipped: list[tuple[str, str]]) -> None:
    """Print every table and persist all three artefacts, writing nothing at all when no image was scored."""
    # Same rule the clip measurement adopted after review: a run where every
    # image failed has nothing to save, and writing headers only would leave
    # files that appear to exist and read as a result. The clip measurement's
    # own artefacts are never touched here -- these are separate paths.
    if not measurements:
        print('\n[test-split] No image was scored, so nothing was written.')
        for image, reason in skipped:
            print(f'[test-split]   {image}: {reason}')
        print(
            f'[test-split] Existing files in {OUTPUT_DIR} are left untouched rather than '
            f'overwritten with empty results.'
        )
        return

    lines: list[str] = []

    print_table(
        'Identity-level accounting on the grouped test split',
        [
            'images', 'matched', 'confident_but_absent', 'labelled_but_unconfident',
            'labelled_total', 'matched+fn', 'reconciles_labelled',
            'confident_total', 'matched+fp', 'reconciles_confident',
        ],
        identity_table(measurements), lines,
    )
    print_table(
        'Reprojection image accounting — every image is skipped, lost (to either cause) or '
        'scored (degenerate_fits counts individual held-out reprojections, not images)',
        [
            'images', 'skipped_below_5', 'lost_all_fits_degenerate', 'lost_all_fits_collinear',
            'scored', 'accounted', 'reconciles', 'degenerate_fits', 'collinear_fits',
            'observations',
        ],
        coverage_table(measurements), lines,
    )
    print_table(
        'Composition of the pooled reprojection pixel column by resolution — scored images '
        'only, since a skipped or lost image contributes no residual. The share is weighted '
        'by observation, not by image, because one image contributes several held-out '
        'residuals. This is the reason the pixel column below is indicative only',
        ['resolution', 'images', 'observations', 'observation_share', 'diagonal_px'],
        resolution_table(measurements), lines,
    )
    print_table(
        'Leave-one-out reprojection on the grouped test split by confident-count bucket '
        '(section 4.2, the same instrument as the clip measurement). COMPARE ON METRES: '
        'median_m/iqr_m are comparable with the clip table, median_px/iqr_px are INDICATIVE '
        'ONLY because they pool across resolutions (see limitation 4)',
        ['scope', 'bucket', 'n', 'median_m', 'iqr_m', 'median_px_indicative', 'iqr_px_indicative'],
        reprojection_bucket_table(measurements), lines,
    )
    print_table(
        'DLT conditioning of each fit by confident-count bucket (recorded, never used to '
        'reject) — sigma[7]/sigma[0], near zero meaning the solution is barely distinguished '
        'from a family; never pooled across buckets, since the matrix shape and so the '
        'meaning of sigma[7] changes with the count',
        ['scope', 'bucket', 'n', 'min', 'q1', 'median', 'q3', 'max'],
        conditioning_table(measurements), lines,
    )
    print_table(
        'Direct error against the split labels, per keypoint index and pooled — '
        'normalised by image diagonal because the split spans resolutions',
        ['index', 'n', 'median_px', 'iqr_px', 'median_norm', 'iqr_norm'],
        direct_error_index_table(measurements), lines,
    )

    if skipped:
        print_table(
            'Skipped images', ['image', 'reason'],
            [list(entry) for entry in skipped], lines,
        )

    limitations = (
        '\nLimitations, stated with the results rather than discovered afterwards:'
        '\n  1. The test images are contrast-equalised and the evaluation clips are not.'
        '\n     Roboflow applied adaptive equalisation to every image in this dataset, so a'
        '\n     clips-versus-test comparison confounds genuine domain gap (different'
        '\n     broadcasts, different courts) with a preprocessing mismatch. K4 predicts'
        '\n     degradation without separating the two, and this script cannot separate them'
        '\n     either. Ablation A2 is the measurement that addresses it.'
        '\n  2. Ground truth here is Roboflow\'s annotation, not this project\'s. Its quality'
        '\n     is unknown and unaudited, unlike the possession and keypoint-audit ground'
        '\n     truth. A large measured direct error could be model error or annotation'
        '\n     error, and this measurement cannot distinguish them.'
        '\n  3. Reprojection is measured only on images already carrying at least five'
        '\n     confident keypoints, so the images where localisation error is most'
        '\n     suspected are precisely the ones excluded — the same coverage limitation the'
        '\n     clip measurement carries, and the reason the two are comparable.'
        '\n  4. COMPARE THE TEST SPLIT AND THE CLIPS ON METRES, NOT PIXELS. The split spans'
        '\n     six resolutions — 1280x720 (92 images), 2336x1752 (49), 1624x1234 (32),'
        '\n     640x360 (19), 1920x1080 (16), 2456x2054 (12) — whose diagonals differ by'
        '\n     roughly 6.8x, and the most common is only 42% of the split. A pooled median'
        '\n     of raw pixel residuals is therefore dominated by which resolutions happen to'
        '\n     be present rather than by localisation quality. The pixel column is retained'
        '\n     because the clip table reports raw pixels and normalising here would break'
        '\n     the comparability this script exists to provide, but it is indicative only.'
        '\n     Metres are resolution-independent, being measured on the court plane, and are'
        '\n     the column the clips-versus-test comparison must be made on. Direct error'
        '\n     carries a normalised column instead precisely because it has no clip-side'
        '\n     counterpart to stay comparable with.'
        '\n  5. THE TWO MEASUREMENTS USE DIFFERENT INFERENCE PATHS, UNVERIFIED. The clip'
        '\n     script infers in batches of 20 through CourtKeypoints.run_detection(), while'
        '\n     this script calls detect_keypoints() on one image at a time. Ultralytics'
        '\n     letterboxes a batch to a common input shape, so a batched prediction and a'
        '\n     single-image prediction are not guaranteed numerically identical — and this'
        '\n     script exists to produce a residual comparable with the clip table. Whether'
        '\n     the two paths agree HAS NOT BEEN CHECKED; it needs the checkpoint and the'
        '\n     dataset, so it is a JupyterHub task. Treat any small clips-versus-test'
        '\n     difference as potentially an artefact of this until it is verified.'
        '\n  6. No keypoint cache is written or fingerprint-validated by this script. Per-'
        '\n     image inference has no cache_path, unlike run_detection(), so every re-run'
        '\n     re-infers all 220 images and no stale-cache check protects these numbers the'
        '\n     way it protects the clip measurement.'
    )
    print(limitations)
    lines.append(limitations)

    write_csv(
        TEST_REPROJECTION_CSV,
        [
            'image', 'keypoint_index', 'confident_count',
            'residual_m', 'residual_px', 'dlt_conditioning',
        ],
        [row for measurement in measurements for row in measurement['reprojection_rows']],
    )
    write_csv(
        TEST_DIRECT_ERROR_CSV,
        [
            'image', 'keypoint_index', 'predicted_x', 'predicted_y', 'true_x', 'true_y',
            'error_px', 'error_normalised', 'image_width', 'image_height',
        ],
        [row for measurement in measurements for row in measurement['direct_rows']],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_SUMMARY_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nWrote {TEST_REPROJECTION_CSV}, {TEST_DIRECT_ERROR_CSV} and {TEST_SUMMARY_TXT}')


def main() -> None:
    """Measure every test-split image and write the results, keeping partial output if some images fail."""
    measurements: list[dict] = []
    skipped: list[tuple[str, str]] = []

    try:
        try:
            paths = image_paths()
        except FileNotFoundError as error:
            # An absent dataset is an operator error with an actionable fix, so
            # it is reported as one line rather than a traceback. The finally
            # below still runs and still declines to write anything.
            print(f'[test-split] {error}')
            return

        print(f'[test-split] {len(paths)} images in {TEST_IMAGE_DIR}')

        detector = CourtKeypoints(
            MODEL_PATH, keypoint_confidence_threshold=TEST_KEYPOINT_CONFIDENCE_THRESHOLD,
        )
        try:
            # Loaded explicitly rather than left to CourtKeypoints' lazy load
            # on first use. An absent checkpoint raises inside measure_image
            # otherwise, where the per-image handler below catches it once per
            # image: the operator would see ~220 identical lines and no results
            # in place of one actionable message. Every image would fail for
            # the same reason, so this is a whole-run failure and is reported
            # as one, matching the absent-dataset case above.
            detector.load_model()
        except (OSError, ValueError) as error:
            print(f'[test-split] {type(error).__name__}: {error}')
            return

        for image_path in paths:
            try:
                measurements.append(measure_image(image_path, detector))
            except (OSError, ValueError) as error:
                if is_checkpoint_level_error(error):
                    # Not a per-image fault. CourtKeypoints raises this from
                    # _parse_result at inference time, not from load_model, so
                    # loading eagerly cannot pre-empt it, but it is a property
                    # of the checkpoint, so every remaining image would raise
                    # it identically. Abandoning on the first occurrence
                    # reports it once instead of ~220 times.
                    print(f'[test-split] {type(error).__name__}: {error}')
                    print(
                        '[test-split] This is a property of the checkpoint rather than of one '
                        'image, so the run is abandoned instead of repeating it for every image.'
                    )
                    return
                # One unreadable image or malformed label must not abort the
                # measurement; the remaining images still produce results.
                print(f'[test-split] Skipping {image_path.name}: {type(error).__name__}: {error}')
                skipped.append((image_path.name, f'{type(error).__name__}: {error}'))
    finally:
        write_outputs(measurements, skipped)


if __name__ == '__main__':
    main()
