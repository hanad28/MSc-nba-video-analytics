"""Unit tests for the grouped test split measurement (scripts/measure_test_split.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.measure_court_keypoints as clip_measure
import scripts.measure_test_split as measure
from basketball.keypoints.court_keypoints import CourtKeypoints, Keypoint
from basketball.keypoints.court_template import NUM_KEYPOINTS, TEMPLATE_POINTS_M
from scripts.measure_court_keypoints import leave_one_out_residuals
from scripts.measure_test_split import (
    TEST_KEYPOINT_CONFIDENCE_THRESHOLD,
    confident_by_index,
    direct_errors,
    identity_table,
    measure_image,
    parse_label,
)

# The same exact similarity transform the clip measurement's tests use, so a
# residual computed here is comparable with one computed there by hand.
SCALE = 20.0
OFFSET = (100.0, 50.0)

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# The six resolutions actually present in the grouped test split, measured on
# JupyterHub: 1280x720 (92 images), 2336x1752 (49), 1624x1234 (32), 640x360
# (19), 1920x1080 (16), 2456x2054 (12). Diagonals differ by ~6.8x and the most
# common is only 42% of the split, so fixtures fixed at 1280x720 cannot tell a
# hardcoded diagonal apart from a correct one, the gap that let M6 survive.
REAL_RESOLUTIONS = (
    (1280, 720), (2336, 1752), (1624, 1234), (640, 360), (1920, 1080), (2456, 2054),
)
SMALLEST_RESOLUTION = (640, 360)
LARGEST_RESOLUTION = (2456, 2054)

# A real per-keypoint confidence vector measured on a broadcast frame. The
# distribution is close to binary (mostly exactly 1.0 or exactly 0.0 with a
# thin middle band), which a fixture using a single synthetic 0.9 never
# exercises. Six of these clear the 0.5 threshold, matching the measured
# median of 6 to 7 confident keypoints per frame.
REAL_CONFIDENCES = (
    1.0, 1.0, 1.0, 0.991, 0.306, 0.108, 0.204, 0.154, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)


def to_image(index: int) -> tuple[float, float]:
    """Map a keypoint's template position into image space under the exact synthetic transform."""
    x_m, y_m = TEMPLATE_POINTS_M[index]
    return (x_m * SCALE + OFFSET[0], y_m * SCALE + OFFSET[1])


def exact_frame(indices: list[int], confidence: float = 0.9) -> list[Keypoint]:
    """A frame whose confident keypoints sit exactly on the synthetic transform of their template positions."""
    frame = [Keypoint(index=i, x=0.0, y=0.0, confidence=0.0) for i in range(NUM_KEYPOINTS)]
    for index in indices:
        x, y = to_image(index)
        frame[index] = Keypoint(index=index, x=x, y=y, confidence=confidence)
    return frame


def label_text(
    positions: dict[int, tuple[float, float]],
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    visibility: float = 2.0,
) -> str:
    """Build one YOLO pose label line placing the given pixel positions, with every other keypoint absent."""
    values = ['0', '0.5', '0.5', '1.0', '1.0']
    for index in range(NUM_KEYPOINTS):
        if index in positions:
            x, y = positions[index]
            values += [f'{x / width:.10g}', f'{y / height:.10g}', f'{visibility:.10g}']
        else:
            values += ['0', '0', '0']
    return ' '.join(values)


class StubDetector:
    """A CourtKeypoints stand-in returning one caller-supplied frame per image and recording every call."""

    def __init__(self, per_image: dict[str, list[Keypoint]]) -> None:
        # Keyed by image name rather than a fixed list: a stub returning the
        # same output regardless of input cannot catch a bug that pairs an
        # image with another image's keypoints.
        self.per_image = per_image
        self.calls: list[np.ndarray] = []
        self.shapes: list[tuple[int, ...]] = []
        self.loads = 0

    def load_model(self) -> None:
        """Record the eager load the script performs before its per-image loop."""
        # Present because the real CourtKeypoints has it and main() now calls
        # it explicitly: a stub missing a method the caller uses cannot catch a
        # bug in how the caller uses it.
        self.loads += 1

    def detect_keypoints(self, frame: np.ndarray) -> list[Keypoint]:
        """Record the frame it was given and return the keypoints registered for it."""
        self.calls.append(frame)
        self.shapes.append(frame.shape)
        # Identified by the frame's own encoded marker so the returned
        # keypoints genuinely depend on which image was passed in.
        marker = int(frame[0, 0, 0])
        name = sorted(self.per_image)[marker]
        return self.per_image[name]


def write_image_and_label(
    directory: Path,
    name: str,
    marker: int,
    label: str,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
) -> Path:
    """Write one synthetic image carrying an identifying marker pixel, plus its label, and return the image path."""
    images = directory / 'images'
    labels = directory / 'labels'
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[0, 0, 0] = marker
    image_path = images / f'{name}.png'
    cv2.imwrite(str(image_path), frame)
    (labels / f'{name}.txt').write_text(label, encoding='utf-8')
    return image_path


def redirect(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Point the script's dataset and output paths at a temporary directory."""
    monkeypatch.setattr(measure, 'TEST_IMAGE_DIR', directory / 'images')
    monkeypatch.setattr(measure, 'TEST_LABEL_DIR', directory / 'labels')
    monkeypatch.setattr(measure, 'OUTPUT_DIR', directory / 'out')
    monkeypatch.setattr(measure, 'TEST_REPROJECTION_CSV', directory / 'out' / 'r.csv')
    monkeypatch.setattr(measure, 'TEST_DIRECT_ERROR_CSV', directory / 'out' / 'd.csv')
    monkeypatch.setattr(measure, 'TEST_SUMMARY_TXT', directory / 'out' / 's.txt')


# --- the stub itself reflects its input ----------------------------------

def test_the_detector_stub_returns_one_result_per_image_and_records_its_calls(monkeypatch, tmp_path):
    # Guards the fixture, not the code: three consecutive PRs in this phase had
    # their gap in a stub that produced fixed-length output regardless of
    # input, which cannot catch a shape bug.
    redirect(monkeypatch, tmp_path)
    first = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    second = write_image_and_label(tmp_path, 'b', 1, label_text({5: to_image(5)}))

    detector = StubDetector({'a.png': exact_frame([0]), 'b.png': exact_frame([5])})
    measure_image(first, detector)
    measure_image(second, detector)

    assert len(detector.calls) == 2
    assert detector.shapes == [(IMAGE_HEIGHT, IMAGE_WIDTH, 3)] * 2
    assert int(detector.calls[0][0, 0, 0]) == 0
    assert int(detector.calls[1][0, 0, 0]) == 1


# --- identity-level accounting -------------------------------------------

def test_a_confident_keypoint_the_label_marks_absent_is_a_false_positive():
    # The mutation this pins: counting it as a zero-error match would invent a
    # perfect prediction where the label says nothing exists.
    frame = exact_frame([0, 5])

    matched, false_positives, false_negatives = direct_errors(
        frame, {0: to_image(0)}, IMAGE_WIDTH, IMAGE_HEIGHT,
    )

    assert false_positives == 1
    assert false_negatives == 0
    assert [entry[0] for entry in matched] == [0]
    assert 5 not in [entry[0] for entry in matched]


def test_a_labelled_keypoint_the_model_missed_is_a_false_negative():
    frame = exact_frame([0])

    matched, false_positives, false_negatives = direct_errors(
        frame, {0: to_image(0), 9: to_image(9)}, IMAGE_WIDTH, IMAGE_HEIGHT,
    )

    assert false_negatives == 1
    assert false_positives == 0
    assert [entry[0] for entry in matched] == [0]


def test_a_sub_threshold_prediction_on_a_labelled_keypoint_is_a_false_negative_not_a_match():
    # Confidence just below the threshold: present in the model's output but
    # not trusted, so it must be a false negative rather than a match whose
    # error would silently enter the localisation medians.
    frame = exact_frame([0])
    frame[9] = Keypoint(
        index=9, x=to_image(9)[0], y=to_image(9)[1],
        confidence=TEST_KEYPOINT_CONFIDENCE_THRESHOLD - 0.01,
    )

    matched, false_positives, false_negatives = direct_errors(
        frame, {0: to_image(0), 9: to_image(9)}, IMAGE_WIDTH, IMAGE_HEIGHT,
    )

    assert false_negatives == 1
    assert [entry[0] for entry in matched] == [0]


def test_a_keypoint_neither_predicted_nor_labelled_counts_as_neither():
    frame = exact_frame([0])

    matched, false_positives, false_negatives = direct_errors(
        frame, {0: to_image(0)}, IMAGE_WIDTH, IMAGE_HEIGHT,
    )

    assert (len(matched), false_positives, false_negatives) == (1, 0, 0)


def test_the_identity_table_reconciles_matched_and_missed_against_the_labelled_total(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(
        tmp_path, 'a', 0, label_text({0: to_image(0), 5: to_image(5), 9: to_image(9)}),
    )
    detector = StubDetector({'a.png': exact_frame([0, 5, 15])})

    row = identity_table([measure_image(path, detector)])[0]

    # images, matched, fp, fn, labelled_total, matched+fn, reconciles, ...
    assert row[1:7] == ['2', '1', '1', '3', '3', 'yes']
    assert row[6] == 'yes' and row[9] == 'yes'


# --- direct error arithmetic ---------------------------------------------

def test_direct_error_equals_a_known_displacement():
    frame = exact_frame([8])
    displacement = 12.0
    frame[8] = Keypoint(
        index=8, x=to_image(8)[0] + displacement, y=to_image(8)[1], confidence=0.9,
    )

    matched, _, _ = direct_errors(frame, {8: to_image(8)}, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert matched[0][5] == pytest.approx(displacement)


def test_direct_error_is_euclidean_not_axis_wise():
    # A 3-4-5 triangle: an implementation summing the axes would report 7.
    frame = exact_frame([8])
    frame[8] = Keypoint(
        index=8, x=to_image(8)[0] + 3.0, y=to_image(8)[1] + 4.0, confidence=0.9,
    )

    matched, _, _ = direct_errors(frame, {8: to_image(8)}, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert matched[0][5] == pytest.approx(5.0)


def test_normalised_error_is_the_pixel_error_over_the_image_diagonal():
    frame = exact_frame([8])
    frame[8] = Keypoint(index=8, x=to_image(8)[0] + 10.0, y=to_image(8)[1], confidence=0.9)

    matched, _, _ = direct_errors(frame, {8: to_image(8)}, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert matched[0][6] == pytest.approx(10.0 / np.hypot(IMAGE_WIDTH, IMAGE_HEIGHT))


def test_normalisation_is_invariant_to_a_uniform_rescaling_of_image_and_labels():
    # The reason the normalised column exists: the split spans source videos at
    # differing resolutions, so the same geometry at a different size must
    # score the same even though its raw pixel error does not.
    factor = 3.0
    frame = exact_frame([8])
    frame[8] = Keypoint(index=8, x=to_image(8)[0] + 10.0, y=to_image(8)[1], confidence=0.9)
    baseline, _, _ = direct_errors(frame, {8: to_image(8)}, IMAGE_WIDTH, IMAGE_HEIGHT)

    scaled = exact_frame([8])
    scaled[8] = Keypoint(
        index=8, x=(to_image(8)[0] + 10.0) * factor, y=to_image(8)[1] * factor, confidence=0.9,
    )
    truth_scaled = (to_image(8)[0] * factor, to_image(8)[1] * factor)
    rescaled, _, _ = direct_errors(
        scaled, {8: truth_scaled}, int(IMAGE_WIDTH * factor), int(IMAGE_HEIGHT * factor),
    )

    assert rescaled[0][5] == pytest.approx(baseline[0][5] * factor)
    assert rescaled[0][6] == pytest.approx(baseline[0][6])


# --- label parsing --------------------------------------------------------

def test_parse_label_scales_normalised_coordinates_by_the_images_own_size():
    text = label_text({3: (640.0, 360.0)})

    parsed = parse_label(text, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert set(parsed) == {3}
    assert parsed[3] == pytest.approx((640.0, 360.0))


def test_parse_label_scales_by_the_supplied_size_not_a_fixed_resolution():
    # The split spans source videos at differing resolutions, so a label must
    # be scaled by ITS OWN image's size. Every other fixture here is 1280x720,
    # which makes a hardcoded 1280x720 indistinguishable from correct; this
    # test is at a deliberately different size for that reason.
    width, height = 640, 480
    text = label_text({3: (160.0, 120.0)}, width=width, height=height)

    parsed = parse_label(text, width, height)

    assert parsed[3] == pytest.approx((160.0, 120.0))


def test_direct_error_at_a_non_default_resolution_normalises_by_that_images_diagonal():
    # Guards the same hardcoding at the error boundary rather than the parsing
    # one: a fixed diagonal would silently rescale every error on any image
    # that is not 1280x720.
    width, height = 640, 480
    frame = exact_frame([8])
    frame[8] = Keypoint(index=8, x=110.0, y=60.0, confidence=0.9)

    matched, _, _ = direct_errors(frame, {8: (100.0, 60.0)}, width, height)

    assert matched[0][5] == pytest.approx(10.0)
    assert matched[0][6] == pytest.approx(10.0 / np.hypot(width, height))


@pytest.mark.parametrize('width,height', REAL_RESOLUTIONS)
def test_direct_error_normalises_by_the_diagonal_at_every_real_split_resolution(width, height):
    # Parametrised over the six resolutions the split actually holds rather
    # than one: a hardcoded diagonal is indistinguishable from correct at
    # whichever single resolution a fixture happens to pick.
    frame = exact_frame([8])
    frame[8] = Keypoint(index=8, x=110.0, y=60.0, confidence=0.9)

    matched, _, _ = direct_errors(frame, {8: (100.0, 60.0)}, width, height)

    assert matched[0][5] == pytest.approx(10.0)
    assert matched[0][6] == pytest.approx(10.0 / np.hypot(width, height))


def test_the_same_pixel_error_normalises_differently_across_the_splits_extremes():
    # The measurable consequence of the 6.8x diagonal spread: an identical
    # pixel error is a very different fraction of the smallest image than of
    # the largest, which is why the pooled raw-pixel column cannot be compared
    # across resolutions.
    frame = exact_frame([8])
    frame[8] = Keypoint(index=8, x=110.0, y=60.0, confidence=0.9)

    small, _, _ = direct_errors(frame, {8: (100.0, 60.0)}, *SMALLEST_RESOLUTION)
    large, _, _ = direct_errors(frame, {8: (100.0, 60.0)}, *LARGEST_RESOLUTION)

    assert small[0][5] == pytest.approx(large[0][5])
    ratio = small[0][6] / large[0][6]
    assert ratio == pytest.approx(
        np.hypot(*LARGEST_RESOLUTION) / np.hypot(*SMALLEST_RESOLUTION)
    )
    assert ratio > 4.0


def test_parse_label_treats_occluded_keypoints_as_present():
    # Ultralytics trains on visibility 1 and 2 alike, and both mark a landmark
    # that exists in the image, so v == 1 must not be read as absent.
    text = label_text({3: (640.0, 360.0)}, visibility=1.0)

    assert set(parse_label(text, IMAGE_WIDTH, IMAGE_HEIGHT)) == {3}


def test_parse_label_excludes_absent_keypoints():
    text = label_text({3: (640.0, 360.0)}, visibility=0.0)

    assert parse_label(text, IMAGE_WIDTH, IMAGE_HEIGHT) == {}


def test_parse_label_raises_on_a_short_keypoint_list():
    truncated = ' '.join(['0', '0.5', '0.5', '1.0', '1.0'] + ['0.1', '0.1', '2'] * 4)

    with pytest.raises(ValueError, match='expected 18'):
        parse_label(truncated, IMAGE_WIDTH, IMAGE_HEIGHT)


def test_parse_label_returns_nothing_for_an_empty_label():
    assert parse_label('\n  \n', IMAGE_WIDTH, IMAGE_HEIGHT) == {}


# --- the shared reprojection code really is shared ------------------------

def test_reprojection_matches_the_clip_measurement_on_identical_input(monkeypatch, tmp_path):
    # Pins that this script reuses measure_court_keypoints' implementation
    # rather than reimplementing it. A reimplementation would silently
    # reintroduce the collinearity specification error that took five review
    # rounds to find.
    redirect(monkeypatch, tmp_path)
    frame = exact_frame([0, 5, 8, 9, 10, 15])
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))

    result = measure_image(path, StubDetector({'a.png': frame}))
    expected, _, _ = leave_one_out_residuals(frame)

    assert len(result['reprojection_rows']) == len(expected)
    for row, (index, metres, pixels, _) in zip(result['reprojection_rows'], expected):
        assert row[1] == str(index)
        # Compared at the CSV's own six-decimal precision: the rows are the
        # persisted strings, so full float equality would fail on rounding
        # rather than on any disagreement between the two implementations.
        assert float(row[3]) == pytest.approx(metres, abs=1e-6)
        assert float(row[4]) == pytest.approx(pixels, abs=1e-6)


def test_the_reprojection_bucket_table_uses_the_shared_bucket_boundaries(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))

    rows = measure.reprojection_bucket_table([result])

    assert rows and all(len(row) == 7 for row in rows)
    assert {row[1] for row in rows} == {'6'}


# --- reprojection accounting ---------------------------------------------

def test_the_four_image_categories_appear_as_separate_coverage_columns(monkeypatch, tmp_path):
    # Mirrors the clip measurement's own coverage test. Without this table the
    # collected rejection counts were stored and never reported, so the
    # reprojection n could not be reconciled against the images attempted.
    redirect(monkeypatch, tmp_path)
    label = label_text({0: to_image(0)})
    cases = {
        'a.png': exact_frame([0, 5, 10, 15]),        # skipped, below five
        'b.png': exact_frame([0, 1, 2, 3, 4, 5]),    # lost to under-determined sets
        'c.png': exact_frame([0, 5, 8, 9, 10, 15]),  # scored
    }
    detector = StubDetector(cases)
    measurements = [
        measure_image(write_image_and_label(tmp_path, name[:-4], i, label), detector)
        for i, name in enumerate(sorted(cases))
    ]

    row = measure.coverage_table(measurements)[0]

    # images, skipped, lost_degenerate, lost_collinear, scored, accounted, reconciles
    assert row[0:7] == ['3', '1', '0', '1', '1', '3', 'yes']
    assert row[8] == '6', 'six collinear fit rejections from the baseline-only image'


def test_the_reconciles_column_reports_no_when_a_category_is_unrecognised(monkeypatch, tmp_path):
    # Without this the column is a tautology: summing four buckets that every
    # measurement increments exactly one of always equals the image count, so
    # 'yes' would print unconditionally and verify nothing. An unrecognised
    # category is the only way the sum can fall short, so it is the only thing
    # that makes the check real.
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    good = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))
    corrupt = dict(good, category='not_a_real_category')

    assert measure.coverage_table([good])[0][6] == 'yes'

    row = measure.coverage_table([good, corrupt])[0]

    assert row[6].startswith('NO')
    assert '1 unknown' in row[6]
    assert row[5] == '1', 'the unrecognised measurement must not be counted as accounted'


def test_the_coverage_table_reports_the_rejection_counts_it_collects(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 1, 2, 3, 4, 5])}))

    row = measure.coverage_table([result])[0]

    assert result['collinear_fits'] == 6
    assert row[8] == str(result['collinear_fits'])
    assert row[7] == str(result['degenerate_fits'])


def test_an_image_below_five_confident_keypoints_is_categorised_as_skipped(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))

    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 10, 15])}))

    assert result['category'] == 'skipped'
    assert result['reprojection_rows'] == []


def test_the_resolution_table_counts_scored_images_per_distinct_resolution(monkeypatch, tmp_path):
    # The table that justifies calling the pixel column indicative: it must
    # show the spread rather than assert it.
    redirect(monkeypatch, tmp_path)
    label = label_text({0: to_image(0)})
    frame = exact_frame([0, 5, 8, 9, 10, 15])
    detector = StubDetector({'a.png': frame, 'b.png': frame, 'c.png': frame})
    measurements = [
        measure_image(
            write_image_and_label(tmp_path, name, i, label, width=width, height=height), detector,
        )
        for i, (name, (width, height)) in enumerate([
            ('a', SMALLEST_RESOLUTION), ('b', LARGEST_RESOLUTION), ('c', SMALLEST_RESOLUTION),
        ])
    ]

    rows = measure.resolution_table(measurements)

    assert [row[0] for row in rows] == ['640x360', '2456x2054']
    assert [row[1] for row in rows] == ['2', '1']
    # Six residuals per scored image, so observations are 12 against 6.
    assert [row[2] for row in rows] == ['12', '6']
    # Compared at the table's own three-decimal rendering, not full precision.
    assert float(rows[0][3]) == pytest.approx(12 / 18, abs=1e-3)
    assert float(rows[0][4]) == pytest.approx(np.hypot(*SMALLEST_RESOLUTION), abs=0.1)


def test_the_resolution_table_counts_only_images_that_produced_residuals(monkeypatch, tmp_path):
    # A skipped or lost image contributes no row to the pooled pixel column, so
    # counting it would describe the composition of a set the pooled median is
    # not drawn from. The previous fixture used only scored images and so could
    # not tell the two apart.
    redirect(monkeypatch, tmp_path)
    label = label_text({0: to_image(0)})
    cases = {
        'a.png': exact_frame([0, 5, 8, 9, 10, 15]),  # scored, smallest
        'b.png': exact_frame([0, 5, 10, 15]),        # skipped, largest
        'c.png': exact_frame([0, 1, 2, 3, 4, 5]),    # lost collinear, largest
    }
    detector = StubDetector(cases)
    resolutions = [SMALLEST_RESOLUTION, LARGEST_RESOLUTION, LARGEST_RESOLUTION]
    measurements = [
        measure_image(
            write_image_and_label(
                tmp_path, name[:-4], i, label, width=width, height=height,
            ),
            detector,
        )
        for i, (name, (width, height)) in enumerate(zip(sorted(cases), resolutions))
    ]

    assert [m['category'] for m in measurements] == ['scored', 'skipped', 'lost_collinear']

    rows = measure.resolution_table(measurements)

    # Only the scored image appears; the largest resolution is absent entirely
    # despite carrying two of the three images.
    assert [row[0] for row in rows] == ['640x360']
    assert rows[0][1] == '1'
    assert rows[0][2] == '6'
    assert float(rows[0][3]) == pytest.approx(1.0)


def test_the_resolution_share_is_weighted_by_observation_not_by_image(monkeypatch, tmp_path):
    # One image contributes several held-out residuals, so it is the
    # observation count that determines the pooled median. Two resolutions with
    # equal image counts but different residual counts must not report equal
    # shares.
    redirect(monkeypatch, tmp_path)
    label = label_text({0: to_image(0)})
    cases = {
        'a.png': exact_frame([0, 5, 8, 9, 10, 15]),      # 6 residuals
        'b.png': exact_frame([0, 5, 8, 9, 10, 15, 6, 7]),  # 8 residuals
    }
    detector = StubDetector(cases)
    measurements = [
        measure_image(
            write_image_and_label(
                tmp_path, name[:-4], i, label, width=width, height=height,
            ),
            detector,
        )
        for i, (name, (width, height)) in enumerate(
            zip(sorted(cases), [SMALLEST_RESOLUTION, LARGEST_RESOLUTION])
        )
    ]

    rows = {row[0]: row for row in measure.resolution_table(measurements)}

    # One image each, so an image-weighted share would report 0.5 for both.
    assert rows['640x360'][1] == rows['2456x2054'][1] == '1'
    assert rows['640x360'][2] == '6'
    assert rows['2456x2054'][2] == '8'
    assert float(rows['640x360'][3]) == pytest.approx(6 / 14, abs=1e-3)
    assert float(rows['2456x2054'][3]) == pytest.approx(8 / 14, abs=1e-3)
    assert float(rows['640x360'][3]) != pytest.approx(0.5, abs=1e-3)


def test_a_real_confidence_vector_yields_the_measured_number_of_confident_keypoints():
    # Guards against a fixture that only ever uses one synthetic confidence:
    # the real distribution is near-binary with a thin middle band, and the
    # middle band is exactly where a threshold comparison can go wrong.
    frame = [
        Keypoint(index=i, x=to_image(i)[0], y=to_image(i)[1], confidence=confidence)
        for i, confidence in enumerate(REAL_CONFIDENCES)
    ]

    confident = confident_by_index(frame)

    assert sorted(confident) == [0, 1, 2, 3, 8, 9]
    # Within the measured per-frame range of 1 to 8, median 6 to 7.
    assert 1 <= len(confident) <= 8


def test_a_confident_keypoint_can_still_be_badly_wrong():
    # Confidence does not imply correctness: a keypoint measured a metre from
    # its true position carried confidence exactly 1.0. Nothing in this script
    # may treat high confidence as evidence of a small error.
    frame = exact_frame([8], confidence=1.0)
    displaced = 200.0
    frame[8] = Keypoint(
        index=8, x=to_image(8)[0] + displaced, y=to_image(8)[1], confidence=1.0,
    )

    matched, false_positives, false_negatives = direct_errors(
        frame, {8: to_image(8)}, IMAGE_WIDTH, IMAGE_HEIGHT,
    )

    assert (false_positives, false_negatives) == (0, 0)
    assert matched[0][5] == pytest.approx(displaced)


# --- DLT conditioning -----------------------------------------------------

def test_conditioning_is_persisted_as_a_column_on_every_residual(monkeypatch, tmp_path):
    # The clip script persists this column and reports its own conditioning
    # table; discarding it here would leave the two artefacts not
    # column-comparable, which is the whole purpose of this script.
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))

    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))

    assert result['reprojection_rows']
    for row in result['reprojection_rows']:
        assert len(row) == 6
        conditioning = float(row[5])
        assert conditioning == conditioning, 'conditioning must not be NaN for a real fit'
        assert conditioning > 0.0


def test_the_persisted_conditioning_matches_the_shared_implementation(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    frame = exact_frame([0, 5, 8, 9, 10, 15])
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))

    result = measure_image(path, StubDetector({'a.png': frame}))
    expected, _, _ = leave_one_out_residuals(frame)

    for row, (_, _, _, conditioning) in zip(result['reprojection_rows'], expected):
        assert float(row[5]) == pytest.approx(conditioning, rel=1e-6)


def test_the_conditioning_table_matches_the_clip_scripts_column_shape(monkeypatch, tmp_path):
    # The two scripts' reprojection rows differ by a leading frame/image
    # column, so the clip table is built from clip-shaped rows rather than
    # from this script's. What must match is the rendered table: same column
    # count, same bucket labels, same values for the same fits.
    redirect(monkeypatch, tmp_path)
    frame = exact_frame([0, 5, 8, 9, 10, 15])
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': frame}))

    residuals, _, _ = leave_one_out_residuals(frame)
    clip_shaped = {
        'clip': 'clip_1',
        'reprojection_rows': [
            ['clip_1', '0', str(index), '6', f'{metres:.6f}', f'{pixels:.6f}', f'{cond:.6e}']
            for index, metres, pixels, cond in residuals
        ],
    }

    rows = measure.conditioning_table([result])
    clip_rows = [
        row for row in clip_measure.conditioning_table([clip_shaped]) if row[0] == 'clip_1'
    ]

    assert rows and all(len(row) == 8 for row in rows)
    assert len(rows) == len(clip_rows)
    # Same bucket labels, counts and distribution values, differing only in the
    # scope label, so the two artefacts can be read side by side.
    for ours, theirs in zip(rows, clip_rows):
        assert ours[1:] == theirs[1:]
    assert rows[0][1] == '6'


def test_the_conditioning_table_is_never_pooled_across_buckets(monkeypatch, tmp_path):
    # The DLT matrix shape changes with the fitting-point count, so sigma[7] is
    # a different quantity at 4 fitting points than at 5 or more. Merging the
    # buckets would average two different measurements.
    redirect(monkeypatch, tmp_path)
    label = label_text({0: to_image(0)})
    cases = {
        'a.png': exact_frame([0, 5, 8, 9, 10, 15]),
        'b.png': exact_frame([0, 5, 8, 9, 10, 15, 6, 7]),
    }
    detector = StubDetector(cases)
    measurements = [
        measure_image(write_image_and_label(tmp_path, name[:-4], i, label), detector)
        for i, name in enumerate(sorted(cases))
    ]

    buckets = [row[1] for row in measure.conditioning_table(measurements)]

    assert buckets == ['6', '8']
    assert len(buckets) == len(set(buckets))


def test_a_nan_conditioning_value_is_excluded_rather_than_propagated(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))
    result['reprojection_rows'][0][5] = 'nan'

    rows = measure.conditioning_table([result])

    assert rows[0][2] == '5', 'the NaN row must be dropped, not counted'
    assert all(value != 'nan' for value in rows[0][3:])


def test_the_reprojection_csv_carries_the_conditioning_column(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))

    measure.write_outputs([result], [])
    header = (tmp_path / 'out' / 'r.csv').read_text(encoding='utf-8').splitlines()[0]

    assert header.split(',') == [
        'image', 'keypoint_index', 'confident_count',
        'residual_m', 'residual_px', 'dlt_conditioning',
    ]


# --- threshold coupling ---------------------------------------------------

def test_the_per_keypoint_threshold_is_the_clip_measurements_own_constant():
    # Bound rather than redeclared: the imported leave_one_out_residuals()
    # filters with the clip measurement's constant, so a local copy at the same
    # value would let the two drift apart silently, desynchronising the
    # residuals from the recorded confident_count and direct-error matching.
    assert measure.TEST_KEYPOINT_CONFIDENCE_THRESHOLD is clip_measure.KEYPOINT_CONFIDENCE_THRESHOLD


def test_changing_the_shared_threshold_moves_this_scripts_matching(monkeypatch):
    # Proves the binding is live rather than a coincidence of equal literals:
    # confidences here sit between the two thresholds, so raising the shared
    # constant must change which keypoints this script treats as confident.
    frame = exact_frame([0, 5], confidence=0.6)

    assert set(confident_by_index(frame)) == {0, 5}

    monkeypatch.setattr(measure, 'TEST_KEYPOINT_CONFIDENCE_THRESHOLD', 0.7)

    assert set(confident_by_index(frame)) == set()


# --- write refusal --------------------------------------------------------

def test_nothing_is_written_when_no_image_could_be_scored(monkeypatch, tmp_path, capsys):
    # Same rule the clip measurement adopted after review: an empty file must
    # never masquerade as a result.
    redirect(monkeypatch, tmp_path)

    measure.write_outputs([], [('a.png', 'OSError: unreadable')])

    assert not (tmp_path / 'out').exists()
    assert 'nothing was written' in capsys.readouterr().out


def test_the_clip_measurements_own_artefacts_are_never_written_by_this_script():
    # These two scripts share an output directory, so a copied constant would
    # silently overwrite the clip measurement's results with test-split rows.
    theirs = {
        clip_measure.SUFFICIENCY_CSV, clip_measure.REPROJECTION_CSV,
        clip_measure.STABILITY_CSV, clip_measure.SUMMARY_TXT,
    }
    ours = {
        measure.TEST_REPROJECTION_CSV, measure.TEST_DIRECT_ERROR_CSV, measure.TEST_SUMMARY_TXT,
    }

    assert not (theirs & ours)


def test_outputs_are_written_when_at_least_one_image_scored(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(
        tmp_path, 'a', 0, label_text({0: to_image(0), 5: to_image(5)}),
    )
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))

    measure.write_outputs([result], [])

    assert (tmp_path / 'out' / 'r.csv').exists()
    assert (tmp_path / 'out' / 'd.csv').exists()
    assert (tmp_path / 'out' / 's.txt').exists()


def test_the_summary_states_both_confounds(monkeypatch, tmp_path):
    # Required to be printed with the results rather than discovered later, so
    # a reader of the artefact cannot miss them.
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))

    measure.write_outputs([result], [])
    summary = (tmp_path / 'out' / 's.txt').read_text(encoding='utf-8')

    assert 'equalis' in summary
    assert 'Roboflow' in summary


def test_the_summary_marks_metres_comparable_and_pixels_indicative(monkeypatch, tmp_path):
    # The pixel column stays raw so it matches the clip table's units, which
    # makes it easy to compare across resolutions by mistake. Both the table
    # heading and the limitations must say not to.
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))

    measure.write_outputs([result], [])
    summary = (tmp_path / 'out' / 's.txt').read_text(encoding='utf-8')

    assert 'median_px_indicative' in summary
    assert 'COMPARE ON METRES' in summary
    assert 'COMPARE THE TEST SPLIT AND THE CLIPS ON METRES, NOT PIXELS' in summary
    # The measured spread is the reason, so the figures must be present.
    assert '6.8x' in summary
    assert '2456x2054' in summary


def test_the_summary_records_the_inference_path_difference_as_unverified(monkeypatch, tmp_path):
    # The clip script batches at 20 through run_detection while this one infers
    # per image, and ultralytics letterboxes a batch to a common input shape,
    # so the two paths are not guaranteed numerically identical. The script
    # exists to produce a comparable residual, so the caveat must be in the
    # artefact rather than only in a review thread.
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))

    measure.write_outputs([result], [])
    summary = (tmp_path / 'out' / 's.txt').read_text(encoding='utf-8')

    assert 'DIFFERENT INFERENCE PATHS, UNVERIFIED' in summary
    assert 'letterbox' in summary
    assert 'HAS NOT BEEN CHECKED' in summary
    assert 'batches of 20' in summary


def test_the_summary_records_that_no_keypoint_cache_is_written(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))

    measure.write_outputs([result], [])
    summary = (tmp_path / 'out' / 's.txt').read_text(encoding='utf-8')

    assert 'No keypoint cache is written or fingerprint-validated' in summary
    assert 're-infers all 220 images' in summary


def test_the_summary_carries_the_reprojection_accounting_table(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    result = measure_image(path, StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])}))

    measure.write_outputs([result], [])
    summary = (tmp_path / 'out' / 's.txt').read_text(encoding='utf-8')

    assert 'lost_all_fits_degenerate' in summary
    assert 'reconciles' in summary
    assert 'diagonal_px' in summary


# --- per-image failure handling ------------------------------------------

def test_an_image_with_no_label_is_reported_rather_than_scored(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    path = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    (tmp_path / 'labels' / 'a.txt').unlink()

    with pytest.raises(OSError, match='no label'):
        measure_image(path, StubDetector({'a.png': exact_frame([0])}))


def test_an_unreadable_image_raises_rather_than_scoring_zeros(monkeypatch, tmp_path):
    redirect(monkeypatch, tmp_path)
    images = tmp_path / 'images'
    images.mkdir(parents=True, exist_ok=True)
    broken = images / 'broken.png'
    broken.write_bytes(b'not an image')

    with pytest.raises(OSError, match='Could not read'):
        measure_image(broken, StubDetector({}))


def test_one_failing_image_does_not_abort_the_run(monkeypatch, tmp_path, capsys):
    redirect(monkeypatch, tmp_path)
    good = write_image_and_label(tmp_path, 'a', 0, label_text({0: to_image(0)}))
    broken = tmp_path / 'images' / 'b.png'
    broken.write_bytes(b'not an image')

    detector = StubDetector({'a.png': exact_frame([0, 5, 8, 9, 10, 15])})
    monkeypatch.setattr(measure, 'CourtKeypoints', lambda *a, **k: detector)
    measure.main()

    output = capsys.readouterr().out
    assert 'Skipping b.png' in output
    assert (tmp_path / 'out' / 'r.csv').exists()
    # Loaded once for the whole run rather than lazily per image.
    assert detector.loads == 1


def test_an_absent_dataset_is_reported_as_one_line_and_writes_nothing(monkeypatch, tmp_path, capsys):
    # The local state of this repo: the split lives on JupyterHub. An operator
    # error with an actionable fix should not surface as a traceback, and must
    # not leave empty artefacts behind either.
    redirect(monkeypatch, tmp_path / 'absent')

    measure.main()

    output = capsys.readouterr().out
    assert 'split_court_keypoints.py' in output
    assert not (tmp_path / 'absent' / 'out').exists()


def test_a_missing_checkpoint_is_reported_once_rather_than_once_per_image(monkeypatch, tmp_path, capsys):
    # CourtKeypoints loads lazily, so an absent checkpoint would otherwise
    # raise inside measure_image for every image, where the per-image handler
    # catches it: ~220 identical lines and no results in place of one
    # actionable message.
    redirect(monkeypatch, tmp_path)
    label = label_text({0: to_image(0)})
    for i, name in enumerate(('a', 'b', 'c', 'd', 'e')):
        write_image_and_label(tmp_path, name, i, label)

    class MissingCheckpointDetector:
        def __init__(self) -> None:
            self.loads = 0

        def load_model(self) -> None:
            self.loads += 1
            raise FileNotFoundError(
                'Court keypoint weights do not exist: models/keypoints.pt.'
            )

        def detect_keypoints(self, frame: np.ndarray) -> list[Keypoint]:
            raise AssertionError('inference must not be attempted without a checkpoint')

    detector = MissingCheckpointDetector()
    monkeypatch.setattr(measure, 'CourtKeypoints', lambda *a, **k: detector)

    measure.main()

    output = capsys.readouterr().out
    assert output.count('Court keypoint weights do not exist') == 1
    assert 'Skipping' not in output
    assert detector.loads == 1
    assert not (tmp_path / 'out').exists()


def test_a_checkpoint_without_confidences_abandons_the_run_after_one_report(monkeypatch, tmp_path, capsys):
    # This one is raised at inference time rather than by load_model, so an
    # eager load cannot pre-empt it, but it is still a property of the
    # checkpoint, so every remaining image would raise it identically.
    redirect(monkeypatch, tmp_path)
    label = label_text({0: to_image(0)})
    for i, name in enumerate(('a', 'b', 'c', 'd', 'e')):
        write_image_and_label(tmp_path, name, i, label)

    class NoConfidenceDetector:
        def __init__(self) -> None:
            self.calls = 0

        def load_model(self) -> None:
            return None

        def detect_keypoints(self, frame: np.ndarray) -> list[Keypoint]:
            self.calls += 1
            raise ValueError(
                f'Checkpoint models/keypoints.pt {measure.NO_CONFIDENCE_MARKER}, '
                f'so it cannot be used by this stage.'
            )

    detector = NoConfidenceDetector()
    monkeypatch.setattr(measure, 'CourtKeypoints', lambda *a, **k: detector)

    measure.main()

    output = capsys.readouterr().out
    assert output.count(measure.NO_CONFIDENCE_MARKER) == 1
    assert detector.calls == 1, 'the run must stop after the first image, not try all five'
    assert 'property of the checkpoint' in output


def test_the_no_confidence_marker_matches_the_message_court_keypoints_actually_raises(tmp_path):
    # Pins the substring against the real class rather than a copy of it: the
    # marker is matched on message text, so a reworded message would silently
    # degrade this script back to one report per image.
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'not a real checkpoint')
    detector = CourtKeypoints(str(checkpoint))

    class ConfidencelessKeypoints:
        xy = np.zeros((1, 18, 2))
        conf = None

    class Result:
        keypoints = ConfidencelessKeypoints()

    with pytest.raises(ValueError) as raised:
        detector._parse_result(Result())

    assert measure.NO_CONFIDENCE_MARKER in str(raised.value)
    assert measure.is_checkpoint_level_error(raised.value)


def test_an_ordinary_per_image_error_is_not_treated_as_checkpoint_level():
    # The predicate must not swallow the per-image failures the loop is
    # supposed to skip and continue past.
    assert not measure.is_checkpoint_level_error(ValueError('Label holds 4 keypoints'))
    assert not measure.is_checkpoint_level_error(OSError('Could not read image a.png'))


def test_confident_by_index_applies_this_scripts_own_threshold():
    frame = exact_frame([0], confidence=TEST_KEYPOINT_CONFIDENCE_THRESHOLD)
    frame[5] = Keypoint(
        index=5, x=1.0, y=1.0, confidence=TEST_KEYPOINT_CONFIDENCE_THRESHOLD - 0.001,
    )

    assert set(confident_by_index(frame)) == {0}
