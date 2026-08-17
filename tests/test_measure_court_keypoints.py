"""Unit tests for Stage 7's measurement script (scripts/measure_court_keypoints.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.measure_court_keypoints as measure
from basketball.keypoints.court_keypoints import Keypoint
from basketball.keypoints.court_template import TEMPLATE_POINTS_M
from scripts.measure_court_keypoints import (
    KEYPOINT_CONFIDENCE_THRESHOLD,
    below_threshold_runs,
    bucket_for,
    confident_keypoints,
    dlt_conditioning,
    hartley_normalise,
    fit_homography,
    has_general_position_quadruple,
    leave_one_out_residuals,
    measure_clip,
)

# A similarity transform from metres to pixels: exact, invertible, and simple
# enough that an expected residual can be computed by hand.
SCALE = 20.0
OFFSET = (100.0, 50.0)


def to_image(index: int) -> tuple[float, float]:
    """Map a keypoint's template position into image space under the exact synthetic transform."""
    x_m, y_m = TEMPLATE_POINTS_M[index]
    return (x_m * SCALE + OFFSET[0], y_m * SCALE + OFFSET[1])


def exact_frame(indices: list[int], confidence: float = 0.9) -> list[Keypoint]:
    """A frame whose confident keypoints sit exactly on the synthetic transform of their template positions."""
    frame = [Keypoint(index=i, x=0.0, y=0.0, confidence=0.0) for i in range(18)]
    for index in indices:
        x, y = to_image(index)
        frame[index] = Keypoint(index=index, x=x, y=y, confidence=confidence)
    return frame


# --- residual arithmetic -------------------------------------------------

def test_points_on_an_exact_transform_have_zero_residual():
    frame = exact_frame([0, 5, 10, 15, 8, 9])

    residuals, degenerate, collinear = leave_one_out_residuals(frame)

    assert degenerate == 0
    assert len(residuals) == 6
    # findHomography solves by SVD, so an exact fit lands within float noise
    # rather than at exactly zero -- 1e-4 m is 0.1 mm on a 28.65 m court.
    assert all(metres == pytest.approx(0.0, abs=1e-4) for _, metres, _, _ in residuals)
    assert all(pixels == pytest.approx(0.0, abs=1e-3) for _, _, pixels, _ in residuals)


def test_displacing_one_point_yields_a_residual_equal_to_the_displacement():
    # The other points still define the exact transform, so the held-out
    # point's pixel residual is precisely the displacement applied to it.
    frame = exact_frame([0, 5, 10, 15, 8, 9])
    moved_index = 8
    displacement_px = 12.0
    original = frame[moved_index]
    frame[moved_index] = Keypoint(
        index=moved_index, x=original.x + displacement_px, y=original.y, confidence=0.9,
    )

    residuals, degenerate, collinear = leave_one_out_residuals(frame)
    by_index = {index: (metres, pixels) for index, metres, pixels, _ in residuals}

    assert degenerate == 0
    assert by_index[moved_index][1] == pytest.approx(displacement_px, abs=1e-4)
    # And in metres, the same displacement divided by the transform's scale.
    assert by_index[moved_index][0] == pytest.approx(displacement_px / SCALE, abs=1e-4)


# --- leave-one-out exclusion --------------------------------------------

def test_the_held_out_keypoint_is_excluded_from_its_own_fit():
    # Point 8 is displaced far enough that including it in the fit would drag
    # the homography toward it and shrink its own residual. Excluding it keeps
    # the fit exact, so the residual equals the full displacement.
    frame = exact_frame([0, 5, 10, 15, 8])
    displacement_px = 40.0
    original = frame[8]
    frame[8] = Keypoint(index=8, x=original.x + displacement_px, y=original.y, confidence=0.9)

    residuals, _, _ = leave_one_out_residuals(frame)
    by_index = {index: pixels for index, _, pixels, _ in residuals}

    assert by_index[8] == pytest.approx(displacement_px, abs=1e-4)
    # The four exact points are fitted against a set that includes the moved
    # point, so their residuals are non-zero -- proving the moved point does
    # participate in other keypoints' fits and is excluded only from its own.
    assert any(by_index[index] > 1e-3 for index in (0, 5, 10, 15))


def test_each_fit_uses_exactly_the_other_confident_keypoints(monkeypatch):
    # Pins the exclusion at the call boundary: every fit must see one fewer
    # point than the frame holds, and never the held-out index itself.
    frame = exact_frame([0, 5, 10, 15, 8, 9])
    seen: list[list[tuple[float, float]]] = []
    real_fit = measure.fit_homography

    def spy(
        image_points: list[tuple[float, float]],
        template_points: list[tuple[float, float]],
    ) -> object:
        seen.append(list(image_points))
        return real_fit(image_points, template_points)

    monkeypatch.setattr(measure, 'fit_homography', spy)
    leave_one_out_residuals(frame)

    assert len(seen) == 6
    # confident_keypoints() yields index order, not the order they were set.
    for held_out, points in zip([0, 5, 8, 9, 10, 15], seen):
        assert len(points) == 5
        assert to_image(held_out) not in points


# --- the degenerate-fit guard -------------------------------------------

def test_collinear_points_are_rejected_rather_than_scored():
    # Indices 0 to 5 all lie on the left baseline (x = 0) in the template, so
    # their template side is exactly collinear and no homography exists.
    frame = exact_frame([0, 1, 2, 3, 4, 5])

    residuals, degenerate, collinear = leave_one_out_residuals(frame)

    # Six confident points, so six held-out fits. Every fitting set is five
    # baseline points with no general-position quadruple among them, so all
    # six are rejected before the determinant check is reached.
    assert collinear == 6
    assert degenerate == 0
    assert residuals == []


def test_fit_homography_returns_none_for_a_collinear_point_set():
    collinear_image = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    collinear_template = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]

    assert fit_homography(collinear_image, collinear_template) is None


def test_fit_homography_returns_none_for_a_near_collinear_point_set():
    # The case the determinant check exists for, and the one findHomography
    # does NOT catch itself: perturbing collinear points by 1e-9 makes it
    # return a finite matrix whose determinant is ~1e-18. Without the guard
    # that matrix would be used and would produce a plausible-looking but
    # meaningless residual, which is worse than no measurement at all.
    near_image = [(0.0, 0.0), (10.0, 1e-9), (20.0, 0.0), (30.0, 1e-9)]
    near_template = [(0.0, 0.0), (1.0, 0.0), (2.0, 1e-9), (3.0, 0.0)]

    raw, _ = cv2.findHomography(
        np.array(near_image, dtype=np.float64).reshape(-1, 1, 2),
        np.array(near_template, dtype=np.float64).reshape(-1, 1, 2),
        method=0,
    )
    assert raw is not None, 'fixture no longer exercises the determinant guard'
    assert abs(float(np.linalg.det(raw))) < 1e-8

    assert fit_homography(near_image, near_template) is None


def test_a_degenerate_frame_is_counted_separately_from_a_coverage_skip(monkeypatch, tmp_path):
    # Frame 0 is collinear (scored attempt, degenerate); frame 1 has four
    # confident points (never attempted, coverage). The two must not be
    # conflated -- they mean different things about the model.
    collinear = exact_frame([0, 1, 2, 3, 4, 5])
    too_few = exact_frame([0, 5, 10, 15])
    stub_clip(monkeypatch, [collinear, too_few])

    result = measure_clip('clip_1')

    assert result['lost_collinear_frames'] == 1
    assert result['lost_degenerate_frames'] == 0
    assert result['coverage_skipped'] == 1
    assert result['scored_frames'] == 0
    assert result['reprojection_rows'] == []


def test_a_frame_with_a_mix_of_scored_and_rejected_fits_keeps_its_good_residuals(monkeypatch, tmp_path):
    # Indices 0, 1, 2 lie on the left baseline; 6 and 7 are on the centre line.
    # Holding out 0, 1 or 2 leaves a well-conditioned set and is scored; holding
    # out 6 or 7 leaves {0,1,2} plus one point, whose collinear triple makes the
    # homography under-determined, so it is rejected. The frame must count as
    # scored, keep its three good residuals, and record the two failures as
    # FITS rather than as a lost frame -- the double-count this test guards.
    stub_clip(monkeypatch, [exact_frame([0, 1, 2, 6, 7])])

    result = measure_clip('clip_1')

    assert result['scored_frames'] == 1
    assert result['lost_degenerate_frames'] == 0
    assert result['lost_collinear_frames'] == 0
    assert result['coverage_skipped'] == 0
    assert result['collinear_fits'] == 2
    assert len(result['reprojection_rows']) == 3
    # And the surviving residuals are correct, not merely present.
    assert all(float(row[4]) == pytest.approx(0.0, abs=1e-4) for row in result['reprojection_rows'])


def test_the_three_frame_categories_reconcile_against_the_frame_count(monkeypatch, tmp_path):
    # Every frame is skipped, lost or scored, so the three must sum to the
    # frame count; a frame appearing in two categories is exactly the bug.
    stub_clip(monkeypatch, [
        exact_frame([0, 5, 10, 15]),           # coverage-skipped
        exact_frame([0, 1, 2, 3, 4, 5]),       # fully lost
        exact_frame([0, 5, 10, 15, 8, 9]),     # scored
        exact_frame([0, 1, 2, 6, 7]),          # scored, with two collinear rejections
    ])

    result = measure_clip('clip_1')
    accounted = (
        result['coverage_skipped'] + result['lost_degenerate_frames']
        + result['lost_collinear_frames'] + result['scored_frames']
    )

    assert accounted == result['n_frames'] == 4
    assert (result['coverage_skipped'], result['lost_collinear_frames'],
            result['scored_frames']) == (1, 1, 2)
    assert [row[7] for row in measure.coverage_table([result])] == ['yes']


def test_a_four_point_set_containing_a_collinear_triple_is_rejected():
    # Three points on the left baseline plus one interior point. With exactly
    # four points the criterion reduces to 'no three collinear', so this is
    # rejected. findHomography still returns a matrix whose determinant is
    # ~0.046 -- nowhere near singular -- so neither the finite check nor the
    # determinant threshold catches it. Only this criterion does.
    fitting = [0, 1, 2, 8]
    template = [TEMPLATE_POINTS_M[index] for index in fitting]

    assert not has_general_position_quadruple(template)

    raw, _ = cv2.findHomography(
        np.array([to_image(index) for index in fitting], dtype=np.float64).reshape(-1, 1, 2),
        np.array(template, dtype=np.float64).reshape(-1, 1, 2),
        method=0,
    )
    assert raw is not None, 'fixture no longer exercises the triple check'
    assert abs(float(np.linalg.det(raw))) > 1e-8, 'determinant guard would have caught this'


def test_a_well_conditioned_four_point_set_is_not_rejected():
    template = [TEMPLATE_POINTS_M[index] for index in (0, 5, 10, 15)]

    assert has_general_position_quadruple(template)


def test_a_five_point_set_with_a_collinear_triple_is_still_determined():
    # The correction that matters: with five or more points, extra rows cannot
    # reduce the DLT's rank, so a collinear triple elsewhere in the set is
    # harmless as long as SOME four points are in general position.
    # {0, 1, 2, 8, 9} contains the collinear triple (0, 1, 2) yet recovers an
    # exact transform, so rejecting it would discard a valid measurement.
    fitting = [0, 1, 2, 8, 9]
    template = [TEMPLATE_POINTS_M[index] for index in fitting]

    assert has_general_position_quadruple(template)

    matrix = fit_homography([to_image(index) for index in fitting], template)
    assert matrix is not None
    for index in fitting:
        predicted = measure.apply_homography(matrix, to_image(index))
        assert predicted == pytest.approx(TEMPLATE_POINTS_M[index], abs=1e-4)


def test_a_five_point_set_with_no_general_position_quadruple_is_rejected():
    # Five points that are all collinear: no four of them qualify, so there is
    # genuinely no homography to recover.
    template = [TEMPLATE_POINTS_M[index] for index in (0, 1, 2, 3, 4)]

    assert not has_general_position_quadruple(template)


def test_collinear_rejections_are_counted_apart_from_determinant_rejections(monkeypatch, tmp_path):
    # Different causes: a collinear triple is a property of WHICH landmarks were
    # confident, a degenerate determinant a property of the fit itself.
    stub_clip(monkeypatch, [exact_frame([0, 1, 2, 3, 4, 5])])

    result = measure_clip('clip_1')

    assert result['collinear_fits'] == 6
    assert result['degenerate_fits'] == 0


def test_the_index_table_reports_per_clip_and_pooled_with_iqr(monkeypatch, tmp_path):
    # Section 4.2 asks for per clip AND per index with median and IQR: a
    # clip-specific index failure is invisible in a pooled median alone.
    stub_clip(monkeypatch, [exact_frame([0, 5, 8, 9, 10, 15])])
    result = measure_clip('clip_1')

    rows = measure.reprojection_index_table([result])
    scopes = {row[0] for row in rows}

    assert scopes == {'clip_1', 'pooled'}
    assert len(rows) == 2 * 18
    # Seven columns: scope, index, n, median_m, iqr_m, median_px, iqr_px.
    assert all(len(row) == 7 for row in rows)
    scored = [row for row in rows if row[0] == 'clip_1' and row[2] != '0']
    assert scored and all('-' in row[4] for row in scored)


def test_a_baseline_only_frame_is_attributed_to_the_collinear_category(monkeypatch, tmp_path):
    # Every fit here is rejected for an under-determined fitting set, not for a
    # degenerate determinant, so the frame must land in the collinear category
    # with degenerate_fits at zero -- previously it was reported as
    # lost_all_fits_degenerate while degenerate_fits read 0, which contradicted
    # itself.
    stub_clip(monkeypatch, [exact_frame([0, 1, 2, 3, 4, 5])])

    result = measure_clip('clip_1')

    assert result['lost_collinear_frames'] == 1
    assert result['lost_degenerate_frames'] == 0
    assert result['degenerate_fits'] == 0
    assert result['collinear_fits'] == 6


def test_the_four_frame_categories_appear_as_separate_coverage_columns(monkeypatch, tmp_path):
    stub_clip(monkeypatch, [
        exact_frame([0, 5, 10, 15]),        # coverage-skipped
        exact_frame([0, 1, 2, 3, 4, 5]),    # lost to under-determined sets
        exact_frame([0, 5, 8, 9, 10, 15]),  # scored
    ])

    row = measure.coverage_table([measure_clip('clip_1')])[0]

    # clip, frames, skipped, lost_degenerate, lost_collinear, scored, accounted, reconciles, ...
    assert row[2:8] == ['1', '0', '1', '1', '3', 'yes']


# --- DLT conditioning ----------------------------------------------------

def test_conditioning_is_recorded_for_every_residual(monkeypatch, tmp_path):
    stub_clip(monkeypatch, [exact_frame([0, 5, 8, 9, 10, 15])])

    result = measure_clip('clip_1')

    assert result['reprojection_rows']
    for row in result['reprojection_rows']:
        assert len(row) == 7
        conditioning = float(row[6])
        assert conditioning == conditioning, 'conditioning must not be NaN for a real fit'
        assert conditioning > 0.0


def test_a_well_spread_fit_is_better_conditioned_than_a_nearly_collinear_one():
    # The measure must actually discriminate, otherwise recording it answers
    # nothing. Both sets are in general position on the TEMPLATE side; the
    # second is squashed toward a line in the IMAGE, which is exactly the case
    # the template-side check cannot see.
    spread = [0, 5, 10, 15]
    template = [TEMPLATE_POINTS_M[index] for index in spread]

    well_spread = dlt_conditioning([to_image(index) for index in spread], template)
    squashed = dlt_conditioning(
        [(to_image(index)[0], to_image(index)[1] * 0.001) for index in spread], template,
    )

    assert well_spread > squashed


def test_conditioning_is_invariant_to_a_uniform_rescaling_of_image_coordinates():
    # The same geometry at a different resolution must score the same. Without
    # Hartley normalisation the raw DLT mixes ~10^3 pixel with ~10^1 metre
    # magnitudes, and the ratio is depressed by that disparity rather than by
    # geometry: measured 2.8e-5 at one scale against 7.6e-7 at 1000x, a 37-fold
    # swing from units alone. A threshold chosen on that would read resolution.
    indices = [0, 5, 10, 15]
    template = [TEMPLATE_POINTS_M[index] for index in indices]

    baseline = dlt_conditioning([to_image(index) for index in indices], template)
    rescaled = dlt_conditioning(
        [(x * 1000.0, y * 1000.0) for x, y in (to_image(index) for index in indices)], template,
    )

    assert rescaled == pytest.approx(baseline, rel=1e-9)


def test_conditioning_is_invariant_to_translating_the_image_coordinates():
    # Centring is the other half of the normalisation: an offset frame origin
    # must not move the measure either.
    indices = [0, 5, 10, 15]
    template = [TEMPLATE_POINTS_M[index] for index in indices]

    baseline = dlt_conditioning([to_image(index) for index in indices], template)
    shifted = dlt_conditioning(
        [(x + 5000.0, y - 2000.0) for x, y in (to_image(index) for index in indices)], template,
    )

    assert shifted == pytest.approx(baseline, rel=1e-9)


def test_hartley_normalise_centres_and_scales_to_root_two():
    points = [(100.0, 50.0), (300.0, 50.0), (300.0, 250.0), (100.0, 250.0)]

    normalised = hartley_normalise(points)

    centroid_x = sum(x for x, _ in normalised) / len(normalised)
    centroid_y = sum(y for _, y in normalised) / len(normalised)
    mean_distance = sum(np.hypot(x, y) for x, y in normalised) / len(normalised)

    assert centroid_x == pytest.approx(0.0, abs=1e-12)
    assert centroid_y == pytest.approx(0.0, abs=1e-12)
    assert mean_distance == pytest.approx(np.sqrt(2.0))


def test_hartley_normalise_survives_coincident_points():
    # No scale to recover; dividing by a zero mean distance would give NaN.
    coincident = [(7.0, 7.0)] * 4

    normalised = hartley_normalise(coincident)

    assert all(point == (0.0, 0.0) for point in normalised)


def test_conditioning_is_recorded_but_never_rejects(monkeypatch, tmp_path):
    # An image-side near-collinear frame must still be scored: the measure is
    # evidence for choosing a threshold later, not a threshold itself.
    frame = exact_frame([0, 5, 8, 9, 10, 15])
    squashed = [
        Keypoint(index=k.index, x=k.x, y=k.y * 0.001 if k.confidence else k.y, confidence=k.confidence)
        for k in frame
    ]
    stub_clip(monkeypatch, [squashed])

    result = measure_clip('clip_1')

    assert result['scored_frames'] == 1
    assert result['reprojection_rows']


def test_the_conditioning_table_reports_per_clip_and_pooled(monkeypatch, tmp_path):
    stub_clip(monkeypatch, [exact_frame([0, 5, 8, 9, 10, 15])])
    result = measure_clip('clip_1')

    rows = measure.conditioning_table([result])

    assert [row[0] for row in rows] == ['clip_1', 'pooled']
    # scope, bucket, n, min, q1, median, q3, max
    assert all(len(row) == 8 for row in rows)


def test_the_conditioning_table_is_bucketed_by_confident_count(monkeypatch, tmp_path):
    # The DLT matrix shape changes with the fitting-point count, so sigma[7]
    # is the smallest singular value at 4 fitting points and the second
    # smallest at 5 or more. Pooling those would average different
    # quantities, so each bucket must be reported on its own row -- and the
    # buckets must match the residual table's, so the two can be read
    # against each other.
    stub_clip(monkeypatch, [
        exact_frame([0, 5, 8, 9, 10]),          # 5 confident -> bucket '5'
        exact_frame([0, 5, 8, 9, 10, 15]),      # 6 confident -> bucket '6'
    ])
    result = measure_clip('clip_1')

    rows = measure.conditioning_table([result])
    per_clip = [row for row in rows if row[0] == 'clip_1']

    assert [row[1] for row in per_clip] == ['5', '6']
    # Bucket labels match those the residual table uses.
    residual_buckets = {
        row[1] for row in measure.reprojection_bucket_table([result]) if row[0] == 'clip_1'
    }
    assert {row[1] for row in per_clip} == residual_buckets


def test_conditioning_values_are_not_pooled_across_buckets(monkeypatch, tmp_path):
    # The counts per bucket must sum to the total, with none double-counted
    # into a single pooled-across-buckets row.
    stub_clip(monkeypatch, [
        exact_frame([0, 5, 8, 9, 10]),
        exact_frame([0, 5, 8, 9, 10, 15]),
    ])
    result = measure_clip('clip_1')

    per_clip = [row for row in measure.conditioning_table([result]) if row[0] == 'clip_1']

    assert sum(int(row[2]) for row in per_clip) == len(result['reprojection_rows'])

# --- coverage and bucketing ---------------------------------------------

def test_a_four_keypoint_frame_is_coverage_skipped_not_scored(monkeypatch, tmp_path):
    stub_clip(monkeypatch, [exact_frame([0, 5, 10, 15])])

    result = measure_clip('clip_1')

    assert result['coverage_skipped'] == 1
    assert result['lost_degenerate_frames'] == 0
    assert result['lost_collinear_frames'] == 0
    assert result['degenerate_fits'] == 0
    assert result['reprojection_rows'] == []
    # Still counted for sufficiency: it reaches four, just not five.
    assert result['sufficiency_rows'][0][3] == 'True'
    assert result['sufficiency_rows'][0][4] == 'False'


@pytest.mark.parametrize('count,expected', [(5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9+'), (14, '9+')])


def test_bucket_for_assigns_the_matching_bucket(count, expected):
    assert bucket_for(count) == expected


def test_a_scored_frame_is_recorded_under_its_own_confident_count(monkeypatch, tmp_path):
    stub_clip(monkeypatch, [exact_frame([0, 5, 10, 15, 8, 9])])

    result = measure_clip('clip_1')

    assert {row[3] for row in result['reprojection_rows']} == {'6'}
    assert all(bucket_for(int(row[3])) == '6' for row in result['reprojection_rows'])


# --- sufficiency runs ----------------------------------------------------

def test_below_threshold_runs_reports_contiguous_windows():
    # Frames 3, 4, 5 and 9 are below four: two runs, of lengths 3 and 1.
    counts = [8, 8, 8, 2, 1, 0, 8, 8, 8, 3, 8]

    assert below_threshold_runs(counts) == [3, 1]


def test_a_run_reaching_the_end_of_the_clip_is_closed():
    assert below_threshold_runs([8, 8, 1, 1]) == [2]


def test_no_sub_threshold_frames_yields_no_runs():
    assert below_threshold_runs([8, 8, 8]) == []


def test_an_entirely_sub_threshold_clip_is_one_run():
    assert below_threshold_runs([0, 0, 0]) == [3]


# --- thresholding and stability -----------------------------------------

def test_confident_keypoints_uses_the_per_keypoint_threshold():
    frame = [
        Keypoint(index=0, x=1.0, y=1.0, confidence=KEYPOINT_CONFIDENCE_THRESHOLD),
        Keypoint(index=1, x=1.0, y=1.0, confidence=KEYPOINT_CONFIDENCE_THRESHOLD - 0.01),
    ]

    assert [keypoint.index for keypoint in confident_keypoints(frame)] == [0]


def test_stability_pairs_only_frames_where_the_index_is_confident_in_both(monkeypatch, tmp_path):
    first = exact_frame([0, 5, 10])
    second = exact_frame([0, 5])          # index 10 drops out
    moved = [
        Keypoint(index=k.index, x=k.x + 3.0, y=k.y + 4.0, confidence=k.confidence) for k in second
    ]
    stub_clip(monkeypatch, [first, moved])

    result = measure_clip('clip_1')
    by_index = {int(row[2]): float(row[3]) for row in result['stability_rows']}

    assert set(by_index) == {0, 5}        # index 10 contributes no pair
    assert by_index[0] == pytest.approx(5.0)   # 3-4-5 triangle


# --- an all-failed run must not truncate previous results ----------------

def test_a_run_where_every_clip_fails_leaves_existing_outputs_untouched(monkeypatch, tmp_path, capsys):
    # These outputs cost a full inference pass to regenerate, and the failure
    # is silent: the run still reports skipped clips and leaves four files that
    # appear to exist. Writing headers over them is pure loss.
    output_dir = tmp_path / 'keypoint_evaluation'
    output_dir.mkdir()
    existing = {
        'sufficiency.csv': 'clip,frame,confident_count\nclip_1,0,8\n',
        'reprojection.csv': 'clip,frame,keypoint_index\nclip_1,0,3\n',
        'stability.csv': 'clip,frame,keypoint_index\nclip_1,1,3\n',
        'summary.txt': 'previous run\n',
    }
    for name, content in existing.items():
        (output_dir / name).write_text(content, encoding='utf-8')

    redirect_outputs(monkeypatch, output_dir)
    monkeypatch.setattr(
        measure, 'measure_clip',
        lambda clip: (_ for _ in ()).throw(OSError(f'no such video: {clip}')),
    )

    measure.main()

    for name, content in existing.items():
        assert (output_dir / name).read_text(encoding='utf-8') == content, f'{name} was overwritten'
    printed = capsys.readouterr().out
    assert 'No clip was scored' in printed
    assert 'left untouched' in printed


def test_a_partially_failed_run_still_writes_the_clips_that_succeeded(monkeypatch, tmp_path):
    # The partial-results intent holds as soon as one clip scores: the
    # all-failed skip must not suppress a genuine partial result.
    output_dir = tmp_path / 'keypoint_evaluation'
    redirect_outputs(monkeypatch, output_dir)
    stub_clip(monkeypatch, [exact_frame([0, 5, 10, 15, 8, 9])])
    real_measure_clip = measure.measure_clip

    def one_good_two_bad(clip: str) -> dict:
        if clip == 'clip_1':
            return real_measure_clip(clip)
        raise OSError(f'no such video: {clip}')

    monkeypatch.setattr(measure, 'measure_clip', one_good_two_bad)

    measure.main()

    assert (output_dir / 'sufficiency.csv').exists()
    assert (output_dir / 'summary.txt').exists()
    assert 'clip_1' in (output_dir / 'reprojection.csv').read_text(encoding='utf-8')


# --- helpers -------------------------------------------------------------

def stub_clip(monkeypatch, keypoints_per_frame: list[list[Keypoint]]) -> None:
    """Replace frame loading and detection so measure_clip runs offline with synthetic keypoints."""
    # The detection run itself needs weights, clips and the GPU stack, so it
    # is exercised on JupyterHub rather than here; everything downstream of it
    # is pure arithmetic and is covered with synthetic data.
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in keypoints_per_frame]
    monkeypatch.setattr(measure, 'load_video', lambda path: list(enumerate(frames)))

    class _StubDetector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_detection(self, **kwargs: object) -> list[list[Keypoint]]:
            return keypoints_per_frame

    monkeypatch.setattr(measure, 'CourtKeypoints', _StubDetector)


def redirect_outputs(monkeypatch, output_dir: Path) -> None:
    """Point every output path at a temporary directory so a test run cannot touch the real artefacts."""
    monkeypatch.setattr(measure, 'OUTPUT_DIR', output_dir)
    monkeypatch.setattr(measure, 'SUFFICIENCY_CSV', output_dir / 'sufficiency.csv')
    monkeypatch.setattr(measure, 'REPROJECTION_CSV', output_dir / 'reprojection.csv')
    monkeypatch.setattr(measure, 'STABILITY_CSV', output_dir / 'stability.csv')
    monkeypatch.setattr(measure, 'SUMMARY_TXT', output_dir / 'summary.txt')
