"""Unit tests for Stage 8's Homography (basketball/homography/transform.py)."""

from __future__ import annotations

import numpy as np
import pytest

from basketball.homography.transform import (
    DegenerateCorrespondencesError,
    Homography,
    HomographyError,
    InsufficientCorrespondencesError,
    MalformedCorrespondencesError,
    has_general_position_quadruple,
    is_collinear,
)
from basketball.keypoints.court_template import TEMPLATE_POINTS_M
from scripts.measure_court_keypoints import (
    has_general_position_quadruple as stage_7_has_general_position_quadruple,
)

# An exact similarity transform from metres to pixels: invertible and simple
# enough that the expected court position of any image point is known by hand.
SCALE = 20.0
OFFSET = (100.0, 50.0)

# Index sets taken from the measured framing behaviour: a frame showing the
# left half of the court yields 0-5, 8, 9; a frame showing the right half
# yields 10-17. Six confident keypoints is the measured mode.
LEFT_HALF_SIX = [0, 1, 2, 3, 8, 9]
RIGHT_HALF_SIX = [10, 11, 12, 13, 16, 17]
VIABLE_FOUR = [0, 5, 8, 9]
ONE_BASELINE_FOUR = [0, 1, 2, 3]


def to_image(index: int) -> tuple[float, float]:
    """Map a keypoint's template position into image space under the exact synthetic transform."""
    x_m, y_m = TEMPLATE_POINTS_M[index]
    return (x_m * SCALE + OFFSET[0], y_m * SCALE + OFFSET[1])


def correspondences(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return the paired image and template arrays for a set of keypoint indices."""
    image = np.array([to_image(index) for index in indices], dtype=np.float64)
    template = np.array([TEMPLATE_POINTS_M[index] for index in indices], dtype=np.float64)
    return image, template


def fitted(indices: list[int]) -> Homography:
    """Return a Homography fitted to the given indices under the synthetic transform."""
    homography = Homography()
    homography.transform_points(*correspondences(indices))
    return homography


# --- the round trip ------------------------------------------------------

@pytest.mark.parametrize('indices', [LEFT_HALF_SIX, RIGHT_HALF_SIX, VIABLE_FOUR])
def test_a_known_transform_round_trips_to_the_template_positions(indices):
    # findHomography solves by SVD, so an exact transform lands within float
    # noise rather than at exactly zero: measured ~6e-7 m on this fixture,
    # which is 0.6 micrometres on a 28.65 m court.
    homography = fitted(indices)

    for index in indices:
        recovered = homography.apply_homography(to_image(index))
        assert recovered == pytest.approx(TEMPLATE_POINTS_M[index], abs=1e-4)


def test_both_court_halves_fit_independently():
    # Both framings occur within a single clip, so a fixture that only ever
    # supplies one index set would not exercise the other.
    left = fitted(LEFT_HALF_SIX)
    right = fitted(RIGHT_HALF_SIX)

    assert left.apply_homography(to_image(0)) == pytest.approx(TEMPLATE_POINTS_M[0], abs=1e-4)
    assert right.apply_homography(to_image(10)) == pytest.approx(TEMPLATE_POINTS_M[10], abs=1e-4)


def test_a_point_between_keypoints_maps_consistently():
    # The transform must be right everywhere on the plane, not only at the
    # points it was fitted to.
    homography = fitted(LEFT_HALF_SIX)
    midpoint_m = (2.0, 3.0)
    midpoint_px = (midpoint_m[0] * SCALE + OFFSET[0], midpoint_m[1] * SCALE + OFFSET[1])

    assert homography.apply_homography(midpoint_px) == pytest.approx(midpoint_m, abs=1e-4)


def test_the_matrix_is_stored_and_exposed():
    homography = Homography()
    assert homography.matrix is None

    returned = homography.transform_points(*correspondences(LEFT_HALF_SIX))

    assert homography.matrix is not None
    assert np.allclose(returned, homography.matrix)


# --- guards --------------------------------------------------------------

@pytest.mark.parametrize('count', [0, 1, 2, 3])
def test_fewer_than_four_correspondences_raises_insufficient(count):
    # Three confident keypoints is a real measured state: clip_3's minimum is 1.
    indices = [0, 5, 8, 9][:count]
    image, template = correspondences(indices)

    with pytest.raises(InsufficientCorrespondencesError, match='at least 4'):
        Homography().transform_points(image, template)


def test_mismatched_lengths_raise_malformed_not_insufficient():
    # A different cause from too-few points, and the caller counts them apart.
    image, template = correspondences(LEFT_HALF_SIX)

    with pytest.raises(MalformedCorrespondencesError, match='must be paired'):
        Homography().transform_points(image, template[:-1])


def test_non_two_dimensional_points_raise_malformed():
    image = np.array([[1.0, 2.0, 3.0]] * 4, dtype=np.float64)
    template = np.array([TEMPLATE_POINTS_M[i] for i in VIABLE_FOUR], dtype=np.float64)

    with pytest.raises(MalformedCorrespondencesError, match='two columns'):
        Homography().transform_points(image, template)


def test_a_one_dimensional_array_raises_malformed():
    with pytest.raises(MalformedCorrespondencesError, match='2D arrays'):
        Homography().transform_points(np.array([1.0, 2.0]), np.array([3.0, 4.0]))


def test_non_finite_coordinates_raise_malformed():
    image, template = correspondences(VIABLE_FOUR)
    image[0][0] = np.nan

    with pytest.raises(MalformedCorrespondencesError, match='non-finite'):
        Homography().transform_points(image, template)


def test_every_error_is_a_homography_error():
    # A caller that does not care which cause applies can catch one type.
    for error in (
        InsufficientCorrespondencesError,
        MalformedCorrespondencesError,
        DegenerateCorrespondencesError,
    ):
        assert issubclass(error, HomographyError)


def test_applying_before_fitting_raises():
    with pytest.raises(HomographyError, match='holds no valid transform'):
        Homography().apply_homography((10.0, 20.0))


def test_a_failed_refit_discards_the_previous_matrix():
    # The guarantee this test exists for: no-fallback must be a property of the
    # class, not of CourtMapper's habit of building a fresh Homography per
    # frame. Without it, a caller reusing one instance gets stale geometry from
    # a failed re-fit, silently and indistinguishable from a real result.
    homography = fitted(LEFT_HALF_SIX)
    assert homography.apply_homography(to_image(8)) == pytest.approx(
        TEMPLATE_POINTS_M[8], abs=1e-4,
    )

    with pytest.raises(InsufficientCorrespondencesError):
        homography.transform_points(*correspondences([0, 5]))

    assert homography.matrix is None
    with pytest.raises(HomographyError, match='holds no valid transform'):
        homography.apply_homography(to_image(8))


@pytest.mark.parametrize('bad_indices,error', [
    ([0, 5], InsufficientCorrespondencesError),
    (ONE_BASELINE_FOUR, DegenerateCorrespondencesError),
])
def test_every_failure_path_leaves_the_object_unusable(bad_indices, error):
    # Each named error must clear the matrix, not only the first one checked.
    homography = fitted(LEFT_HALF_SIX)

    with pytest.raises(error):
        homography.transform_points(*correspondences(bad_indices))

    assert homography.matrix is None
    with pytest.raises(HomographyError):
        homography.apply_homography(to_image(8))


def test_a_malformed_refit_also_clears_the_matrix():
    homography = fitted(LEFT_HALF_SIX)
    image, template = correspondences(LEFT_HALF_SIX)

    with pytest.raises(MalformedCorrespondencesError):
        homography.transform_points(image, template[:-1])

    assert homography.matrix is None


def test_a_successful_refit_replaces_the_previous_matrix():
    # The reset must not break the ordinary case: a valid re-fit still works,
    # and yields the NEW transform rather than the old one.
    homography = fitted(LEFT_HALF_SIX)
    first = homography.matrix.copy()

    homography.transform_points(*correspondences(RIGHT_HALF_SIX))

    assert homography.matrix is not None
    assert homography.apply_homography(to_image(10)) == pytest.approx(
        TEMPLATE_POINTS_M[10], abs=1e-4,
    )
    assert not np.allclose(first, homography.matrix)


# --- the general-position rule -------------------------------------------

def test_four_points_from_one_baseline_are_rejected_as_degenerate():
    # TEMPLATE_POINTS_M puts indices 0-5 on the line x = 0, so four of them
    # are collinear. This is the case the determinant guard does NOT catch.
    image, template = correspondences(ONE_BASELINE_FOUR)

    assert not has_general_position_quadruple([tuple(point) for point in template])

    with pytest.raises(DegenerateCorrespondencesError, match='general position'):
        Homography().transform_points(image, template)


def test_five_points_with_a_collinear_triple_are_accepted():
    # The rule that took five review rounds to get right in Stage 7: with five
    # or more points the fit is determined as long as SOME four are in general
    # position, so a collinear triple elsewhere is harmless. Rejecting these
    # would discard valid frames, since partially-collinear sets are the norm.
    indices = [0, 1, 2, 8, 9]
    image, template = correspondences(indices)

    assert is_collinear(TEMPLATE_POINTS_M[0], TEMPLATE_POINTS_M[1], TEMPLATE_POINTS_M[2])
    assert has_general_position_quadruple([tuple(point) for point in template])

    homography = Homography()
    homography.transform_points(image, template)
    for index in indices:
        assert homography.apply_homography(to_image(index)) == pytest.approx(
            TEMPLATE_POINTS_M[index], abs=1e-4,
        )


def test_six_points_all_on_one_baseline_are_still_rejected():
    # More points do not rescue a set with no general-position quadruple at all.
    template = [TEMPLATE_POINTS_M[i] for i in (0, 1, 2, 3, 4, 5)]

    assert not has_general_position_quadruple(template)


def test_the_degenerate_check_agrees_with_the_stage_7_implementation():
    # Stage 7's measurement script carries its own copy of this criterion.
    # They must not drift: this one lives in basketball/ because scripts/ is
    # not importable from the package, so agreement is pinned instead.
    for indices in ([0, 1, 2, 3], [0, 1, 2, 8, 9], [0, 5, 10, 15], [0, 1, 2, 3, 4, 5], LEFT_HALF_SIX):
        template = [TEMPLATE_POINTS_M[i] for i in indices]
        assert (
            has_general_position_quadruple(template)
            == stage_7_has_general_position_quadruple(template)
        ), f'the two criteria disagree on {indices}'


def test_a_singular_matrix_is_rejected_by_the_determinant_guard():
    # All four image points coincide, so no invertible transform exists.
    image = np.array([[100.0, 50.0]] * 4, dtype=np.float64)
    template = np.array([TEMPLATE_POINTS_M[i] for i in VIABLE_FOUR], dtype=np.float64)

    with pytest.raises(DegenerateCorrespondencesError):
        Homography().transform_points(image, template)


# --- the horizon ---------------------------------------------------------

def test_a_point_behind_the_camera_plane_is_rejected():
    # The far side of the horizon. w is comfortably non-zero there, so the
    # magnitude guard passes it, and it divides through to a finite coordinate
    # that can land inside the court-plus-margin window and be recorded as a
    # real mapped position.
    homography = Homography()
    # w = 1 - 0.004 * y, so the horizon sits at y = 250 and y > 250 is behind it.
    homography._matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.004, 1.0]], dtype=np.float64,
    )
    homography._reference_sign = 1.0

    behind = (5.0, 400.0)
    scale = float(homography._matrix[2] @ np.array([behind[0], behind[1], 1.0]))
    assert abs(scale) > 1e-6, 'the fixture must clear the magnitude guard to be meaningful'

    with pytest.raises(DegenerateCorrespondencesError, match='behind the camera plane'):
        homography.apply_homography(behind)


def test_a_point_in_front_of_the_camera_plane_still_maps():
    homography = Homography()
    homography._matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.004, 1.0]], dtype=np.float64,
    )
    homography._reference_sign = 1.0

    assert homography.apply_homography((5.0, 100.0)) == pytest.approx(
        (5.0 / 0.6, 100.0 / 0.6), rel=1e-9,
    )


def test_a_negated_matrix_maps_its_own_points_rather_than_rejecting_them():
    # findHomography returns a matrix defined only up to scale, so a globally
    # negated matrix is an equally valid fit and makes every legitimate point's
    # w negative. A fixed w > 0 rule would reject all of them.
    #
    # Measured across 26,586 valid general-position fits under a projective
    # camera, OpenCV never actually returned a negated matrix, so this state is
    # constructed rather than found. The guard is written against sign
    # consistency anyway, because nothing documents that normalisation as a
    # guarantee; this test pins that the code does not depend on it.
    image, template = correspondences(LEFT_HALF_SIX)
    fitted_normally = Homography()
    fitted_normally.transform_points(image, template)

    negated = Homography()
    negated._matrix = -fitted_normally.matrix
    scales = image @ negated._matrix[2, :2] + negated._matrix[2, 2]
    assert all(scale < 0 for scale in scales), 'the fixture must actually be negated'
    negated._reference_sign = -1.0

    for index in LEFT_HALF_SIX:
        assert negated.apply_homography(to_image(index)) == pytest.approx(
            TEMPLATE_POINTS_M[index], abs=1e-4,
        )


def test_the_reference_sign_is_derived_from_the_fit_not_assumed_positive():
    # Pins that _reference_sign is computed rather than hardcoded. Constructed
    # by negating a real fit, since no natural negated fit exists on this data.
    image, template = correspondences(LEFT_HALF_SIX)
    homography = Homography()
    homography.transform_points(image, template)

    scales = image @ homography.matrix[2, :2] + homography.matrix[2, 2]
    expected = 1.0 if float(np.median(scales)) > 0.0 else -1.0

    assert homography._reference_sign == expected
    # And the derived sign agrees with every fitted point, so no fitted
    # correspondence is ever rejected as being behind the camera.
    assert all(scale * homography._reference_sign > 0 for scale in scales)


def test_the_reference_sign_is_taken_from_the_fitted_correspondences():
    image, template = correspondences(LEFT_HALF_SIX)
    homography = Homography()
    homography.transform_points(image, template)

    scales = image @ homography.matrix[2, :2] + homography.matrix[2, 2]

    assert homography._reference_sign == pytest.approx(np.sign(np.median(scales)))
    assert all(scale * homography._reference_sign > 0 for scale in scales)


def test_a_point_on_the_horizon_raises_rather_than_mapping_to_the_origin():
    # cv2.perspectiveTransform substitutes 0 for a division by a zero
    # homogeneous scale rather than producing a non-finite value, measured
    # directly. The origin is a real court position and would pass bounds
    # filtering, so the scale is checked before dividing.
    homography = Homography()
    homography._matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.001, 0.0]], dtype=np.float64,
    )

    with pytest.raises(DegenerateCorrespondencesError, match='horizon'):
        homography.apply_homography((5.0, 0.0))


def test_a_point_off_the_horizon_still_maps_under_the_same_matrix():
    homography = Homography()
    homography._matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.001, 0.0]], dtype=np.float64,
    )

    assert homography.apply_homography((5.0, 1.0)) == pytest.approx((5000.0, 1000.0))
