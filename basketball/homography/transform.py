"""
transform.py computes the image-to-court homography from CourtKeypoints
correspondences and applies it to individual points, raising a distinct named
error for each way a fit can be unusable rather than returning a wrong matrix.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

# Minimum triangle area, in square metres, for three template points to count
# as non-collinear. Applied on the TEMPLATE side, where positions are exact and
# collinearity is a geometric fact rather than a threshold question; the image
# side carries projection noise and would turn this into a judgement call.
# 0.01 m^2 is far below any genuine court triangle and far above float error.
MIN_TRIANGLE_AREA_M2 = 0.01

# Below this absolute determinant the recovered matrix is treated as singular.
MIN_DETERMINANT = 1e-8

# Below this absolute homogeneous scale a point is treated as lying on the
# horizon rather than being divided through, which would otherwise produce an
# arbitrarily large court position from a rounding-level denominator.
MIN_HOMOGENEOUS_SCALE = 1e-12

MIN_CORRESPONDENCES = 4


class HomographyError(Exception):
    """Base class for every reason a homography could not be computed."""


class InsufficientCorrespondencesError(HomographyError):
    """Raised when fewer than four point correspondences were supplied."""


class MalformedCorrespondencesError(HomographyError):
    """Raised when the two point sets differ in length or are not 2D coordinates."""


class DegenerateCorrespondencesError(HomographyError):
    """Raised when the supplied correspondences cannot determine a homography."""


def is_collinear(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    """Return whether three template positions are collinear, by triangle area."""
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0 < MIN_TRIANGLE_AREA_M2


def has_general_position_quadruple(template_points: list[tuple[float, float]]) -> bool:
    """Return whether some four of the given template positions have no three collinear, which is what determines the homography."""
    # A plane-to-plane homography needs four correspondences in general
    # position. For a set of five or more, extra rows cannot REDUCE the DLT's
    # rank, so the fit is determined as long as SOME four points qualify -- a
    # collinear triple elsewhere in the set is harmless. Rejecting those would
    # discard valid frames, since TEMPLATE_POINTS_M puts indices 0-5 on one
    # baseline and 10-15 on the other, so partially-collinear sets are the
    # norm on this data rather than an exotic case.
    #
    # For exactly four points this reduces to 'no three of them collinear'.
    # That case is the one that bites: four points from a single baseline
    # yield a matrix whose determinant is nowhere near singular, so neither
    # the finite check nor the determinant threshold catches it, and the
    # result is a confidently wrong transform.
    for quad in itertools.combinations(template_points, 4):
        if not any(is_collinear(*triple) for triple in itertools.combinations(quad, 3)):
            return True
    return False


class Homography:
    """Computes the perspective matrix mapping broadcast image coordinates onto the metric court plane, and applies it to single points."""

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        # The sign of the homogeneous scale at the correspondences the matrix
        # was fitted to. See apply_homography: findHomography returns a matrix
        # defined only up to scale, so the overall sign carries no guarantee
        # and the far-side test is written against this rather than w > 0.
        self._reference_sign: float = 1.0

    @property
    def matrix(self) -> np.ndarray | None:
        """The most recently computed matrix, or None if transform_points has not succeeded yet."""
        return self._matrix

    def _validate(self, image_points: np.ndarray, template_points: np.ndarray) -> None:
        """Raise the specific error describing why the supplied correspondences cannot be used."""
        # Separate exception types rather than one: a caller needs to tell
        # 'this frame had too few keypoints' from 'the keypoints it had are
        # unusable', because the two say different things about the frame and
        # CourtMapper reports them as different counts.
        # Checked before the shape rules: an empty list arrives as shape (0,)
        # rather than (0, 2), and reporting 'no keypoints at all' as malformed
        # input would miscount the most common unmapped frame as a bad-input
        # bug. A frame with zero confident keypoints is measured and real:
        # clip_3's minimum is one.
        if len(image_points) == 0 and len(template_points) == 0:
            raise InsufficientCorrespondencesError(
                f'A homography needs at least {MIN_CORRESPONDENCES} correspondences, got 0.'
            )

        if image_points.ndim != 2 or template_points.ndim != 2:
            raise MalformedCorrespondencesError(
                f'Correspondences must be 2D arrays of (x, y) points, got shapes '
                f'{image_points.shape} and {template_points.shape}.'
            )
        if image_points.shape[1] != 2 or template_points.shape[1] != 2:
            raise MalformedCorrespondencesError(
                f'Correspondences must have two columns (x, y), got '
                f'{image_points.shape[1]} and {template_points.shape[1]}.'
            )
        if len(image_points) != len(template_points):
            raise MalformedCorrespondencesError(
                f'Got {len(image_points)} image points against {len(template_points)} '
                f'template points — correspondences must be paired.'
            )
        if len(image_points) < MIN_CORRESPONDENCES:
            raise InsufficientCorrespondencesError(
                f'A homography needs at least {MIN_CORRESPONDENCES} correspondences, '
                f'got {len(image_points)}.'
            )
        if not np.all(np.isfinite(image_points)) or not np.all(np.isfinite(template_points)):
            raise MalformedCorrespondencesError(
                'Correspondences contain non-finite coordinates.'
            )

        template_list = [(float(x), float(y)) for x, y in template_points]
        if not has_general_position_quadruple(template_list):
            raise DegenerateCorrespondencesError(
                f'No four of the {len(template_list)} template points are in general position '
                f'(every quadruple contains three collinear points), so they cannot determine '
                f'a homography.'
            )

    def transform_points(
        self,
        image_points: np.ndarray,
        template_points: np.ndarray,
    ) -> np.ndarray:
        """Estimate and store the (3, 3) image-to-court homography from matched point pairs, raising a named error when it cannot be determined."""
        image_points = np.asarray(image_points, dtype=np.float64)
        template_points = np.asarray(template_points, dtype=np.float64)

        # Discarded BEFORE validation, so every failure path below leaves this
        # object unusable rather than holding the previous fit. Otherwise a
        # caller reusing one instance across frames gets stale geometry from a
        # failed re-fit instead of an error, silently and indistinguishably
        # from a real result. No-fallback is this stage's central decision, so
        # it must be a property of the class rather than of CourtMapper's
        # habit of constructing a fresh Homography per frame.
        self._matrix = None
        self._reference_sign = 1.0

        self._validate(image_points, template_points)

        # method=0 (the plain least-squares DLT) rather than RANSAC: the
        # correspondences are already confidence-gated by the caller and there
        # are typically six or seven of them, too few for RANSAC's sampling to
        # help and enough for one rejected outlier to drop the set below four.
        matrix, _ = cv2.findHomography(
            image_points.reshape(-1, 1, 2), template_points.reshape(-1, 1, 2), method=0,
        )

        if matrix is None:
            raise DegenerateCorrespondencesError(
                'cv2.findHomography could not fit a matrix to the supplied correspondences.'
            )
        if not np.all(np.isfinite(matrix)):
            raise DegenerateCorrespondencesError(
                'The fitted homography contains non-finite values.'
            )
        if abs(float(np.linalg.det(matrix))) < MIN_DETERMINANT:
            raise DegenerateCorrespondencesError(
                f'The fitted homography is singular (|det| < {MIN_DETERMINANT}), so it '
                f'cannot be inverted or trusted.'
            )

        # Recorded from the fitting points themselves rather than assumed
        # positive. The median is taken over all of them so one point sitting
        # near the horizon cannot set the reference for the rest.
        scales = image_points @ matrix[2, :2] + matrix[2, 2]
        reference = float(np.median(scales))
        if reference == 0.0:
            raise DegenerateCorrespondencesError(
                'The fitted homography puts its own correspondences on the horizon, so no '
                'consistent camera side can be established.'
            )

        self._matrix = matrix
        self._reference_sign = 1.0 if reference > 0.0 else -1.0
        return matrix

    def apply_homography(self, point: tuple[float, float]) -> tuple[float, float]:
        """Map a single image point onto the court plane using the stored matrix, raising when no matrix has been computed or the point maps to infinity."""
        if self._matrix is None:
            raise HomographyError(
                'This Homography holds no valid transform — either transform_points() has '
                'not been called, or its most recent call failed and the previous matrix '
                'was discarded rather than reused.'
            )

        # The homogeneous scale is computed here rather than left to
        # perspectiveTransform, because OpenCV substitutes 0 for a division by
        # a zero w instead of producing a non-finite value -- measured
        # directly. A finiteness check on its output is therefore dead code,
        # and a point on the horizon would silently arrive at the court origin,
        # which is a real position on the court and would pass bounds
        # filtering. Checking w first is the only place this is catchable.
        vector = self._matrix @ np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
        w = float(vector[2])
        if not np.isfinite(w) or abs(w) < MIN_HOMOGENEOUS_SCALE:
            raise DegenerateCorrespondencesError(
                f'Image point {point} lies on or near this homography\'s horizon, so it '
                f'maps to the line at infinity rather than to a court position.'
            )

        # Magnitude alone is NOT the complete check, though abs(w) > eps looks
        # like it is. The horizon divides the image plane into two sides, and w
        # changes sign across it: a point on the far side is behind the camera
        # plane and has no physical position on the court, yet it still divides
        # through to a perfectly finite coordinate that can land inside the
        # court-plus-margin window and be recorded as a real mapped position. A
        # false foot point in the stands, or a subtly wrong matrix, produces
        # exactly that.
        #
        # The test is sign CONSISTENCY with the fitted correspondences rather
        # than a fixed w > 0. findHomography returns a matrix defined only up
        # to scale, so a globally negated matrix is an equally valid fit and
        # would make every legitimate point's w negative, which w > 0 would
        # reject wholesale. Measured across 26,586 valid general-position fits
        # under a projective camera, OpenCV in fact never returned a negated
        # matrix here, so the two rules agree on this data; consistency is
        # used anyway because it depends on no OpenCV normalisation guarantee
        # that is documented anywhere, and costs one multiply.
        if w * self._reference_sign <= 0.0:
            raise DegenerateCorrespondencesError(
                f'Image point {point} lies on the far side of this homography\'s horizon '
                f'(behind the camera plane), so it has no position on the court.'
            )

        x, y = float(vector[0]) / w, float(vector[1]) / w
        if not np.isfinite(x) or not np.isfinite(y):
            raise DegenerateCorrespondencesError(
                f'Image point {point} maps to a non-finite court position.'
            )
        return (x, y)
