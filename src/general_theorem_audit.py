"""Independent exact pointwise audit for the finite lower-graph theorem.

The two feasibility systems below encode the same membership question through
different variables.  ``direct_predictor_weight_witness`` uses the original
predictor weights and donor-QP first-order inequalities.
``normal_projection_witness`` uses the face/orthant normal characterization
from Theorem 1.  Both use only rational arithmetic and exhaustive vertex
enumeration of a bounded nonnegative equality polytope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .scm_oracle import Matrix, Rat, Vector, _dot, polytope_vertices


@dataclass(frozen=True)
class ProjectionAuditWitness:
    predictor_weights: Vector
    normal: Vector
    route: str


def _validate_point(
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
) -> tuple[Vector, Vector]:
    dimension = len(predictor_target)
    if dimension < 1 or len(predictor_donors) != dimension:
        raise ValueError("predictor dimensions do not match")
    donor_count = len(donor_weights)
    if donor_count < 1 or any(len(row) != donor_count for row in predictor_donors):
        raise ValueError("donor dimensions do not match")
    if any(value < 0 for value in donor_weights) or sum(donor_weights, Rat(0)) != 1:
        raise ValueError("donor weights must lie in the simplex")
    fitted = tuple(_dot(row, donor_weights) for row in predictor_donors)
    residual = tuple(
        target - value for target, value in zip(predictor_target, fitted)
    )
    return fitted, residual


def direct_predictor_weight_witness(
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
) -> Optional[ProjectionAuditWitness]:
    """Decide whether ``w`` solves the lower problem for some ``v in Delta``.

    For fixed ``w``, the conditions

        q = diag(v) (t - Xw),  q'(x_j-Xw) <= 0  for every donor j

    are linear in ``v``.  Nonnegative slacks convert them to an exact bounded
    equality-polytope feasibility problem.
    """

    fitted, residual = _validate_point(
        predictor_target, predictor_donors, donor_weights
    )
    dimension = len(predictor_target)
    donor_count = len(donor_weights)
    rows = [
        tuple(Rat(1) for _ in range(dimension))
        + tuple(Rat(0) for _ in range(donor_count))
    ]
    rhs = [Rat(1)]
    for donor in range(donor_count):
        rows.append(
            tuple(
                residual[k] * (predictor_donors[k][donor] - fitted[k])
                for k in range(dimension)
            )
            + tuple(
                Rat(1) if index == donor else Rat(0)
                for index in range(donor_count)
            )
        )
        rhs.append(Rat(0))
    try:
        variables = polytope_vertices(tuple(rows), tuple(rhs))[0]
    except ValueError:
        return None
    predictor_weights = variables[:dimension]
    normal = tuple(
        predictor_weights[k] * residual[k] for k in range(dimension)
    )
    return ProjectionAuditWitness(predictor_weights, normal, "direct-kkt")


def normal_projection_witness(
    predictor_target: Vector,
    predictor_donors: Matrix,
    donor_weights: Vector,
) -> Optional[ProjectionAuditWitness]:
    """Decide membership in Theorem 1's coordinate/normal-face union.

    When no residual vanishes, normalize a sign-compatible normal by its
    coordinatewise absolute sum.  The normal inequalities at ``y`` imply that
    ``y`` lies in the exposed face selected by that normal.  This includes the
    full hull when the hull is lower-dimensional.
    """

    fitted, residual = _validate_point(
        predictor_target, predictor_donors, donor_weights
    )
    dimension = len(predictor_target)
    donor_count = len(donor_weights)
    zero_coordinate = next(
        (index for index, value in enumerate(residual) if value == 0), None
    )
    if zero_coordinate is not None:
        predictor_weights = tuple(
            Rat(1) if index == zero_coordinate else Rat(0)
            for index in range(dimension)
        )
        return ProjectionAuditWitness(
            predictor_weights,
            tuple(Rat(0) for _ in range(dimension)),
            "coordinate-match",
        )

    sign = tuple(Rat(1) if value > 0 else Rat(-1) for value in residual)
    rows = [
        tuple(Rat(1) for _ in range(dimension))
        + tuple(Rat(0) for _ in range(donor_count))
    ]
    rhs = [Rat(1)]
    for donor in range(donor_count):
        rows.append(
            tuple(
                sign[k] * (predictor_donors[k][donor] - fitted[k])
                for k in range(dimension)
            )
            + tuple(
                Rat(1) if index == donor else Rat(0)
                for index in range(donor_count)
            )
        )
        rhs.append(Rat(0))
    try:
        variables = polytope_vertices(tuple(rows), tuple(rhs))[0]
    except ValueError:
        return None
    magnitudes = variables[:dimension]
    normal = tuple(sign[k] * magnitudes[k] for k in range(dimension))
    unnormalized = tuple(normal[k] / residual[k] for k in range(dimension))
    scale = sum(unnormalized, Rat(0))
    if scale <= 0:
        raise AssertionError("nonzero sign-compatible normal induced zero weight")
    predictor_weights = tuple(value / scale for value in unnormalized)
    return ProjectionAuditWitness(predictor_weights, normal, "normal-face")
