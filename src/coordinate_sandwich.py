"""Certified high-dimensional bounds from simplex and coordinate branches.

The donor simplex contains the complete ADH lower graph, so optimizing over it
gives exact outer (lower-loss) bounds.  Conversely, whenever the treated value
of predictor ``k`` is in the donor convex hull, every donor vector matching
that coordinate is inner-optimal for ``V = e_k``.  Optimizing over that
coordinate-match polytope therefore gives an exact feasible ADH candidate.

This module combines those two observations without claiming that an unmatched
sandwich is sharp.  Floating-point solvers only propose supports; the imported
scalable routines accept every LP/QP result with rational primal-dual checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .general_predictor import _coordinate_branch
from .scalable_qp import ScalableQPCertificate
from .scm_oracle import Matrix, Rat, Vector, _dot
from .simplex_relaxation import (
    ScalableSimplexRelaxation,
    reduce_to_simplex_relaxation,
)
from .two_predictor import (
    CertifiedInterval,
    CertifiedNearEffectEnvelope,
    GraphPolytope,
    _with_effect_constraint,
)
from .two_predictor_scalable import (
    _branch_effect_range,
    _solve_branch,
)


@dataclass(frozen=True)
class CoordinateCandidate:
    """Exact outer optimization on one ADH coordinate-match branch."""

    coordinate: int
    branch: GraphPolytope
    optimum: ScalableQPCertificate
    exact_optimum_effects: Tuple[Rat, Rat]
    attainable_effects: Tuple[Rat, Rat]
    zero_effect_optimum: Optional[ScalableQPCertificate]

    def as_relaxation(
        self,
        validation_target: Vector,
        validation_donors: Matrix,
        treated_post: Rat,
        donor_post: Vector,
    ) -> ScalableSimplexRelaxation:
        """Expose the branch through the common exact cutoff-QP interface."""

        return ScalableSimplexRelaxation(
            self.branch,
            self.optimum,
            self.exact_optimum_effects,
            self.attainable_effects,
            None,
            None,
            validation_target,
            validation_donors,
            treated_post,
            donor_post,
        )


@dataclass(frozen=True)
class RelativeEnvelopeSandwich:
    """Outward boxes for both endpoints at a relative outer-loss gap."""

    relative_gap: Rat
    outer_epsilon: Rat
    inner_epsilon: Rat
    relaxation_envelope: CertifiedNearEffectEnvelope
    coordinate_envelope: CertifiedNearEffectEnvelope
    full_lower_effect: CertifiedInterval
    full_upper_effect: CertifiedInterval


@dataclass(frozen=True)
class CoordinateSandwich:
    """Exact global bounds formed without enumerating the full lower graph."""

    relaxation: ScalableSimplexRelaxation
    candidates: Tuple[CoordinateCandidate, ...]
    best_candidate: CoordinateCandidate
    best_zero_candidate: Optional[CoordinateCandidate]
    validation_target: Vector
    validation_donors: Matrix
    treated_post: Rat
    donor_post: Vector

    @property
    def optimum_loss_bounds(self) -> Tuple[Rat, Rat]:
        return self.relaxation.outer_optimum, self.best_candidate.optimum.exact.value

    @property
    def zero_loss_bounds(self) -> Optional[Tuple[Rat, Rat]]:
        zero_gap = self.relaxation.attained_zero_epsilon
        lower = (
            self.relaxation.outer_optimum + zero_gap
            if zero_gap is not None
            else None
        )
        candidate = self.best_zero_candidate
        if lower is None or candidate is None or candidate.zero_effect_optimum is None:
            return None
        return lower, candidate.zero_effect_optimum.exact.value

    @property
    def absolute_zero_gap_bounds(self) -> Optional[Tuple[Rat, Rat]]:
        zero = self.zero_loss_bounds
        if zero is None:
            return None
        loss_lower, loss_upper = self.optimum_loss_bounds
        zero_lower, zero_upper = zero
        return zero_lower - loss_upper, zero_upper - loss_lower

    @property
    def relative_zero_gap_bounds(self) -> Optional[Tuple[Rat, Rat]]:
        zero = self.zero_loss_bounds
        if zero is None:
            return None
        loss_lower, loss_upper = self.optimum_loss_bounds
        zero_lower, zero_upper = zero
        if loss_lower <= 0:
            return None
        return zero_lower / loss_upper - 1, zero_upper / loss_lower - 1

    def optimum_effect_outer_box(self, digits: int = 6) -> CertifiedInterval:
        """Contain every effect attached to a globally minimal ADH solution."""

        loss_lower, loss_upper = self.optimum_loss_bounds
        envelope = self.relaxation.near_effect_envelope(
            loss_upper - loss_lower,
            digits=digits,
        )
        return CertifiedInterval(
            envelope.lower_effect.lower,
            envelope.upper_effect.upper,
        )

    def relative_envelope_sandwich(
        self,
        relative_gap: Rat,
        digits: int = 6,
    ) -> RelativeEnvelopeSandwich:
        """Bound the full ADH endpoints at ``L <= (1 + gap) L*``.

        The relaxation uses the largest possible threshold, based on the
        feasible upper bound for ``L*``.  The coordinate subgraph uses the
        smallest possible threshold, based on the simplex lower bound.  Thus
        the former contains the full near-optimal graph and the latter is
        contained in it.
        """

        if relative_gap < 0:
            raise ValueError("relative gap must be nonnegative")
        loss_lower, loss_upper = self.optimum_loss_bounds
        outer_epsilon = (1 + relative_gap) * loss_upper - loss_lower
        inner_epsilon = (1 + relative_gap) * loss_lower - loss_upper
        if inner_epsilon < 0:
            raise ValueError(
                "coordinate upper bound is too loose to give an inner envelope "
                "at this relative gap"
            )
        outer = self.relaxation.near_effect_envelope(
            outer_epsilon,
            digits=digits,
        )
        coordinate = self.best_candidate.as_relaxation(
            self.validation_target,
            self.validation_donors,
            self.treated_post,
            self.donor_post,
        ).near_effect_envelope(inner_epsilon, digits=digits)
        return RelativeEnvelopeSandwich(
            relative_gap,
            outer_epsilon,
            inner_epsilon,
            outer,
            coordinate,
            CertifiedInterval(
                outer.lower_effect.lower,
                coordinate.lower_effect.upper,
            ),
            CertifiedInterval(
                coordinate.upper_effect.lower,
                outer.upper_effect.upper,
            ),
        )


def reduce_with_coordinate_sandwich(
    predictor_target: Vector,
    predictor_donors: Matrix,
    validation_target: Vector,
    validation_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
) -> CoordinateSandwich:
    """Construct exact donor-simplex/coordinate-branch global bounds."""

    if not predictor_target or len(predictor_donors) != len(predictor_target):
        raise ValueError("predictor dimensions do not match")
    donor_count = len(donor_post)
    if any(len(row) != donor_count for row in predictor_donors):
        raise ValueError("predictor donor dimensions do not match")

    relaxation = reduce_to_simplex_relaxation(
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
    candidates = []
    for coordinate in range(len(predictor_target)):
        graph_branch = _coordinate_branch(
            coordinate,
            predictor_target,
            predictor_donors,
        )
        if graph_branch is None:
            continue
        branch = graph_branch.polytope
        optimum = _solve_branch(branch, validation_target, validation_donors)
        fitted = tuple(
            _dot(row, optimum.exact.optimizer) for row in validation_donors
        )
        optimizer_branch = GraphPolytope(
            f"coordinate-{coordinate}-validation-optimizer",
            donor_count,
            branch.equality_rows + validation_donors,
            branch.equality_rhs + fitted,
        )
        exact_effects = _branch_effect_range(
            optimizer_branch,
            treated_post,
            donor_post,
        )
        attainable = _branch_effect_range(branch, treated_post, donor_post)
        zero_optimum = None
        if attainable[0] <= 0 <= attainable[1]:
            zero_branch = _with_effect_constraint(
                branch,
                donor_post,
                "equal",
                Rat(0),
                treated_post,
            )
            zero_optimum = _solve_branch(
                zero_branch,
                validation_target,
                validation_donors,
            )
        candidates.append(
            CoordinateCandidate(
                coordinate,
                branch,
                optimum,
                exact_effects,
                attainable,
                zero_optimum,
            )
        )

    if not candidates:
        raise ValueError("no treated predictor coordinate is donor-matchable")
    best = min(candidates, key=lambda item: item.optimum.exact.value)
    zero_candidates = [item for item in candidates if item.zero_effect_optimum]
    best_zero = (
        min(zero_candidates, key=lambda item: item.zero_effect_optimum.exact.value)
        if zero_candidates
        else None
    )
    return CoordinateSandwich(
        relaxation,
        tuple(candidates),
        best,
        best_zero,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
