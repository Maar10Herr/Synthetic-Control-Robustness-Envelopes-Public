"""Exact-acceptance validation-loss relaxation over the donor simplex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .scalable_qp import ScalableQPCertificate
from .scm_oracle import Matrix, Rat, Vector, _dot
from .two_predictor import (
    CertifiedInterval,
    CertifiedNearEffectEnvelope,
    GraphPolytope,
    _with_effect_constraint,
)
from .two_predictor_scalable import _branch_effect_range, _effect, _solve_branch


@dataclass(frozen=True)
class ScalableSimplexRelaxation:
    branch: GraphPolytope
    optimum: ScalableQPCertificate
    exact_optimum_effects: Tuple[Rat, Rat]
    attainable_effects: Tuple[Rat, Rat]
    sign_breakdown_epsilon: Optional[Rat]
    attained_zero_epsilon: Optional[Rat]
    validation_target: Vector
    validation_donors: Matrix
    treated_post: Rat
    donor_post: Vector

    @property
    def outer_optimum(self) -> Rat:
        return self.optimum.exact.value

    def effect_bound_certificate(
        self,
        cutoff: Rat,
        relation: str,
    ) -> Optional[ScalableQPCertificate]:
        if relation not in ("at_most", "at_least", "equal"):
            raise ValueError("unknown effect relation")
        lower, upper = self.attainable_effects
        if relation == "at_most" and cutoff < lower:
            return None
        if relation == "at_least" and cutoff > upper:
            return None
        if relation == "equal" and not lower <= cutoff <= upper:
            return None
        if relation == "at_most" and upper <= cutoff:
            return self.optimum
        if relation == "at_least" and lower >= cutoff:
            return self.optimum
        constrained = _with_effect_constraint(
            self.branch,
            self.donor_post,
            relation,
            cutoff,
            self.treated_post,
        )
        try:
            return _solve_branch(
                constrained,
                self.validation_target,
                self.validation_donors,
            )
        except ValueError:
            return None

    def minimum_loss_with_effect_bound(
        self,
        cutoff: Rat,
        relation: str,
    ) -> Optional[Rat]:
        certificate = self.effect_bound_certificate(cutoff, relation)
        return certificate.exact.value if certificate is not None else None

    def near_effect_envelope(
        self,
        epsilon: Rat,
        digits: int = 6,
    ) -> CertifiedNearEffectEnvelope:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if digits < 1:
            raise ValueError("digits must be positive")
        if epsilon == 0:
            lower, upper = self.exact_optimum_effects
            return CertifiedNearEffectEnvelope(
                CertifiedInterval(lower, lower),
                CertifiedInterval(upper, upper),
                0,
            )

        threshold = self.outer_optimum + epsilon
        width = Rat(1, 10**digits)
        steps = 0
        lower, upper = self.attainable_effects[0], self.exact_optimum_effects[0]
        extreme_value = self.minimum_loss_with_effect_bound(lower, "at_most")
        if extreme_value is not None and extreme_value <= threshold:
            lower_box = CertifiedInterval(lower, lower)
        else:
            while upper - lower > width:
                midpoint = (lower + upper) / 2
                value = self.minimum_loss_with_effect_bound(midpoint, "at_most")
                if value is not None and value <= threshold:
                    upper = midpoint
                else:
                    lower = midpoint
                steps += 1
            lower_box = CertifiedInterval(lower, upper)

        lower, upper = self.exact_optimum_effects[1], self.attainable_effects[1]
        extreme_value = self.minimum_loss_with_effect_bound(upper, "at_least")
        if extreme_value is not None and extreme_value <= threshold:
            upper_box = CertifiedInterval(upper, upper)
        else:
            while upper - lower > width:
                midpoint = (lower + upper) / 2
                value = self.minimum_loss_with_effect_bound(midpoint, "at_least")
                if value is not None and value <= threshold:
                    lower = midpoint
                else:
                    upper = midpoint
                steps += 1
            upper_box = CertifiedInterval(lower, upper)
        return CertifiedNearEffectEnvelope(lower_box, upper_box, steps)


def reduce_to_simplex_relaxation(
    validation_target: Vector,
    validation_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
) -> ScalableSimplexRelaxation:
    donor_count = len(donor_post)
    if (
        donor_count < 2
        or not validation_target
        or len(validation_donors) != len(validation_target)
        or any(
            len(row) != donor_count for row in validation_donors
        )
    ):
        raise ValueError("validation and post-treatment dimensions do not match")
    branch = GraphPolytope(
        "donor-simplex",
        donor_count,
        (tuple(Rat(1) for _ in range(donor_count)),),
        (Rat(1),),
    )
    optimum = _solve_branch(branch, validation_target, validation_donors)
    fitted = tuple(
        _dot(row, optimum.exact.optimizer) for row in validation_donors
    )
    optimizer_branch = GraphPolytope(
        "validation-optimizer",
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
    provisional = ScalableSimplexRelaxation(
        branch,
        optimum,
        exact_effects,
        attainable,
        None,
        None,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
    nonpositive = provisional.minimum_loss_with_effect_bound(Rat(0), "at_most")
    nonnegative = provisional.minimum_loss_with_effect_bound(Rat(0), "at_least")
    zero = provisional.minimum_loss_with_effect_bound(Rat(0), "equal")
    span = (
        max(nonpositive, nonnegative) - optimum.exact.value
        if nonpositive is not None and nonnegative is not None
        else None
    )
    attained_zero = zero - optimum.exact.value if zero is not None else None
    return ScalableSimplexRelaxation(
        branch,
        optimum,
        exact_effects,
        attainable,
        span,
        attained_zero,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
