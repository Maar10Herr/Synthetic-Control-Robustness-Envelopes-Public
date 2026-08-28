"""Scalable K=2 envelope solver with numerical discovery and exact acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .scalable_qp import (
    ScalableQPCertificate,
    propose_and_certify_lp,
    propose_and_certify_qp,
)
from .scm_oracle import Matrix, Rat, Vector, _dot
from .two_predictor import (
    CertifiedInterval,
    CertifiedNearEffectEnvelope,
    GraphPolytope,
    Point,
    _construct_graph_branches,
    _with_effect_constraint,
)


@dataclass(frozen=True)
class ScalableBranchResult:
    branch: GraphPolytope
    certificate: ScalableQPCertificate
    effect_range: Tuple[Rat, Rat]


@dataclass(frozen=True)
class ScalableTwoPredictorReduction:
    hull: Tuple[Point, ...]
    branch_results: Tuple[ScalableBranchResult, ...]
    outer_optimum: Rat
    exact_optimum_effects: Tuple[Rat, Rat]
    attainable_effects: Tuple[Rat, Rat]
    sign_breakdown_epsilon: Optional[Rat]
    attained_zero_epsilon: Optional[Rat]
    validation_target: Vector
    validation_donors: Matrix
    treated_post: Rat
    donor_post: Vector

    @property
    def branches(self) -> Tuple[GraphPolytope, ...]:
        return tuple(result.branch for result in self.branch_results)

    def minimum_loss_with_effect_bound(
        self,
        cutoff: Rat,
        relation: str,
    ) -> Optional[Rat]:
        if relation not in ("at_most", "at_least", "equal"):
            raise ValueError("unknown effect relation")
        return _minimum_with_effect_constraint(self, cutoff, relation)

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
        attainable_lower, attainable_upper = self.attainable_effects

        lower, upper = attainable_lower, self.exact_optimum_effects[0]
        if _bound_feasible(self, attainable_lower, "at_most", threshold):
            lower_box = CertifiedInterval(attainable_lower, attainable_lower)
        else:
            while upper - lower > width:
                midpoint = (lower + upper) / 2
                if _bound_feasible(self, midpoint, "at_most", threshold):
                    upper = midpoint
                else:
                    lower = midpoint
                steps += 1
            lower_box = CertifiedInterval(lower, upper)

        lower, upper = self.exact_optimum_effects[1], attainable_upper
        if _bound_feasible(self, attainable_upper, "at_least", threshold):
            upper_box = CertifiedInterval(attainable_upper, attainable_upper)
        else:
            while upper - lower > width:
                midpoint = (lower + upper) / 2
                if _bound_feasible(self, midpoint, "at_least", threshold):
                    lower = midpoint
                else:
                    upper = midpoint
                steps += 1
            upper_box = CertifiedInterval(lower, upper)
        return CertifiedNearEffectEnvelope(lower_box, upper_box, steps)


def _augmented_validation(branch: GraphPolytope, validation_donors: Matrix) -> Matrix:
    auxiliary_count = branch.variable_count - branch.donor_count
    return tuple(
        tuple(row) + tuple(Rat(0) for _ in range(auxiliary_count))
        for row in validation_donors
    )


def _solve_branch(
    branch: GraphPolytope,
    validation_target: Vector,
    validation_donors: Matrix,
) -> ScalableQPCertificate:
    row_weights = tuple(Rat(1, len(validation_target)) for _ in validation_target)
    return propose_and_certify_qp(
        validation_target,
        _augmented_validation(branch, validation_donors),
        row_weights,
        branch.equality_rows,
        branch.equality_rhs,
    )


def _effect(treated_post: Rat, donor_post: Vector, variables: Vector) -> Rat:
    return treated_post - _dot(donor_post, variables[: len(donor_post)])


def _branch_effect_range(
    branch: GraphPolytope,
    treated_post: Rat,
    donor_post: Vector,
) -> Tuple[Rat, Rat]:
    auxiliary_count = branch.variable_count - branch.donor_count
    lower_cost = tuple(-value for value in donor_post) + tuple(
        Rat(0) for _ in range(auxiliary_count)
    )
    upper_cost = tuple(donor_post) + tuple(
        Rat(0) for _ in range(auxiliary_count)
    )
    lower = propose_and_certify_lp(
        lower_cost,
        branch.equality_rows,
        branch.equality_rhs,
    ).exact.value
    upper = propose_and_certify_lp(
        upper_cost,
        branch.equality_rows,
        branch.equality_rhs,
    ).exact.value
    return treated_post + lower, treated_post - upper


def _relation_attainable(effect_range: Tuple[Rat, Rat], cutoff: Rat, relation: str) -> bool:
    lower, upper = effect_range
    if relation == "at_most":
        return lower <= cutoff
    if relation == "at_least":
        return upper >= cutoff
    if relation == "equal":
        return lower <= cutoff <= upper
    raise ValueError("unknown effect relation")


def _minimum_with_effect_constraint(
    reduction: ScalableTwoPredictorReduction,
    cutoff: Rat,
    relation: str,
) -> Optional[Rat]:
    values: List[Rat] = []
    for result in reduction.branch_results:
        if not _relation_attainable(result.effect_range, cutoff, relation):
            continue
        if (
            (relation == "at_most" and result.effect_range[1] <= cutoff)
            or (relation == "at_least" and result.effect_range[0] >= cutoff)
        ):
            values.append(result.certificate.exact.value)
            continue
        constrained = _with_effect_constraint(
            result.branch,
            reduction.donor_post,
            relation,
            cutoff,
            reduction.treated_post,
        )
        values.append(
            _solve_branch(
                constrained,
                reduction.validation_target,
                reduction.validation_donors,
            ).exact.value
        )
    return min(values) if values else None


def _bound_feasible(
    reduction: ScalableTwoPredictorReduction,
    cutoff: Rat,
    relation: str,
    threshold: Rat,
) -> bool:
    value = _minimum_with_effect_constraint(reduction, cutoff, relation)
    return value is not None and value <= threshold


def reduce_two_predictor_problem_scalable(
    predictor_target: Vector,
    predictor_donors: Matrix,
    validation_target: Vector,
    validation_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
) -> ScalableTwoPredictorReduction:
    if len(predictor_target) != 2 or len(predictor_donors) != 2:
        raise ValueError("requires exactly two predictors")
    donor_count = len(predictor_donors[0])
    if donor_count < 3 or len(predictor_donors[1]) != donor_count:
        raise ValueError("predictor donor dimensions do not match")
    if len(donor_post) != donor_count:
        raise ValueError("post-treatment donor dimension does not match")
    if not validation_target or len(validation_target) != len(validation_donors) or any(
        len(row) != donor_count for row in validation_donors
    ):
        raise ValueError("validation dimensions do not match")

    hull, branches = _construct_graph_branches(
        (predictor_target[0], predictor_target[1]), predictor_donors
    )
    branch_results = tuple(
        ScalableBranchResult(
            branch,
            _solve_branch(branch, validation_target, validation_donors),
            _branch_effect_range(branch, treated_post, donor_post),
        )
        for branch in branches
    )
    outer_optimum = min(result.certificate.exact.value for result in branch_results)
    global_results = tuple(
        result
        for result in branch_results
        if result.certificate.exact.value == outer_optimum
    )
    exact_effects: List[Rat] = []
    for result in global_results:
        certificate = result.certificate.exact
        augmented_validation = _augmented_validation(
            result.branch, validation_donors
        )
        optimizer_rows = result.branch.equality_rows + augmented_validation
        optimizer_rhs = result.branch.equality_rhs + tuple(
            _dot(row, certificate.optimizer) for row in augmented_validation
        )
        optimizer_branch = GraphPolytope(
            f"{result.branch.name}-outer-optimizer",
            result.branch.donor_count,
            optimizer_rows,
            optimizer_rhs,
        )
        lower, upper = _branch_effect_range(
            optimizer_branch,
            treated_post,
            donor_post,
        )
        exact_effects.extend((lower, upper))

    attainable_lower = min(result.effect_range[0] for result in branch_results)
    attainable_upper = max(result.effect_range[1] for result in branch_results)
    provisional = ScalableTwoPredictorReduction(
        hull,
        branch_results,
        outer_optimum,
        (min(exact_effects), max(exact_effects)),
        (attainable_lower, attainable_upper),
        None,
        None,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
    nonpositive = _minimum_with_effect_constraint(provisional, Rat(0), "at_most")
    nonnegative = _minimum_with_effect_constraint(provisional, Rat(0), "at_least")
    zero = _minimum_with_effect_constraint(provisional, Rat(0), "equal")
    span = (
        max(nonpositive, nonnegative) - outer_optimum
        if nonpositive is not None and nonnegative is not None
        else None
    )
    attained_zero = zero - outer_optimum if zero is not None else None
    return ScalableTwoPredictorReduction(
        hull,
        branch_results,
        outer_optimum,
        provisional.exact_optimum_effects,
        provisional.attainable_effects,
        span,
        attained_zero,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
