"""Exact continuous-V graph and optimum reduction for K=2 and arbitrary J.

The predictor-donor convex hull is required to be two-dimensional. The full
lower graph, projected onto donor weights, is represented as a finite union of
rational equality polytopes with nonnegative variables. Exact outer optima and
both sign-breakdown notions are obtained by support enumeration; positive-gap
effect endpoints are enclosed by exact cutoff-feasibility bisection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .scm_oracle import (
    Matrix,
    QPSolutionSet,
    Rat,
    Vector,
    _dot,
    equality_constrained_qp_vertices,
    polytope_vertices,
)


Point = Tuple[Rat, Rat]


@dataclass(frozen=True)
class GraphPolytope:
    """A branch ``{variables >= 0: A variables = b}`` of the lower graph.

    The first ``donor_count`` variables are donor weights. Any remaining
    variables encode a fitted-predictor segment as a two-weight simplex.
    """

    name: str
    donor_count: int
    equality_rows: Matrix
    equality_rhs: Vector

    @property
    def variable_count(self) -> int:
        return len(self.equality_rows[0])


@dataclass(frozen=True)
class ExactBranchOptimum:
    branch: GraphPolytope
    value: Rat
    optimizer_vertices: Tuple[Vector, ...]


@dataclass(frozen=True)
class CertifiedInterval:
    lower: Rat
    upper: Rat


@dataclass(frozen=True)
class CertifiedNearEffectEnvelope:
    lower_effect: CertifiedInterval
    upper_effect: CertifiedInterval
    bisection_steps: int


@dataclass(frozen=True)
class TwoPredictorReduction:
    hull: Tuple[Point, ...]
    branches: Tuple[GraphPolytope, ...]
    branch_optima: Tuple[ExactBranchOptimum, ...]
    outer_optimum: Rat
    exact_optimum_effects: Tuple[Rat, Rat]
    attainable_effects: Tuple[Rat, Rat]
    sign_breakdown_epsilon: Optional[Rat]
    attained_zero_epsilon: Optional[Rat]
    validation_target: Vector
    validation_donors: Matrix
    treated_post: Rat
    donor_post: Vector

    def minimum_loss_with_effect_bound(
        self,
        cutoff: Rat,
        relation: str,
    ) -> Optional[Rat]:
        """Exact minimum loss subject to ``tau <= cutoff`` or ``tau >= cutoff``."""

        if relation not in ("at_most", "at_least"):
            raise ValueError("relation must be 'at_most' or 'at_least'")
        return _minimum_with_effect_constraint(
            self.branches,
            self.validation_target,
            self.validation_donors,
            self.treated_post,
            self.donor_post,
            relation,
            cutoff,
        )

    def near_effect_envelope(
        self,
        epsilon: Rat,
        digits: int = 12,
    ) -> CertifiedNearEffectEnvelope:
        """Return rigorous rational boxes for the sharp positive-gap endpoints."""

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
        target_width = Rat(1, 10**digits)
        steps = 0

        attainable_lower, attainable_upper = self.attainable_effects
        if _effect_bound_feasible(self, attainable_lower, "at_most", threshold):
            lower_box = CertifiedInterval(attainable_lower, attainable_lower)
        else:
            lower, upper = attainable_lower, self.exact_optimum_effects[0]
            while upper - lower > target_width:
                midpoint = (lower + upper) / 2
                if _effect_bound_feasible(self, midpoint, "at_most", threshold):
                    upper = midpoint
                else:
                    lower = midpoint
                steps += 1
            lower_box = CertifiedInterval(lower, upper)

        if _effect_bound_feasible(self, attainable_upper, "at_least", threshold):
            upper_box = CertifiedInterval(attainable_upper, attainable_upper)
        else:
            lower, upper = self.exact_optimum_effects[1], attainable_upper
            while upper - lower > target_width:
                midpoint = (lower + upper) / 2
                if _effect_bound_feasible(self, midpoint, "at_least", threshold):
                    lower = midpoint
                else:
                    upper = midpoint
                steps += 1
            upper_box = CertifiedInterval(lower, upper)

        return CertifiedNearEffectEnvelope(lower_box, upper_box, steps)


def _column(matrix: Matrix, index: int) -> Point:
    return matrix[0][index], matrix[1][index]


def _cross(origin: Point, first: Point, second: Point) -> Rat:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def _convex_hull(points: Sequence[Point]) -> Tuple[Point, ...]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return tuple(unique)

    lower: List[Point] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Point] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _linear_root(intercept: Rat, slope: Rat) -> Optional[Rat]:
    return None if slope == 0 else -intercept / slope


def _linear_inequality_interval(intercept: Rat, slope: Rat, lower: Rat, upper: Rat):
    if slope == 0:
        return None if intercept < 0 else (lower, upper)
    boundary = -intercept / slope
    if slope > 0:
        lower = max(lower, boundary)
    else:
        upper = min(upper, boundary)
    lower, upper = max(lower, Rat(0)), min(upper, Rat(1))
    return None if lower > upper else (lower, upper)


def _edge_feasible_at(
    parameter: Rat,
    target: Point,
    start: Point,
    end: Point,
    donor_points: Sequence[Point],
) -> bool:
    direction = end[0] - start[0], end[1] - start[1]
    residual = (
        start[0] + parameter * direction[0] - target[0],
        start[1] + parameter * direction[1] - target[1],
    )
    active_gradient_differences = (
        direction[0] * residual[0],
        direction[1] * residual[1],
    )
    inactive_gradient_differences = tuple(
        (
            (point[0] - start[0]) * residual[0],
            (point[1] - start[1]) * residual[1],
        )
        for point in donor_points
    )

    if active_gradient_differences[0] == active_gradient_differences[1]:
        if active_gradient_differences[0] != 0:
            return False
        interval: Optional[Tuple[Rat, Rat]] = (Rat(0), Rat(1))
        for first, second in inactive_gradient_differences:
            if interval is None:
                return False
            interval = _linear_inequality_interval(
                second,
                first - second,
                interval[0],
                interval[1],
            )
        return interval is not None

    predictor_weight = -active_gradient_differences[1] / (
        active_gradient_differences[0] - active_gradient_differences[1]
    )
    return Rat(0) <= predictor_weight <= Rat(1) and all(
        predictor_weight * first + (1 - predictor_weight) * second >= 0
        for first, second in inactive_gradient_differences
    )


def _edge_parameter_intervals(
    target: Point,
    start: Point,
    end: Point,
    donor_points: Sequence[Point],
) -> Tuple[Tuple[Rat, Rat], ...]:
    direction = end[0] - start[0], end[1] - start[1]
    residual_intercept = start[0] - target[0], start[1] - target[1]
    breakpoints = {Rat(0), Rat(1)}

    active_intercepts = tuple(
        direction[k] * residual_intercept[k] for k in range(2)
    )
    active_slopes = tuple(direction[k] * direction[k] for k in range(2))
    for intercept, slope in zip(active_intercepts, active_slopes):
        root = _linear_root(intercept, slope)
        if root is not None and 0 <= root <= 1:
            breakpoints.add(root)
    root = _linear_root(
        active_intercepts[0] - active_intercepts[1],
        active_slopes[0] - active_slopes[1],
    )
    if root is not None and 0 <= root <= 1:
        breakpoints.add(root)

    ordered = sorted(breakpoints)
    pieces: List[Tuple[Rat, Rat]] = []
    for left, right in zip(ordered, ordered[1:]):
        if _edge_feasible_at((left + right) / 2, target, start, end, donor_points):
            pieces.append((left, right))
    for point in ordered:
        if _edge_feasible_at(point, target, start, end, donor_points):
            pieces.append((point, point))

    pieces.sort()
    merged: List[Tuple[Rat, Rat]] = []
    for left, right in pieces:
        if merged and left <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], right)
        else:
            merged.append((left, right))
    return tuple(merged)


def _vertex_attainable(target: Point, vertex: Point, donor_points: Sequence[Point]) -> bool:
    interval: Optional[Tuple[Rat, Rat]] = (Rat(0), Rat(1))
    residual = vertex[0] - target[0], vertex[1] - target[1]
    for point in donor_points:
        if interval is None:
            return False
        first = (point[0] - vertex[0]) * residual[0]
        second = (point[1] - vertex[1]) * residual[1]
        interval = _linear_inequality_interval(
            second,
            first - second,
            interval[0],
            interval[1],
        )
    return interval is not None


def _inside_convex_polygon(point: Point, hull: Sequence[Point]) -> bool:
    return all(
        _cross(hull[index], hull[(index + 1) % len(hull)], point) >= 0
        for index in range(len(hull))
    )


def _point_branch(name: str, point: Point, predictor_donors: Matrix) -> GraphPolytope:
    donor_count = len(predictor_donors[0])
    return GraphPolytope(
        name,
        donor_count,
        (tuple(Rat(1) for _ in range(donor_count)),) + tuple(predictor_donors),
        (Rat(1),) + point,
    )


def _segment_branch(
    name: str,
    start: Point,
    end: Point,
    predictor_donors: Matrix,
) -> GraphPolytope:
    if start == end:
        return _point_branch(name, start, predictor_donors)
    if end < start:
        start, end = end, start
    donor_count = len(predictor_donors[0])
    zeros = (Rat(0), Rat(0))
    rows: List[Vector] = [tuple(Rat(1) for _ in range(donor_count)) + zeros]
    rhs: List[Rat] = [Rat(1)]
    rows.append(tuple(Rat(0) for _ in range(donor_count)) + (Rat(1), Rat(1)))
    rhs.append(Rat(1))
    for coordinate in range(2):
        rows.append(
            tuple(predictor_donors[coordinate])
            + (-start[coordinate], -end[coordinate])
        )
        rhs.append(Rat(0))
    return GraphPolytope(name, donor_count, tuple(rows), tuple(rhs))


def _boundary_branch(
    name: str,
    predictor_row: int,
    fitted_value: Rat,
    predictor_donors: Matrix,
) -> GraphPolytope:
    donor_count = len(predictor_donors[0])
    return GraphPolytope(
        name,
        donor_count,
        (tuple(Rat(1) for _ in range(donor_count)), predictor_donors[predictor_row]),
        (Rat(1), fitted_value),
    )


def _construct_graph_branches(
    predictor_target: Point,
    predictor_donors: Matrix,
) -> Tuple[Tuple[Point, ...], Tuple[GraphPolytope, ...]]:
    donor_count = len(predictor_donors[0])
    donor_points = tuple(_column(predictor_donors, index) for index in range(donor_count))
    hull = _convex_hull(donor_points)
    if len(hull) < 3:
        raise ValueError("predictor-donor convex hull must be two-dimensional")

    branches: List[GraphPolytope] = []
    for coordinate in range(2):
        donor_values = predictor_donors[coordinate]
        fitted = min(max(predictor_target[coordinate], min(donor_values)), max(donor_values))
        branches.append(
            _boundary_branch(
                f"predictor-boundary-{coordinate}",
                coordinate,
                fitted,
                predictor_donors,
            )
        )

    geometry = {}
    if _inside_convex_polygon(predictor_target, hull):
        geometry[(predictor_target, predictor_target)] = "interior-match"

    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        for interval_index, (left, right) in enumerate(
            _edge_parameter_intervals(predictor_target, start, end, donor_points)
        ):
            interval_start = tuple(
                start[k] + left * (end[k] - start[k]) for k in range(2)
            )
            interval_end = tuple(
                start[k] + right * (end[k] - start[k]) for k in range(2)
            )
            key = tuple(sorted((interval_start, interval_end)))
            geometry.setdefault(key, f"hull-edge-{index}-{interval_index}")

    for index, vertex in enumerate(hull):
        if _vertex_attainable(predictor_target, vertex, donor_points):
            geometry.setdefault((vertex, vertex), f"hull-vertex-{index}")

    for (start, end), name in sorted(geometry.items()):
        branches.append(_segment_branch(name, start, end, predictor_donors))
    return hull, tuple(branches)


def _solve_branch(
    branch: GraphPolytope,
    validation_target: Vector,
    validation_donors: Matrix,
) -> QPSolutionSet:
    auxiliary_count = branch.variable_count - branch.donor_count
    augmented_donors = tuple(
        tuple(row) + tuple(Rat(0) for _ in range(auxiliary_count))
        for row in validation_donors
    )
    validation_weights = tuple(Rat(1, len(validation_target)) for _ in validation_target)
    return equality_constrained_qp_vertices(
        validation_target,
        augmented_donors,
        validation_weights,
        branch.equality_rows,
        branch.equality_rhs,
    )


def _effect(treated_post: Rat, donor_post: Vector, variables: Vector) -> Rat:
    return treated_post - _dot(donor_post, variables[: len(donor_post)])


def _with_effect_constraint(
    branch: GraphPolytope,
    donor_post: Vector,
    relation: str,
    effect_cutoff: Rat,
    treated_post: Rat,
) -> GraphPolytope:
    auxiliary_count = branch.variable_count - branch.donor_count
    effect_coefficients = tuple(donor_post) + tuple(
        Rat(0) for _ in range(auxiliary_count)
    )
    effect_rhs = treated_post - effect_cutoff
    if relation == "equal":
        return GraphPolytope(
            f"{branch.name}-effect-equal",
            branch.donor_count,
            branch.equality_rows + (effect_coefficients,),
            branch.equality_rhs + (effect_rhs,),
        )
    if relation not in ("at_least", "at_most"):
        raise ValueError("unknown effect relation")
    slack_coefficient = Rat(1) if relation == "at_least" else Rat(-1)
    return GraphPolytope(
        f"{branch.name}-effect-{relation}",
        branch.donor_count,
        tuple(row + (Rat(0),) for row in branch.equality_rows)
        + (effect_coefficients + (slack_coefficient,),),
        branch.equality_rhs + (effect_rhs,),
    )


def _minimum_with_effect_constraint(
    branches: Sequence[GraphPolytope],
    validation_target: Vector,
    validation_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
    relation: str,
    effect_cutoff: Rat = Rat(0),
) -> Optional[Rat]:
    values: List[Rat] = []
    for branch in branches:
        constrained = _with_effect_constraint(
            branch,
            donor_post,
            relation,
            effect_cutoff,
            treated_post,
        )
        try:
            values.append(_solve_branch(constrained, validation_target, validation_donors).value)
        except ValueError:
            continue
    return min(values) if values else None


def _effect_bound_feasible(
    reduction: TwoPredictorReduction,
    cutoff: Rat,
    relation: str,
    threshold: Rat,
) -> bool:
    for branch in reduction.branches:
        constrained = _with_effect_constraint(
            branch,
            reduction.donor_post,
            relation,
            cutoff,
            reduction.treated_post,
        )
        try:
            value = _solve_branch(
                constrained,
                reduction.validation_target,
                reduction.validation_donors,
            ).value
        except ValueError:
            continue
        if value <= threshold:
            return True
    return False


def reduce_two_predictor_problem(
    predictor_target: Vector,
    predictor_donors: Matrix,
    validation_target: Vector,
    validation_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
) -> TwoPredictorReduction:
    if len(predictor_target) != 2 or len(predictor_donors) != 2:
        raise ValueError("requires exactly two predictors")
    if not predictor_donors[0] or len(predictor_donors[1]) != len(predictor_donors[0]):
        raise ValueError("predictor donor matrix is empty or ragged")
    donor_count = len(predictor_donors[0])
    if len(donor_post) != donor_count:
        raise ValueError("post-treatment donor dimension does not match")
    if not validation_target or len(validation_target) != len(validation_donors) or any(
        len(row) != donor_count for row in validation_donors
    ):
        raise ValueError("validation dimensions do not match")

    hull, branches = _construct_graph_branches(
        (predictor_target[0], predictor_target[1]), predictor_donors
    )
    solved = tuple(
        ExactBranchOptimum(branch, solution.value, solution.vertices)
        for branch in branches
        for solution in (_solve_branch(branch, validation_target, validation_donors),)
    )
    outer_optimum = min(result.value for result in solved)
    exact_effects = tuple(
        _effect(treated_post, donor_post, variables)
        for result in solved
        if result.value == outer_optimum
        for variables in result.optimizer_vertices
    )
    attainable_effects = tuple(
        _effect(treated_post, donor_post, variables)
        for branch in branches
        for variables in polytope_vertices(branch.equality_rows, branch.equality_rhs)
    )

    nonpositive = _minimum_with_effect_constraint(
        branches,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
        "at_most",
    )
    nonnegative = _minimum_with_effect_constraint(
        branches,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
        "at_least",
    )
    zero = _minimum_with_effect_constraint(
        branches,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
        "equal",
    )
    span_breakdown = (
        max(nonpositive, nonnegative) - outer_optimum
        if nonpositive is not None and nonnegative is not None
        else None
    )
    attained_zero = zero - outer_optimum if zero is not None else None

    return TwoPredictorReduction(
        hull,
        branches,
        solved,
        outer_optimum,
        (min(exact_effects), max(exact_effects)),
        (min(attainable_effects), max(attainable_effects)),
        span_breakdown,
        attained_zero,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
