"""Certified continuous-V reduction for K=2, J=3 with affine-independent donors.

The lower-level graph projected onto donor weights is represented exactly as a
finite union of rational line segments. Outer-loss minima and both breakdown
thresholds are rational. Positive-gap effect endpoints are quadratic surds and
are returned with rigorous rational enclosures.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import isqrt
from typing import List, Optional, Sequence, Tuple

from .scm_oracle import Matrix, Rat, Vector, _dot, _solve_square, polytope_vertices
from .two_donor import AlgebraicInterval, QuadraticSurd


@dataclass(frozen=True)
class WeightBranch:
    name: str
    start: Vector
    end: Vector


@dataclass(frozen=True)
class BranchProfile:
    branch: WeightBranch
    alpha: Rat
    beta: Rat
    gamma: Rat
    minimum: Rat
    optimizer: Tuple[Rat, Rat]
    effect_intercept: Rat
    effect_slope: Rat

    def loss(self, parameter: Rat) -> Rat:
        return self.alpha * parameter * parameter - 2 * self.beta * parameter + self.gamma

    def effect(self, parameter: Rat) -> Rat:
        return self.effect_intercept + self.effect_slope * parameter


@dataclass(frozen=True)
class CertifiedScalar:
    lower: Rat
    upper: Rat


@dataclass(frozen=True)
class CertifiedEffectEnvelope:
    lower_effect: CertifiedScalar
    upper_effect: CertifiedScalar
    branch_effect_intervals: Tuple[Tuple[str, AlgebraicInterval], ...]


@dataclass(frozen=True)
class ThreeDonorK2Reduction:
    branches: Tuple[WeightBranch, ...]
    profiles: Tuple[BranchProfile, ...]
    outer_optimum: Rat
    exact_optimum_effects: Tuple[Rat, Rat]
    sign_breakdown_epsilon: Optional[Rat]
    attained_zero_epsilon: Optional[Rat]

    def near_effect_envelope(self, epsilon: Rat, digits: int = 30) -> CertifiedEffectEnvelope:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        threshold = self.outer_optimum + epsilon
        intervals: List[Tuple[str, AlgebraicInterval]] = []
        lower_boxes: List[Tuple[Rat, Rat]] = []
        upper_boxes: List[Tuple[Rat, Rat]] = []
        for profile in self.profiles:
            parameter_interval = _sublevel_interval(profile, threshold)
            if parameter_interval is None:
                continue
            if profile.effect_slope >= 0:
                lower_parameter, upper_parameter = parameter_interval.lower, parameter_interval.upper
            else:
                lower_parameter, upper_parameter = parameter_interval.upper, parameter_interval.lower
            effect_interval = AlgebraicInterval(
                lower_parameter.affine(profile.effect_intercept, profile.effect_slope),
                upper_parameter.affine(profile.effect_intercept, profile.effect_slope),
            )
            intervals.append((profile.branch.name, effect_interval))
            lower_boxes.append(_surd_enclosure(effect_interval.lower, digits))
            upper_boxes.append(_surd_enclosure(effect_interval.upper, digits))
        if not intervals:
            raise AssertionError("global optimizer disappeared from near-optimal set")
        return CertifiedEffectEnvelope(
            CertifiedScalar(min(box[0] for box in lower_boxes), min(box[1] for box in lower_boxes)),
            CertifiedScalar(max(box[0] for box in upper_boxes), max(box[1] for box in upper_boxes)),
            tuple(intervals),
        )


def _clamp(value: Rat, lower: Rat = Rat(0), upper: Rat = Rat(1)) -> Rat:
    return min(max(value, lower), upper)


def _column(matrix: Matrix, index: int) -> Vector:
    return tuple(row[index] for row in matrix)


def _canonical_branch(name: str, start: Vector, end: Vector) -> WeightBranch:
    if end < start:
        start, end = end, start
    return WeightBranch(name, start, end)


def _linear_root(intercept: Rat, slope: Rat) -> Optional[Rat]:
    return None if slope == 0 else -intercept / slope


def _edge_feasible_at(
    parameter: Rat,
    target: Vector,
    donors: Matrix,
    first: int,
    second: int,
    inactive: int,
) -> bool:
    point_first = _column(donors, first)
    point_second = _column(donors, second)
    point_inactive = _column(donors, inactive)
    difference = tuple(a - b for a, b in zip(point_first, point_second))
    residual = tuple(point_second[k] + parameter * difference[k] - target[k] for k in range(2))
    inactive_difference = tuple(a - b for a, b in zip(point_inactive, point_second))
    u = tuple(difference[k] * residual[k] for k in range(2))
    h = tuple(inactive_difference[k] * residual[k] for k in range(2))
    if u[0] == u[1]:
        if u[0] != 0:
            return False
        return max(h) >= 0
    predictor_weight = -u[1] / (u[0] - u[1])
    return Rat(0) <= predictor_weight <= Rat(1) and predictor_weight * h[0] + (1 - predictor_weight) * h[1] >= 0


def _edge_parameter_intervals(
    target: Vector,
    donors: Matrix,
    first: int,
    second: int,
    inactive: int,
) -> Tuple[Tuple[Rat, Rat], ...]:
    point_first = _column(donors, first)
    point_second = _column(donors, second)
    difference = tuple(a - b for a, b in zip(point_first, point_second))
    offset = tuple(target[k] - point_second[k] for k in range(2))
    breakpoints = {Rat(0), Rat(1)}
    for k in range(2):
        root = _linear_root(-difference[k] * offset[k], difference[k] * difference[k])
        if root is not None and 0 <= root <= 1:
            breakpoints.add(root)
    denominator_intercept = -difference[0] * offset[0] + difference[1] * offset[1]
    denominator_slope = difference[0] * difference[0] - difference[1] * difference[1]
    root = _linear_root(denominator_intercept, denominator_slope)
    if root is not None and 0 <= root <= 1:
        breakpoints.add(root)
    ordered = sorted(breakpoints)
    pieces: List[Tuple[Rat, Rat]] = []
    for left, right in zip(ordered, ordered[1:]):
        midpoint = (left + right) / 2
        if _edge_feasible_at(midpoint, target, donors, first, second, inactive):
            pieces.append((left, right))
    for point in ordered:
        if _edge_feasible_at(point, target, donors, first, second, inactive):
            pieces.append((point, point))
    if not pieces:
        return ()
    pieces.sort()
    merged: List[Tuple[Rat, Rat]] = [pieces[0]]
    for left, right in pieces[1:]:
        old_left, old_right = merged[-1]
        if left <= old_right:
            merged[-1] = (old_left, max(old_right, right))
        else:
            merged.append((left, right))
    return tuple(merged)


def _vertex_attainable(target: Vector, donors: Matrix, vertex: int) -> bool:
    point = _column(donors, vertex)
    lower, upper = Rat(0), Rat(1)
    for other in range(3):
        if other == vertex:
            continue
        other_point = _column(donors, other)
        h_first = (other_point[0] - point[0]) * (point[0] - target[0])
        h_second = (other_point[1] - point[1]) * (point[1] - target[1])
        intercept = h_second
        slope = h_first - h_second
        if slope == 0:
            if intercept < 0:
                return False
            continue
        boundary = -intercept / slope
        if slope > 0:
            lower = max(lower, boundary)
        else:
            upper = min(upper, boundary)
    return max(lower, Rat(0)) <= min(upper, Rat(1))


def _deduplicate_branches(branches: Sequence[WeightBranch]) -> Tuple[WeightBranch, ...]:
    unique = {}
    for branch in branches:
        canonical = _canonical_branch(branch.name, branch.start, branch.end)
        key = (canonical.start, canonical.end)
        if key not in unique:
            unique[key] = canonical
    return tuple(sorted(unique.values(), key=lambda branch: (branch.start, branch.end, branch.name)))


def _branch_profile(
    branch: WeightBranch,
    validation_target: Vector,
    validation_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
) -> BranchProfile:
    direction = tuple(b - a for a, b in zip(branch.start, branch.end))
    validation_start = tuple(_dot(row, branch.start) for row in validation_donors)
    validation_direction = tuple(_dot(row, direction) for row in validation_donors)
    residual_start = tuple(target - fitted for target, fitted in zip(validation_target, validation_start))
    count = Rat(len(validation_target))
    alpha = _dot(validation_direction, validation_direction) / count
    beta = _dot(residual_start, validation_direction) / count
    gamma = _dot(residual_start, residual_start) / count
    if alpha == 0:
        optimizer = (Rat(0), Rat(1))
        minimum = gamma
    else:
        optimum_parameter = _clamp(beta / alpha)
        optimizer = (optimum_parameter, optimum_parameter)
        minimum = alpha * optimum_parameter * optimum_parameter - 2 * beta * optimum_parameter + gamma
    effect_intercept = treated_post - _dot(donor_post, branch.start)
    effect_slope = -_dot(donor_post, direction)
    return BranchProfile(branch, alpha, beta, gamma, minimum, optimizer, effect_intercept, effect_slope)


def _sublevel_interval(profile: BranchProfile, threshold: Rat) -> Optional[AlgebraicInterval]:
    if threshold < profile.minimum:
        return None
    if profile.alpha == 0:
        return AlgebraicInterval(QuadraticSurd.make(Rat(0)), QuadraticSurd.make(Rat(1)))
    center = profile.beta / profile.alpha
    unconstrained_minimum = profile.gamma - profile.beta * profile.beta / profile.alpha
    radius_squared = (threshold - unconstrained_minimum) / profile.alpha
    if center <= 0 or center * center <= radius_squared:
        left = QuadraticSurd.make(Rat(0))
    else:
        left = QuadraticSurd.make(center, Rat(-1), radius_squared)
    if center >= 1 or (1 - center) * (1 - center) <= radius_squared:
        right = QuadraticSurd.make(Rat(1))
    else:
        right = QuadraticSurd.make(center, Rat(1), radius_squared)
    return AlgebraicInterval(left, right)


def _surd_enclosure(value: QuadraticSurd, digits: int) -> Tuple[Rat, Rat]:
    if digits < 1:
        raise ValueError("digits must be positive")
    if value.coefficient == 0:
        return value.offset, value.offset
    scale = 10 ** digits
    scaled_square = value.radicand.numerator * scale * scale // value.radicand.denominator
    root_floor = isqrt(scaled_square)
    root_lower = Rat(root_floor, scale)
    root_upper = Rat(root_floor + 1, scale)
    if value.coefficient > 0:
        return value.offset + value.coefficient * root_lower, value.offset + value.coefficient * root_upper
    return value.offset + value.coefficient * root_upper, value.offset + value.coefficient * root_lower


def _minimum_loss_with_effect_sign(profile: BranchProfile, nonnegative: bool) -> Optional[Rat]:
    lower, upper = Rat(0), Rat(1)
    intercept, slope = profile.effect_intercept, profile.effect_slope
    if slope == 0:
        condition = intercept >= 0 if nonnegative else intercept <= 0
        return profile.minimum if condition else None
    boundary = -intercept / slope
    if nonnegative:
        if slope > 0:
            lower = max(lower, boundary)
        else:
            upper = min(upper, boundary)
    else:
        if slope > 0:
            upper = min(upper, boundary)
        else:
            lower = max(lower, boundary)
    lower, upper = max(lower, Rat(0)), min(upper, Rat(1))
    if lower > upper:
        return None
    if profile.alpha == 0:
        return profile.gamma
    parameter = _clamp(profile.beta / profile.alpha, lower, upper)
    return profile.loss(parameter)


def reduce_three_donor_k2_problem(
    predictor_target: Vector,
    predictor_donors: Matrix,
    validation_target: Vector,
    validation_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
) -> ThreeDonorK2Reduction:
    if len(predictor_target) != 2 or len(predictor_donors) != 2 or any(len(row) != 3 for row in predictor_donors):
        raise ValueError("requires K=2 and J=3")
    if not validation_target or len(validation_target) != len(validation_donors) or any(
        len(row) != 3 for row in validation_donors
    ):
        raise ValueError("validation dimensions do not match J=3")
    if len(donor_post) != 3:
        raise ValueError("requires three post-treatment donor summaries")

    augmented = [list(row) for row in predictor_donors]
    augmented.append([Rat(1), Rat(1), Rat(1)])
    barycentric = _solve_square(augmented, tuple(predictor_target) + (Rat(1),))
    if barycentric is None:
        raise ValueError("predictor donor columns must be affinely independent")

    branches: List[WeightBranch] = []
    if all(value >= 0 for value in barycentric):
        branches.append(WeightBranch("interior-match", barycentric, barycentric))

    for first, second in combinations(range(3), 2):
        inactive = next(index for index in range(3) if index not in (first, second))
        for interval_index, (left, right) in enumerate(
            _edge_parameter_intervals(predictor_target, predictor_donors, first, second, inactive)
        ):
            start = [Rat(0), Rat(0), Rat(0)]
            end = [Rat(0), Rat(0), Rat(0)]
            start[first], start[second] = left, 1 - left
            end[first], end[second] = right, 1 - right
            branches.append(WeightBranch(f"edge-{first}-{second}-{interval_index}", tuple(start), tuple(end)))

    for vertex in range(3):
        if _vertex_attainable(predictor_target, predictor_donors, vertex):
            weights = tuple(Rat(int(index == vertex)) for index in range(3))
            branches.append(WeightBranch(f"vertex-{vertex}", weights, weights))

    for predictor_row in range(2):
        donor_values = predictor_donors[predictor_row]
        fitted = _clamp(predictor_target[predictor_row], min(donor_values), max(donor_values))
        vertices = polytope_vertices(
            (tuple(Rat(1) for _ in range(3)), donor_values),
            (Rat(1), fitted),
        )
        if len(vertices) > 2:
            raise AssertionError("affine independence should make a boundary fiber at most one-dimensional")
        branches.append(
            WeightBranch(
                f"predictor-boundary-{predictor_row}",
                vertices[0],
                vertices[-1],
            )
        )

    branch_tuple = _deduplicate_branches(branches)
    if not branch_tuple:
        raise AssertionError("lower-level graph projection is empty")
    profiles = tuple(
        _branch_profile(branch, validation_target, validation_donors, treated_post, donor_post)
        for branch in branch_tuple
    )
    outer_optimum = min(profile.minimum for profile in profiles)

    exact_effects: List[Rat] = []
    for profile in profiles:
        if profile.minimum != outer_optimum:
            continue
        for parameter in profile.optimizer:
            exact_effects.append(profile.effect(parameter))

    negative_losses = [
        value for profile in profiles if (value := _minimum_loss_with_effect_sign(profile, False)) is not None
    ]
    positive_losses = [
        value for profile in profiles if (value := _minimum_loss_with_effect_sign(profile, True)) is not None
    ]
    span_breakdown = (
        max(min(negative_losses), min(positive_losses)) - outer_optimum
        if negative_losses and positive_losses
        else None
    )

    zero_losses: List[Rat] = []
    for profile in profiles:
        if profile.effect_slope == 0:
            if profile.effect_intercept == 0:
                zero_losses.append(profile.minimum)
            continue
        parameter = -profile.effect_intercept / profile.effect_slope
        if 0 <= parameter <= 1:
            zero_losses.append(profile.loss(parameter))
    attained_zero = min(zero_losses) - outer_optimum if zero_losses else None

    return ThreeDonorK2Reduction(
        branch_tuple,
        profiles,
        outer_optimum,
        (min(exact_effects), max(exact_effects)),
        span_breakdown,
        attained_zero,
    )
