"""Exact continuous-predictor-weight reduction for classical SCM with J=2."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Optional, Tuple

from .scm_oracle import Matrix, Rat, Vector, _dot


def _clamp(value: Rat, lower: Rat, upper: Rat) -> Rat:
    return min(max(value, lower), upper)


def _exact_sqrt(value: Rat) -> Optional[Rat]:
    if value < 0:
        raise ValueError("negative radicand")
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
        return Rat(numerator, denominator)
    return None


@dataclass(frozen=True)
class QuadraticSurd:
    """Exact value ``offset + coefficient * sqrt(radicand)``."""

    offset: Rat
    coefficient: Rat = Rat(0)
    radicand: Rat = Rat(0)

    @classmethod
    def make(cls, offset: Rat, coefficient: Rat = Rat(0), radicand: Rat = Rat(0)) -> "QuadraticSurd":
        if coefficient == 0 or radicand == 0:
            return cls(offset)
        root = _exact_sqrt(radicand)
        if root is not None:
            return cls(offset + coefficient * root)
        return cls(offset, coefficient, radicand)

    def affine(self, intercept: Rat, slope: Rat) -> "QuadraticSurd":
        return QuadraticSurd.make(intercept + slope * self.offset, slope * self.coefficient, self.radicand)

    def approximate(self) -> float:
        return float(self.offset) + float(self.coefficient) * float(self.radicand) ** 0.5

    def __str__(self) -> str:
        if self.coefficient == 0:
            return str(self.offset)
        sign = "+" if self.coefficient > 0 else "-"
        return f"{self.offset} {sign} {abs(self.coefficient)}*sqrt({self.radicand})"


@dataclass(frozen=True)
class AlgebraicInterval:
    lower: QuadraticSurd
    upper: QuadraticSurd


@dataclass(frozen=True)
class TwoDonorReduction:
    attainable_t: Tuple[Rat, Rat]
    outer_alpha: Rat
    outer_beta: Rat
    outer_gamma: Rat
    outer_optimum: Rat
    outer_optimizer_t: Tuple[Rat, Rat]
    effect_intercept: Rat
    effect_t_coefficient: Rat
    sign_breakdown_epsilon: Optional[Rat]

    def outer_loss(self, t: Rat) -> Rat:
        return self.outer_alpha * t * t - 2 * self.outer_beta * t + self.outer_gamma

    def near_t_interval(self, epsilon: Rat) -> AlgebraicInterval:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        lower, upper = self.attainable_t
        if self.outer_alpha == 0:
            return AlgebraicInterval(QuadraticSurd.make(lower), QuadraticSurd.make(upper))

        center = self.outer_beta / self.outer_alpha
        unconstrained_minimum = self.outer_gamma - self.outer_beta * self.outer_beta / self.outer_alpha
        radius_squared = (self.outer_optimum + epsilon - unconstrained_minimum) / self.outer_alpha
        if radius_squared < 0:
            raise AssertionError("nonempty sublevel set produced a negative radius")

        if center <= lower or (center - lower) * (center - lower) <= radius_squared:
            left = QuadraticSurd.make(lower)
        else:
            left = QuadraticSurd.make(center, Rat(-1), radius_squared)
        if center >= upper or (upper - center) * (upper - center) <= radius_squared:
            right = QuadraticSurd.make(upper)
        else:
            right = QuadraticSurd.make(center, Rat(1), radius_squared)
        return AlgebraicInterval(left, right)

    def near_effect_interval(self, epsilon: Rat) -> AlgebraicInterval:
        t_interval = self.near_t_interval(epsilon)
        slope = self.effect_t_coefficient
        if slope >= 0:
            lower_t, upper_t = t_interval.lower, t_interval.upper
        else:
            lower_t, upper_t = t_interval.upper, t_interval.lower
        return AlgebraicInterval(
            lower_t.affine(self.effect_intercept, slope),
            upper_t.affine(self.effect_intercept, slope),
        )


def reduce_two_donor_problem(
    predictor_target: Vector,
    predictor_donors: Matrix,
    validation_target: Vector,
    validation_donors: Matrix,
    treated_post: Rat,
    donor_post: Vector,
) -> TwoDonorReduction:
    if not predictor_donors or any(len(row) != 2 for row in predictor_donors):
        raise ValueError("exactly two predictor-donor columns are required")
    if not validation_donors or any(len(row) != 2 for row in validation_donors):
        raise ValueError("exactly two validation-donor columns are required")
    if len(predictor_target) != len(predictor_donors):
        raise ValueError("predictor dimensions do not match")
    if len(validation_target) != len(validation_donors):
        raise ValueError("validation dimensions do not match")
    if len(donor_post) != 2:
        raise ValueError("exactly two post-treatment donor summaries are required")

    differences = tuple(row[0] - row[1] for row in predictor_donors)
    offsets = tuple(target - row[1] for target, row in zip(predictor_target, predictor_donors))
    if any(difference == 0 for difference in differences):
        attainable = (Rat(0), Rat(1))
    else:
        ratios = tuple(offset / difference for offset, difference in zip(offsets, differences))
        attainable = (_clamp(min(ratios), Rat(0), Rat(1)), _clamp(max(ratios), Rat(0), Rat(1)))

    count = Rat(len(validation_target))
    validation_offsets = tuple(target - row[1] for target, row in zip(validation_target, validation_donors))
    validation_differences = tuple(row[0] - row[1] for row in validation_donors)
    alpha = _dot(validation_differences, validation_differences) / count
    beta = _dot(validation_offsets, validation_differences) / count
    gamma = _dot(validation_offsets, validation_offsets) / count
    lower, upper = attainable
    if alpha == 0:
        optimizer_t = attainable
        optimum = gamma
    else:
        unique_t = _clamp(beta / alpha, lower, upper)
        optimizer_t = (unique_t, unique_t)
        optimum = alpha * unique_t * unique_t - 2 * beta * unique_t + gamma

    effect_intercept = treated_post - donor_post[1]
    effect_slope = -(donor_post[0] - donor_post[1])
    if effect_slope == 0:
        breakdown = Rat(0) if effect_intercept == 0 else None
    else:
        zero_t = -effect_intercept / effect_slope
        breakdown = (
            alpha * zero_t * zero_t - 2 * beta * zero_t + gamma - optimum
            if lower <= zero_t <= upper
            else None
        )
        if breakdown is not None and breakdown < 0:
            raise AssertionError("computed negative sign-breakdown gap")

    return TwoDonorReduction(
        attainable,
        alpha,
        beta,
        gamma,
        optimum,
        optimizer_t,
        effect_intercept,
        effect_slope,
        breakdown,
    )

