"""Exact supportwise polynomial systems for predictor-weight floors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .scm_oracle import Matrix, Rat, Vector, _rank
from .simplex_polynomial import (
    Polynomial,
    polynomial_add,
    polynomial_constant,
    polynomial_determinant,
    polynomial_linear,
    polynomial_multiply,
    polynomial_scale,
)


@dataclass(frozen=True)
class PredictorFloorSupportSystem:
    support: Tuple[int, ...]
    denominator: Polynomial
    weight_numerators: Tuple[Polynomial, ...]
    reduced_gradient_donors: Tuple[int, ...]
    reduced_gradient_numerators: Tuple[Polynomial, ...]
    objective_difference: Polynomial

    @property
    def feasibility_constraints(self) -> Tuple[Polynomial, ...]:
        return self.weight_numerators + self.reduced_gradient_numerators


def treated_sample_variances(target: Vector, donors: Matrix) -> Vector:
    if not target or len(donors) != len(target):
        raise ValueError("predictor dimensions do not match")
    donor_count = len(donors[0])
    if donor_count < 1 or any(len(row) != donor_count for row in donors):
        raise ValueError("predictor donor matrix is empty or ragged")
    sample_size = donor_count + 1
    variances = []
    for treated, row in zip(target, donors):
        values = tuple(row) + (treated,)
        mean = sum(values, Rat(0)) / sample_size
        variance = (
            sum(((value - mean) ** 2 for value in values), Rat(0))
            / (sample_size - 1)
        )
        if variance <= 0:
            raise ValueError("standardized floor requires positive row variance")
        variances.append(variance)
    return tuple(variances)


def floor_metric_polynomials(variances: Vector, delta: Rat) -> Tuple[Polynomial, ...]:
    predictor_count = len(variances)
    if predictor_count < 1 or any(value <= 0 for value in variances):
        raise ValueError("predictor variances must be positive")
    if delta <= 0 or predictor_count * delta >= 1:
        raise ValueError("floor must satisfy 0 < K*delta < 1")
    free_mass = Rat(1) - predictor_count * delta
    result = []
    for predictor, variance in enumerate(variances):
        coefficients = [delta / variance] * predictor_count
        coefficients[predictor] += free_mass / variance
        result.append(polynomial_linear(coefficients))
    return tuple(result)


def _weighted_gram(
    metrics: Sequence[Polynomial],
    left: Vector,
    right: Vector,
) -> Polynomial:
    return polynomial_add(
        *(
            polynomial_scale(metric, left_value * right_value)
            for metric, left_value, right_value in zip(metrics, left, right)
        )
    )


def predictor_floor_support_system(
    predictor_target: Vector,
    predictor_donors: Matrix,
    validation_target: Vector,
    validation_donors: Matrix,
    variances: Vector,
    delta: Rat,
    support: Sequence[int],
    candidate_loss: Rat,
) -> PredictorFloorSupportSystem:
    """Build exact Cramer/KKT polynomials for one declared donor support."""

    predictor_count = len(predictor_target)
    donor_count = len(predictor_donors[0])
    if (
        predictor_count < 1
        or len(predictor_donors) != predictor_count
        or any(len(row) != donor_count for row in predictor_donors)
        or len(variances) != predictor_count
    ):
        raise ValueError("predictor dimensions do not match")
    validation_count = len(validation_target)
    if (
        validation_count < 1
        or len(validation_donors) != validation_count
        or any(len(row) != donor_count for row in validation_donors)
    ):
        raise ValueError("validation dimensions do not match")
    support = tuple(support)
    if (
        not support
        or len(set(support)) != len(support)
        or any(index < 0 or index >= donor_count for index in support)
    ):
        raise ValueError("support is empty, repeated, or out of range")

    metrics = floor_metric_polynomials(variances, delta)
    base = support[0]
    directions = tuple(
        tuple(row[donor] - row[base] for row in predictor_donors)
        for donor in support[1:]
    )
    if directions and _rank(directions) != len(directions):
        raise ValueError("support directions are affinely dependent")
    residual = tuple(
        treated - row[base]
        for treated, row in zip(predictor_target, predictor_donors)
    )
    if directions:
        gram = tuple(
            tuple(_weighted_gram(metrics, left, right) for right in directions)
            for left in directions
        )
        rhs = tuple(_weighted_gram(metrics, direction, residual) for direction in directions)
        denominator = polynomial_determinant(gram, predictor_count)
        if not denominator:
            raise ValueError("support directions are affinely dependent")
        numerators = []
        for column in range(len(directions)):
            replaced = [list(row) for row in gram]
            for row in range(len(directions)):
                replaced[row][column] = rhs[row]
            numerators.append(polynomial_determinant(replaced, predictor_count))
    else:
        denominator = polynomial_constant(Rat(1), predictor_count)
        numerators = []
    weight_numerators = (
        polynomial_add(
            denominator,
            *(polynomial_scale(numerator, Rat(-1)) for numerator in numerators),
        ),
        *numerators,
    )
    residual_numerators = tuple(
        polynomial_add(
            polynomial_scale(denominator, base_residual),
            *(
                polynomial_scale(numerator, -direction[predictor])
                for numerator, direction in zip(numerators, directions)
            ),
        )
        for predictor, base_residual in enumerate(residual)
    )
    inactive = tuple(index for index in range(donor_count) if index not in support)
    reduced = []
    for donor in inactive:
        direction = tuple(
            row[donor] - row[base] for row in predictor_donors
        )
        reduced.append(
            polynomial_add(
                *(
                    polynomial_scale(
                        polynomial_multiply(metric, residual_numerator),
                        -component,
                    )
                    for metric, residual_numerator, component in zip(
                        metrics,
                        residual_numerators,
                        direction,
                    )
                )
            )
        )

    validation_base = tuple(
        treated - row[base]
        for treated, row in zip(validation_target, validation_donors)
    )
    validation_directions = tuple(
        tuple(row[donor] - row[base] for row in validation_donors)
        for donor in support[1:]
    )
    validation_residuals = tuple(
        polynomial_add(
            polynomial_scale(denominator, base_residual),
            *(
                polynomial_scale(numerator, -direction[period])
                for numerator, direction in zip(
                    numerators,
                    validation_directions,
                )
            ),
        )
        for period, base_residual in enumerate(validation_base)
    )
    loss_numerator = polynomial_add(
        *(polynomial_multiply(item, item) for item in validation_residuals)
    )
    objective_difference = polynomial_add(
        loss_numerator,
        polynomial_scale(
            polynomial_multiply(denominator, denominator),
            -validation_count * candidate_loss,
        ),
    )
    return PredictorFloorSupportSystem(
        support,
        denominator,
        tuple(weight_numerators),
        inactive,
        tuple(reduced),
        objective_difference,
    )


def polynomial_at_simplex_vertex(polynomial: Polynomial, vertex: int) -> Rat:
    if not polynomial:
        return Rat(0)
    variable_count = len(next(iter(polynomial)))
    if vertex < 0 or vertex >= variable_count:
        raise ValueError("simplex vertex is out of range")
    return sum(
        (
            coefficient
            for exponent, coefficient in polynomial.items()
            if all(
                index == vertex or power == 0
                for index, power in enumerate(exponent)
            )
        ),
        Rat(0),
    )


def support_weights_at_simplex_vertex(
    system: PredictorFloorSupportSystem,
    vertex: int,
) -> Vector:
    denominator = polynomial_at_simplex_vertex(system.denominator, vertex)
    if denominator == 0:
        raise ValueError("support denominator vanishes at the vertex")
    weights = tuple(
        polynomial_at_simplex_vertex(numerator, vertex) / denominator
        for numerator in system.weight_numerators
    )
    if sum(weights, Rat(0)) != 1:
        raise AssertionError("Cramer support weights do not sum to one")
    return weights
