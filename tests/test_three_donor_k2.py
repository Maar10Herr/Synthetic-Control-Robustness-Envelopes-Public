import json
import random
import unittest
from fractions import Fraction
from pathlib import Path

from src.scm_oracle import as_matrix, as_vector, exact_grid_envelope, simplex_qp_vertices
from src.three_donor_k2 import WeightBranch, reduce_three_donor_k2_problem


ROOT = Path(__file__).resolve().parents[1]


def _load_fixture(name):
    with (ROOT / "instances" / "adversarial" / f"{name}.json").open() as handle:
        data = json.load(handle)
    return data, reduce_three_donor_k2_problem(
        as_vector(data["predictor_target"]),
        as_matrix(data["predictor_donors"]),
        as_vector(data["validation_target"]),
        as_matrix(data["validation_donors"]),
        Fraction(data["treated_post"]),
        as_vector(data["donor_post"]),
    )


def _on_branch(weights, branch: WeightBranch):
    parameters = []
    for value, start, end in zip(weights, branch.start, branch.end):
        direction = end - start
        if direction == 0:
            if value != start:
                return False
        else:
            parameters.append((value - start) / direction)
    return not parameters or (
        all(value == parameters[0] for value in parameters) and 0 <= parameters[0] <= 1
    )


def _lower_kkt_attainable(weights, target, donors):
    """Independent exact test for existence of some V=(a,1-a)."""

    residual = tuple(
        sum((donors[k][j] * weights[j] for j in range(3)), Fraction(0)) - target[k]
        for k in range(2)
    )
    coordinate_gradients = tuple(
        tuple(donors[k][j] * residual[k] for j in range(3)) for k in range(2)
    )
    active = tuple(j for j, value in enumerate(weights) if value > 0)
    reference = active[0]
    fixed_predictor_weight = None

    for donor in active[1:]:
        intercept = coordinate_gradients[1][donor] - coordinate_gradients[1][reference]
        slope = (
            coordinate_gradients[0][donor]
            - coordinate_gradients[0][reference]
            - intercept
        )
        if slope == 0:
            if intercept != 0:
                return False
            continue
        candidate = -intercept / slope
        if fixed_predictor_weight is not None and candidate != fixed_predictor_weight:
            return False
        fixed_predictor_weight = candidate

    if fixed_predictor_weight is not None:
        if not 0 <= fixed_predictor_weight <= 1:
            return False
        return all(
            fixed_predictor_weight
            * (coordinate_gradients[0][donor] - coordinate_gradients[0][reference])
            + (1 - fixed_predictor_weight)
            * (coordinate_gradients[1][donor] - coordinate_gradients[1][reference])
            >= 0
            for donor in range(3)
            if donor not in active
        )

    lower, upper = Fraction(0), Fraction(1)
    for donor in range(3):
        if donor in active:
            continue
        intercept = coordinate_gradients[1][donor] - coordinate_gradients[1][reference]
        slope = (
            coordinate_gradients[0][donor]
            - coordinate_gradients[0][reference]
            - intercept
        )
        if slope == 0:
            if intercept < 0:
                return False
            continue
        boundary = -intercept / slope
        if slope > 0:
            lower = max(lower, boundary)
        else:
            upper = min(upper, boundary)
    return max(lower, Fraction(0)) <= min(upper, Fraction(1))


class ThreeDonorK2ReductionTests(unittest.TestCase):
    def test_two_global_optima_fixture(self):
        _, result = _load_fixture("two_outer_optima_sign_flip")
        self.assertEqual(result.outer_optimum, Fraction(41, 18))
        self.assertEqual(result.exact_optimum_effects, (Fraction(-1, 3), Fraction(1, 3)))
        self.assertEqual(result.sign_breakdown_epsilon, Fraction(0))
        self.assertEqual(result.attained_zero_epsilon, Fraction(26, 81))
        envelope = result.near_effect_envelope(Fraction(0))
        self.assertEqual(envelope.lower_effect.lower, Fraction(-1, 3))
        self.assertEqual(envelope.lower_effect.upper, Fraction(-1, 3))
        self.assertEqual(envelope.upper_effect.lower, Fraction(1, 3))
        self.assertEqual(envelope.upper_effect.upper, Fraction(1, 3))

    def test_unique_optimum_and_two_distinct_breakdowns(self):
        _, result = _load_fixture("near_optimal_sign_breakdown")
        self.assertEqual(result.outer_optimum, Fraction(41072081, 18000000))
        self.assertEqual(result.exact_optimum_effects, (Fraction(-1, 3), Fraction(-1, 3)))
        self.assertEqual(result.sign_breakdown_epsilon, Fraction(1, 1000))
        self.assertEqual(result.attained_zero_epsilon, Fraction(26081, 81000))
        envelope = result.near_effect_envelope(Fraction(1, 1000))
        self.assertEqual(envelope.lower_effect.lower, Fraction(-1, 3))
        self.assertEqual(envelope.upper_effect.upper, Fraction(1, 3))

    def test_irrational_near_endpoint_has_rigorous_narrow_enclosure(self):
        _, result = _load_fixture("near_optimal_sign_breakdown")
        envelope = result.near_effect_envelope(Fraction(1, 3), digits=15)
        self.assertTrue(
            any(
                interval.upper.coefficient != 0
                for _, interval in envelope.branch_effect_intervals
            )
        )
        self.assertLessEqual(
            envelope.upper_effect.upper - envelope.upper_effect.lower,
            Fraction(1, 10**14),
        )

    def test_affinely_dependent_predictor_donors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "affinely independent"):
            reduce_three_donor_k2_problem(
                as_vector([0, 0]),
                as_matrix([[0, 1, 2], [0, 1, 2]]),
                as_vector([0]),
                as_matrix([[0, 1, 2]]),
                Fraction(0),
                as_vector([0, 0, 0]),
            )

    def test_random_branch_soundness_and_grid_coverage(self):
        rng = random.Random(20260826)
        accepted = 0
        for _ in range(60):
            predictor_donors = as_matrix(
                (rng.randint(-3, 3) for _ in range(3)) for _ in range(2)
            )
            predictor_target = as_vector(rng.randint(-3, 3) for _ in range(2))
            validation_donors = as_matrix(
                (rng.randint(-3, 3) for _ in range(3)) for _ in range(2)
            )
            validation_target = as_vector(rng.randint(-3, 3) for _ in range(2))
            donor_post = as_vector(rng.randint(-3, 3) for _ in range(3))
            try:
                result = reduce_three_donor_k2_problem(
                    predictor_target,
                    predictor_donors,
                    validation_target,
                    validation_donors,
                    Fraction(0),
                    donor_post,
                )
            except ValueError:
                continue
            accepted += 1

            for branch in result.branches:
                for parameter in (Fraction(0), Fraction(1, 2), Fraction(1)):
                    weights = tuple(
                        start + parameter * (end - start)
                        for start, end in zip(branch.start, branch.end)
                    )
                    self.assertTrue(
                        _lower_kkt_attainable(weights, predictor_target, predictor_donors)
                    )

            for numerator in range(5):
                predictor_weights = (Fraction(numerator, 4), Fraction(4 - numerator, 4))
                inner = simplex_qp_vertices(
                    predictor_target,
                    predictor_donors,
                    predictor_weights,
                )
                for weights in inner.vertices:
                    self.assertTrue(any(_on_branch(weights, branch) for branch in result.branches))
                replay = exact_grid_envelope(
                    predictor_target,
                    predictor_donors,
                    validation_target,
                    validation_donors,
                    (predictor_weights,),
                    Fraction(0),
                    donor_post,
                )
                for candidate in replay.optimizers:
                    self.assertTrue(
                        any(_on_branch(candidate.donor_weights, branch) for branch in result.branches)
                    )
        self.assertGreaterEqual(accepted, 50)


if __name__ == "__main__":
    unittest.main()
