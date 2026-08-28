import json
import random
import unittest
from fractions import Fraction
from pathlib import Path

from src.scm_oracle import as_matrix, as_vector, polytope_vertices, simplex_qp_vertices
from src.three_donor_k2 import reduce_three_donor_k2_problem
from src.two_predictor import GraphPolytope, reduce_two_predictor_problem


ROOT = Path(__file__).resolve().parents[1]


def _fixture(name):
    with (ROOT / "instances" / "adversarial" / f"{name}.json").open() as handle:
        return json.load(handle)


def _in_branch(weights, branch: GraphPolytope):
    fixed_rows = tuple(
        tuple(Fraction(int(index == donor)) for index in range(branch.variable_count))
        for donor in range(branch.donor_count)
    )
    try:
        polytope_vertices(
            branch.equality_rows + fixed_rows,
            branch.equality_rhs + tuple(weights),
        )
        return True
    except ValueError:
        return False


def _lower_kkt_attainable(weights, target, donors):
    donor_count = len(weights)
    residual = tuple(
        sum((donors[k][j] * weights[j] for j in range(donor_count)), Fraction(0))
        - target[k]
        for k in range(2)
    )
    gradients = tuple(
        tuple(donors[k][j] * residual[k] for j in range(donor_count))
        for k in range(2)
    )
    active = tuple(j for j, value in enumerate(weights) if value > 0)
    reference = active[0]
    fixed = None
    for donor in active[1:]:
        intercept = gradients[1][donor] - gradients[1][reference]
        slope = gradients[0][donor] - gradients[0][reference] - intercept
        if slope == 0:
            if intercept != 0:
                return False
            continue
        candidate = -intercept / slope
        if fixed is not None and candidate != fixed:
            return False
        fixed = candidate

    if fixed is not None:
        return 0 <= fixed <= 1 and all(
            fixed * (gradients[0][donor] - gradients[0][reference])
            + (1 - fixed) * (gradients[1][donor] - gradients[1][reference])
            >= 0
            for donor in range(donor_count)
            if donor not in active
        )

    lower, upper = Fraction(0), Fraction(1)
    for donor in range(donor_count):
        if donor in active:
            continue
        intercept = gradients[1][donor] - gradients[1][reference]
        slope = gradients[0][donor] - gradients[0][reference] - intercept
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


class TwoPredictorReductionTests(unittest.TestCase):
    def _solve_fixture(self, name):
        data = _fixture(name)
        return reduce_two_predictor_problem(
            as_vector(data["predictor_target"]),
            as_matrix(data["predictor_donors"]),
            as_vector(data["validation_target"]),
            as_matrix(data["validation_donors"]),
            Fraction(data["treated_post"]),
            as_vector(data["donor_post"]),
        )

    def test_adversarial_fixtures(self):
        tied = self._solve_fixture("two_outer_optima_sign_flip")
        self.assertEqual(tied.outer_optimum, Fraction(41, 18))
        self.assertEqual(tied.exact_optimum_effects, (Fraction(-1, 3), Fraction(1, 3)))
        self.assertEqual(tied.sign_breakdown_epsilon, Fraction(0))
        self.assertEqual(tied.attained_zero_epsilon, Fraction(26, 81))

        perturbed = self._solve_fixture("near_optimal_sign_breakdown")
        self.assertEqual(perturbed.outer_optimum, Fraction(41072081, 18000000))
        self.assertEqual(perturbed.exact_optimum_effects, (Fraction(-1, 3), Fraction(-1, 3)))
        self.assertEqual(perturbed.sign_breakdown_epsilon, Fraction(1, 1000))
        self.assertEqual(perturbed.attained_zero_epsilon, Fraction(26081, 81000))

    def test_duplicate_fourth_donor_preserves_fixture_certificate(self):
        data = _fixture("near_optimal_sign_breakdown")
        predictor_donors = as_matrix(row + [row[2]] for row in data["predictor_donors"])
        validation_donors = as_matrix(row + [row[2]] for row in data["validation_donors"])
        donor_post = as_vector(data["donor_post"] + [data["donor_post"][2]])
        result = reduce_two_predictor_problem(
            as_vector(data["predictor_target"]),
            predictor_donors,
            as_vector(data["validation_target"]),
            validation_donors,
            Fraction(data["treated_post"]),
            donor_post,
        )
        self.assertEqual(result.outer_optimum, Fraction(41072081, 18000000))
        self.assertEqual(result.exact_optimum_effects, (Fraction(-1, 3), Fraction(-1, 3)))
        self.assertEqual(result.sign_breakdown_epsilon, Fraction(1, 1000))
        self.assertEqual(result.attained_zero_epsilon, Fraction(26081, 81000))

        near = result.near_effect_envelope(Fraction(1, 3), digits=6)
        direct_data = _fixture("near_optimal_sign_breakdown")
        direct = reduce_three_donor_k2_problem(
            as_vector(direct_data["predictor_target"]),
            as_matrix(direct_data["predictor_donors"]),
            as_vector(direct_data["validation_target"]),
            as_matrix(direct_data["validation_donors"]),
            Fraction(direct_data["treated_post"]),
            as_vector(direct_data["donor_post"]),
        ).near_effect_envelope(Fraction(1, 3), digits=12)
        self.assertLessEqual(near.upper_effect.lower, direct.upper_effect.upper)
        self.assertLessEqual(direct.upper_effect.lower, near.upper_effect.upper)
        self.assertLessEqual(
            near.upper_effect.upper - near.upper_effect.lower,
            Fraction(1, 10**6),
        )

    def test_positive_gap_bisection_is_a_rigorous_global_certificate(self):
        data = _fixture("near_optimal_sign_breakdown")
        arguments = (
            as_vector(data["predictor_target"]),
            as_matrix(data["predictor_donors"]),
            as_vector(data["validation_target"]),
            as_matrix(data["validation_donors"]),
            Fraction(data["treated_post"]),
            as_vector(data["donor_post"]),
        )
        direct = reduce_three_donor_k2_problem(*arguments)
        lifted = reduce_two_predictor_problem(*arguments)

        exact = lifted.near_effect_envelope(Fraction(0))
        self.assertEqual(exact.lower_effect.lower, Fraction(-1, 3))
        self.assertEqual(exact.upper_effect.upper, Fraction(-1, 3))

        for epsilon in (Fraction(1, 1000), Fraction(1, 3)):
            symbolic = direct.near_effect_envelope(epsilon, digits=12)
            certified = lifted.near_effect_envelope(epsilon, digits=8)
            self.assertLessEqual(
                certified.lower_effect.lower,
                symbolic.lower_effect.upper,
            )
            self.assertLessEqual(
                symbolic.lower_effect.lower,
                certified.lower_effect.upper,
            )
            self.assertLessEqual(
                certified.upper_effect.lower,
                symbolic.upper_effect.upper,
            )
            self.assertLessEqual(
                symbolic.upper_effect.lower,
                certified.upper_effect.upper,
            )
            self.assertLessEqual(
                certified.lower_effect.upper - certified.lower_effect.lower,
                Fraction(1, 10**8),
            )
            self.assertLessEqual(
                certified.upper_effect.upper - certified.upper_effect.lower,
                Fraction(1, 10**8),
            )

        with self.assertRaises(ValueError):
            lifted.near_effect_envelope(Fraction(-1))
        with self.assertRaises(ValueError):
            lifted.near_effect_envelope(Fraction(0), digits=0)

    def test_random_j3_results_match_independent_segment_solver(self):
        rng = random.Random(271828)
        accepted = 0
        for _ in range(25):
            predictor_donors = as_matrix(
                (rng.randint(-3, 3) for _ in range(3)) for _ in range(2)
            )
            predictor_target = as_vector(rng.randint(-3, 3) for _ in range(2))
            validation_donors = as_matrix(
                (rng.randint(-3, 3) for _ in range(3)) for _ in range(1)
            )
            validation_target = as_vector(rng.randint(-3, 3) for _ in range(1))
            donor_post = as_vector(rng.randint(-3, 3) for _ in range(3))
            try:
                direct = reduce_three_donor_k2_problem(
                    predictor_target,
                    predictor_donors,
                    validation_target,
                    validation_donors,
                    Fraction(0),
                    donor_post,
                )
                lifted = reduce_two_predictor_problem(
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
            self.assertEqual(
                (
                    direct.outer_optimum,
                    direct.exact_optimum_effects,
                    direct.sign_breakdown_epsilon,
                    direct.attained_zero_epsilon,
                ),
                (
                    lifted.outer_optimum,
                    lifted.exact_optimum_effects,
                    lifted.sign_breakdown_epsilon,
                    lifted.attained_zero_epsilon,
                ),
            )
        self.assertGreaterEqual(accepted, 18)

    def test_random_j4_graph_soundness_and_grid_coverage(self):
        rng = random.Random(314159)
        accepted = 0
        for _ in range(8):
            donor_count = 4
            predictor_donors = as_matrix(
                (rng.randint(-3, 3) for _ in range(donor_count)) for _ in range(2)
            )
            predictor_target = as_vector(rng.randint(-3, 3) for _ in range(2))
            validation_donors = as_matrix(
                (rng.randint(-3, 3) for _ in range(donor_count)) for _ in range(1)
            )
            validation_target = as_vector(rng.randint(-3, 3) for _ in range(1))
            donor_post = as_vector(rng.randint(-3, 3) for _ in range(donor_count))
            try:
                result = reduce_two_predictor_problem(
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
                vertices = polytope_vertices(branch.equality_rows, branch.equality_rhs)
                barycenter = tuple(
                    sum((vertex[index] for vertex in vertices), Fraction(0)) / len(vertices)
                    for index in range(branch.variable_count)
                )
                for variables in vertices + (barycenter,):
                    self.assertTrue(
                        _lower_kkt_attainable(
                            variables[:donor_count], predictor_target, predictor_donors
                        )
                    )

            for numerator in range(3):
                predictor_weights = (Fraction(numerator, 2), Fraction(2 - numerator, 2))
                inner = simplex_qp_vertices(
                    predictor_target,
                    predictor_donors,
                    predictor_weights,
                )
                for weights in inner.vertices:
                    self.assertTrue(any(_in_branch(weights, branch) for branch in result.branches))
        self.assertGreaterEqual(accepted, 6)

    def test_collinear_predictor_hull_is_explicitly_out_of_scope(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            reduce_two_predictor_problem(
                as_vector([0, 0]),
                as_matrix([[0, 1, 2, 3], [0, 1, 2, 3]]),
                as_vector([0]),
                as_matrix([[0, 1, 2, 3]]),
                Fraction(0),
                as_vector([0, 0, 0, 0]),
            )


if __name__ == "__main__":
    unittest.main()
