import importlib.util
import json
import random
import unittest
from fractions import Fraction
from pathlib import Path

from src.general_predictor import _normal_orthant_witness
from src.general_predictor_scalable import (
    certify_unique_nonnegative_solution,
    decide_coordinate_refinement,
    decide_normal_orthant_intersection,
)
from src.scm_oracle import as_matrix, as_vector
from src.scalable_qp import propose_and_certify_strict_feasible_point
from src.simplex_relaxation import reduce_to_simplex_relaxation
from src.two_predictor import reduce_two_predictor_problem
from src.two_predictor_scalable import reduce_two_predictor_problem_scalable


ROOT = Path(__file__).resolve().parents[1]
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(SCIPY_AVAILABLE, "optional scipy discovery dependency is absent")
class ScalableQPCertificateTests(unittest.TestCase):
    def test_unique_nonnegative_solution_and_coordinate_refinement_decisions(self):
        unique = certify_unique_nonnegative_solution(
            (as_vector([1, 1]), as_vector([0, 1])),
            as_vector([1, 0]),
            as_vector([1, 0]),
        )
        self.assertEqual(unique.support, (0,))
        self.assertEqual(unique.exposure_margins[0], 0)
        self.assertGreaterEqual(unique.exposure_margins[1], 1)

        target = as_vector([0, 0])
        donors = as_matrix([[0, 0], [0, 1]])
        reachable = decide_coordinate_refinement(
            target,
            donors,
            as_vector([1, 0]),
            0,
        )
        unreachable = decide_coordinate_refinement(
            target,
            donors,
            as_vector([0, 1]),
            0,
        )
        self.assertEqual(reachable.secondary_predictor_weights, as_vector([0, 1]))
        self.assertIsNone(reachable.farkas_witness)
        self.assertIsNone(unreachable.secondary_predictor_weights)
        self.assertIsNotNone(unreachable.farkas_witness)

        strict = propose_and_certify_strict_feasible_point(
            (as_vector([1, 1]),),
            as_vector([1]),
            (0, 1),
        )
        self.assertEqual(strict.point, as_vector(["1/2", "1/2"]))
        self.assertEqual(strict.margin, Fraction(1, 2))

    def test_normal_orthant_primal_and_farkas_decisions_match_exact_oracle(self):
        normals = (as_vector([1, 0]), as_vector([0, 1]))
        feasible = decide_normal_orthant_intersection(normals, (1, 1))
        infeasible = decide_normal_orthant_intersection(normals, (-1, -1))
        self.assertIsNotNone(feasible.witness)
        self.assertIsNone(feasible.farkas_witness)
        self.assertIsNone(infeasible.witness)
        self.assertIsNotNone(infeasible.farkas_witness)

        rng = random.Random(57721)
        for _ in range(12):
            candidate = tuple(
                as_vector([rng.randint(-3, 3), rng.randint(-3, 3)])
                for _ in range(3)
            )
            if any(not any(normal) for normal in candidate):
                continue
            for sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                exact = _normal_orthant_witness(candidate, sign)
                decision = decide_normal_orthant_intersection(candidate, sign)
                self.assertEqual(decision.witness is not None, exact is not None)

    def test_simplex_relaxation_has_exact_breakdown(self):
        result = reduce_to_simplex_relaxation(
            as_vector([0]),
            as_matrix([[0, 1]]),
            Fraction(0),
            as_vector([1, -1]),
        )
        self.assertEqual(result.outer_optimum, Fraction(0))
        self.assertEqual(
            result.exact_optimum_effects,
            (Fraction(-1), Fraction(-1)),
        )
        self.assertEqual(result.sign_breakdown_epsilon, Fraction(1, 4))
        self.assertEqual(result.attained_zero_epsilon, Fraction(1, 4))

    def test_public_demo_matches_saved_exact_and_positive_gap_certificates(self):
        with (ROOT / "instances" / "empirical" / "synth_demo_k2.json").open() as handle:
            data = json.load(handle)
        with (ROOT / "certificates" / "synth_demo_k2.json").open() as handle:
            saved = json.load(handle)
        result = reduce_two_predictor_problem_scalable(
            as_vector(data["predictor_target"]),
            as_matrix(data["predictor_donors"]),
            as_vector(data["validation_target"]),
            as_matrix(data["validation_donors"]),
            Fraction(data["treated_post"]),
            as_vector(data["donor_post"]),
        )
        expected = saved["expected"]
        self.assertEqual(result.outer_optimum, Fraction(expected["outer_optimum"]))
        self.assertEqual(
            result.exact_optimum_effects,
            (
                Fraction(expected["exact_lower_effect"]),
                Fraction(expected["exact_upper_effect"]),
            ),
        )
        self.assertEqual(
            result.sign_breakdown_epsilon,
            Fraction(expected["sign_breakdown_epsilon"]),
        )
        near_item = saved["near_envelopes"][0]
        near = result.near_effect_envelope(Fraction(near_item["epsilon"]), digits=6)
        self.assertEqual(
            (near.lower_effect.lower, near.lower_effect.upper),
            tuple(map(Fraction, near_item["lower_effect_box"])),
        )
        self.assertEqual(
            (near.upper_effect.lower, near.upper_effect.upper),
            tuple(map(Fraction, near_item["upper_effect_box"])),
        )

    def test_random_j3_scalable_certificates_match_exhaustive_solver(self):
        rng = random.Random(161803)
        accepted = 0
        for _ in range(12):
            predictor_donors = as_matrix(
                (rng.randint(-3, 3) for _ in range(3)) for _ in range(2)
            )
            predictor_target = as_vector(rng.randint(-3, 3) for _ in range(2))
            validation_donors = as_matrix(
                (rng.randint(-3, 3) for _ in range(3)) for _ in range(2)
            )
            validation_target = as_vector(rng.randint(-3, 3) for _ in range(2))
            donor_post = as_vector(rng.randint(-3, 3) for _ in range(3))
            arguments = (
                predictor_target,
                predictor_donors,
                validation_target,
                validation_donors,
                Fraction(0),
                donor_post,
            )
            try:
                exhaustive = reduce_two_predictor_problem(*arguments)
                scalable = reduce_two_predictor_problem_scalable(*arguments)
            except ValueError:
                continue
            accepted += 1
            self.assertEqual(
                (
                    scalable.outer_optimum,
                    scalable.exact_optimum_effects,
                    scalable.sign_breakdown_epsilon,
                    scalable.attained_zero_epsilon,
                ),
                (
                    exhaustive.outer_optimum,
                    exhaustive.exact_optimum_effects,
                    exhaustive.sign_breakdown_epsilon,
                    exhaustive.attained_zero_epsilon,
                ),
            )
        self.assertGreaterEqual(accepted, 6)


if __name__ == "__main__":
    unittest.main()
