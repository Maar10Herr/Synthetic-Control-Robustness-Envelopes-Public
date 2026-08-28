import json
import unittest
from fractions import Fraction
from pathlib import Path

from src.scm_oracle import (
    as_matrix,
    as_vector,
    certify_lp_support,
    certify_qp_support,
    exact_grid_envelope,
    simplex_qp_vertices,
)


ROOT = Path(__file__).resolve().parents[1]


class ExactOracleTests(unittest.TestCase):
    def load_pathology(self):
        with (ROOT / "instances" / "tiny" / "exact_sign_flip.json").open() as handle:
            return json.load(handle)

    def test_inner_minimizer_polytope_vertices(self):
        data = self.load_pathology()
        solution = simplex_qp_vertices(
            as_vector(data["predictor_target"]),
            as_matrix(data["predictor_donors"]),
            as_vector(data["predictor_weight_grid"][0]),
        )
        self.assertEqual(solution.value, Fraction(0))
        self.assertEqual(
            set(solution.vertices),
            {
                (Fraction(0), Fraction(0), Fraction(1)),
                (Fraction(1, 2), Fraction(1, 2), Fraction(0)),
            },
        )

    def test_exact_kkt_support_certificate_accepts_and_rejects(self):
        data = self.load_pathology()
        target = as_vector(data["predictor_target"])
        donors = as_matrix(data["predictor_donors"])
        row_weights = as_vector(data["predictor_weight_grid"][0])
        certificate = certify_qp_support(
            target,
            donors,
            row_weights,
            (as_vector([1, 1, 1]),),
            as_vector([1]),
            (2,),
        )
        self.assertEqual(certificate.value, Fraction(0))
        self.assertEqual(certificate.optimizer, (Fraction(0), Fraction(0), Fraction(1)))
        self.assertEqual(certificate.equality_rhs, (Fraction(1),))
        with self.assertRaisesRegex(ValueError, "dual feasibility"):
            certify_qp_support(
                target,
                donors,
                row_weights,
                (as_vector([1, 1, 1]),),
                as_vector([1]),
                (0,),
            )

        dependent_on_support = certify_qp_support(
            as_vector([0]),
            as_matrix([[0, 1]]),
            as_vector([1]),
            (as_vector([1, 1]), as_vector([0, 1])),
            as_vector([1, 0]),
            (0,),
        )
        self.assertEqual(
            dependent_on_support.equality_rhs,
            (Fraction(1), Fraction(0)),
        )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            certify_qp_support(
                as_vector([0]),
                as_matrix([[0, 1]]),
                as_vector([-1]),
                (as_vector([1, 1]),),
                as_vector([1]),
                (0,),
            )

    def test_exact_lp_support_certificate_accepts_and_rejects(self):
        certificate = certify_lp_support(
            as_vector([1, 0]),
            (as_vector([1, 1]),),
            as_vector([1]),
            (1,),
        )
        self.assertEqual(certificate.value, Fraction(0))
        self.assertEqual(certificate.optimizer, (Fraction(0), Fraction(1)))
        with self.assertRaisesRegex(ValueError, "dual feasibility"):
            certify_lp_support(
                as_vector([1, 0]),
                (as_vector([1, 1]),),
                as_vector([1]),
                (0,),
            )
        with self.assertRaisesRegex(ValueError, "counts do not match"):
            certify_lp_support(
                as_vector([1, 0]),
                (as_vector([1, 1]),),
                (),
                (1,),
            )

    def test_exact_lp_multiplier_can_retain_rational_free_coordinates(self):
        certificate = certify_lp_support(
            as_vector([0, -1]),
            (as_vector([1, 1]), as_vector([0, 1])),
            as_vector([1, 0]),
            (0,),
            multiplier_binding_positions=(),
            multiplier_near_proposal=as_vector([0, 2]),
        )
        self.assertEqual(certificate.optimizer, as_vector([1, 0]))
        self.assertEqual(certificate.multipliers, as_vector([0, 2]))
        self.assertEqual(certificate.reduced_costs, as_vector([0, 1]))

    def test_exact_bilevel_sign_flip(self):
        data = self.load_pathology()
        envelope = exact_grid_envelope(
            as_vector(data["predictor_target"]),
            as_matrix(data["predictor_donors"]),
            as_vector(data["validation_target"]),
            as_matrix(data["validation_donors"]),
            tuple(as_vector(v) for v in data["predictor_weight_grid"]),
            Fraction(data["treated_post"]),
            as_vector(data["donor_post"]),
        )
        self.assertEqual(envelope.outer_optimum, Fraction(0))
        self.assertEqual(envelope.lower_effect, Fraction(-2))
        self.assertEqual(envelope.upper_effect, Fraction(2))
        self.assertEqual({candidate.donor_weights for candidate in envelope.optimizers}, {
            (Fraction(0), Fraction(0), Fraction(1)),
            (Fraction(1, 2), Fraction(1, 2), Fraction(0)),
        })

    def test_binary_float_rejected(self):
        with self.assertRaises(TypeError):
            as_vector([0.1])

    def test_adversarial_discovery_grid(self):
        path = ROOT / "instances" / "adversarial" / "two_outer_optima_sign_flip.json"
        with path.open() as handle:
            data = json.load(handle)
        envelope = exact_grid_envelope(
            as_vector(data["predictor_target"]),
            as_matrix(data["predictor_donors"]),
            as_vector(data["validation_target"]),
            as_matrix(data["validation_donors"]),
            tuple(as_vector(v) for v in data["predictor_weight_grid"]),
            Fraction(data["treated_post"]),
            as_vector(data["donor_post"]),
        )
        self.assertEqual(envelope.outer_optimum, Fraction(41, 18))
        self.assertEqual(envelope.lower_effect, Fraction(-1, 3))
        self.assertEqual(envelope.upper_effect, Fraction(1, 3))
        self.assertEqual(
            {(candidate.predictor_weights, candidate.donor_weights) for candidate in envelope.optimizers},
            {
                ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(2, 3), Fraction(1, 3))),
                ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1, 3), Fraction(2, 3))),
            },
        )

    def test_near_optimal_fixture_has_unique_grid_optimum(self):
        path = ROOT / "instances" / "adversarial" / "near_optimal_sign_breakdown.json"
        with path.open() as handle:
            data = json.load(handle)
        grid = tuple((Fraction(i, 8), Fraction(8 - i, 8)) for i in range(9))
        envelope = exact_grid_envelope(
            as_vector(data["predictor_target"]),
            as_matrix(data["predictor_donors"]),
            as_vector(data["validation_target"]),
            as_matrix(data["validation_donors"]),
            grid,
            Fraction(data["treated_post"]),
            as_vector(data["donor_post"]),
        )
        item = data["expected"]["unique_global_optimizer"]
        self.assertEqual(envelope.outer_optimum, Fraction(item["outer_loss"]))
        self.assertEqual(envelope.lower_effect, Fraction(-1, 3))
        self.assertEqual(envelope.upper_effect, Fraction(-1, 3))
        self.assertEqual(len(envelope.optimizers), 1)

    def test_predictor_simplex_vertices_are_not_sufficient(self):
        path = ROOT / "instances" / "tiny" / "interior_v_optimum.json"
        with path.open() as handle:
            data = json.load(handle)
        envelope = exact_grid_envelope(
            as_vector(data["predictor_target"]),
            as_matrix(data["predictor_donors"]),
            as_vector(data["validation_target"]),
            as_matrix(data["validation_donors"]),
            tuple(as_vector(v) for v in data["predictor_weight_grid"]),
            Fraction(data["treated_post"]),
            as_vector(data["donor_post"]),
        )
        self.assertEqual(envelope.outer_optimum, Fraction(0))
        self.assertEqual(len(envelope.optimizers), 1)
        self.assertEqual(envelope.optimizers[0].predictor_weights, (Fraction(1, 2), Fraction(1, 2)))
        self.assertEqual(envelope.optimizers[0].donor_weights, (Fraction(1, 2), Fraction(1, 2)))


if __name__ == "__main__":
    unittest.main()
