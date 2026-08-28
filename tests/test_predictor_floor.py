import sys
import unittest
from fractions import Fraction
from functools import reduce
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictor_floor import (  # noqa: E402
    floor_metric_polynomials,
    predictor_floor_support_system,
    support_weights_at_simplex_vertex,
    treated_sample_variances,
)
from src.scm_oracle import as_matrix, as_vector, simplex_qp_vertices  # noqa: E402
from src.simplex_polynomial import (  # noqa: E402
    SimplexCertificateLeaf,
    certify_semialgebraic_nonnegative,
    polynomial_add,
    polynomial_degree,
    polynomial_linear,
    polynomial_multiply,
    polynomial_scale,
    verify_semialgebraic_certificate,
)


F = Fraction


def evaluate(polynomial, point):
    return sum(
        coefficient
        * reduce(
            lambda left, item: left * point[item[0]] ** item[1],
            enumerate(exponent),
            F(1),
        )
        for exponent, coefficient in polynomial.items()
    )


class SimplexPolynomialTests(unittest.TestCase):
    def test_rational_subdivision_proves_a_square(self):
        z0 = polynomial_linear((F(1), F(0)))
        z1 = polynomial_linear((F(0), F(1)))
        difference = polynomial_add(z0, polynomial_scale(z1, F(-1)))
        square = polynomial_multiply(difference, difference)
        certificate = certify_semialgebraic_nonnegative(square)
        self.assertTrue(certificate.complete)
        self.assertEqual(certificate.max_depth, 1)
        self.assertEqual(
            {leaf.disposition for leaf in certificate.leaves},
            {"objective-nonnegative"},
        )
        verify_semialgebraic_certificate(square, (), certificate)

    def test_tampered_leaf_is_rejected(self):
        z0 = polynomial_linear((F(1), F(0)))
        z1 = polynomial_linear((F(0), F(1)))
        negative_cross = polynomial_add(
            polynomial_multiply(z0, z0),
            polynomial_scale(polynomial_multiply(z0, z1), F(-1)),
        )
        bad = type(certify_semialgebraic_nonnegative(z0))(
            2,
            (SimplexCertificateLeaf((), "objective-nonnegative"),),
            True,
            (),
        )
        with self.assertRaises(ValueError):
            verify_semialgebraic_certificate(negative_cross, (), bad)

    def test_missing_zero_bernstein_terms_prevent_false_strict_pruning(self):
        z0 = polynomial_linear((F(1), F(0)))
        z1 = polynomial_linear((F(0), F(1)))
        boundary_feasible_constraint = polynomial_scale(
            polynomial_multiply(z0, z1), F(-1)
        )
        negative_objective = polynomial_scale(z0, F(-1))
        certificate = certify_semialgebraic_nonnegative(
            negative_objective,
            (boundary_feasible_constraint,),
            max_depth=0,
        )
        self.assertFalse(certificate.complete)
        self.assertEqual(certificate.unresolved_paths, ((),))

        forged = type(certificate)(
            2,
            (SimplexCertificateLeaf((), "constraint-negative", 0),),
            True,
            (),
        )
        with self.assertRaisesRegex(ValueError, "not strictly infeasible"):
            verify_semialgebraic_certificate(
                negative_objective,
                (boundary_feasible_constraint,),
                forged,
            )

    def test_incomplete_sparse_leaf_cover_is_rejected(self):
        objective = polynomial_linear((F(1), F(1)))
        incomplete = type(certify_semialgebraic_nonnegative(objective))(
            2,
            (SimplexCertificateLeaf((0, 0), "objective-nonnegative"),),
            True,
            (),
        )
        with self.assertRaisesRegex(ValueError, "do not partition"):
            verify_semialgebraic_certificate(objective, (), incomplete)

    def test_prefix_conflict_is_rejected(self):
        objective = polynomial_linear((F(1), F(1)))
        conflict = type(certify_semialgebraic_nonnegative(objective))(
            2,
            (
                SimplexCertificateLeaf((0,), "objective-nonnegative"),
                SimplexCertificateLeaf((0, 0), "objective-nonnegative"),
                SimplexCertificateLeaf((1,), "objective-nonnegative"),
            ),
            True,
            (),
        )
        with self.assertRaisesRegex(ValueError, "do not partition"):
            verify_semialgebraic_certificate(objective, (), conflict)

    def test_two_donor_floor_system_has_complete_exact_cover(self):
        predictor_target = (F(1), F(0))
        predictor_donors = ((F(0), F(1)), (F(0), F(1)))
        validation_target = (F(0),)
        validation_donors = ((F(0), F(1)),)
        variances = treated_sample_variances(predictor_target, predictor_donors)
        dispositions = set()
        for support in ((0,), (1,), (0, 1)):
            system = predictor_floor_support_system(
                predictor_target,
                predictor_donors,
                validation_target,
                validation_donors,
                variances,
                F(1, 4),
                support,
                F(1, 16),
            )
            certificate = certify_semialgebraic_nonnegative(
                system.objective_difference,
                system.feasibility_constraints,
            )
            self.assertTrue(certificate.complete)
            dispositions.update(leaf.disposition for leaf in certificate.leaves)
            verify_semialgebraic_certificate(
                system.objective_difference,
                system.feasibility_constraints,
                certificate,
            )
        self.assertEqual(
            dispositions,
            {"objective-nonnegative", "constraint-negative"},
        )

    def test_cramer_polynomials_match_fixed_weight_qp_and_declared_degrees(self):
        predictor_target = as_vector([1, 0])
        predictor_donors = as_matrix([[0, 1], [0, 1]])
        validation_target = as_vector([0])
        validation_donors = as_matrix([[0, 1]])
        variances = treated_sample_variances(predictor_target, predictor_donors)
        delta = F(1, 4)
        system = predictor_floor_support_system(
            predictor_target,
            predictor_donors,
            validation_target,
            validation_donors,
            variances,
            delta,
            (0, 1),
            F(1, 16),
        )
        z = (F(1, 3), F(2, 3))
        denominator = evaluate(system.denominator, z)
        weights = tuple(
            evaluate(numerator, z) / denominator
            for numerator in system.weight_numerators
        )
        standardized = tuple(
            delta + (1 - 2 * delta) * coordinate for coordinate in z
        )
        metric = tuple(
            weight / variance for weight, variance in zip(standardized, variances)
        )
        inner = simplex_qp_vertices(predictor_target, predictor_donors, metric)
        self.assertIn(weights, inner.vertices)
        loss = sum(
            (
                validation_target[t]
                - sum(validation_donors[t][j] * weights[j] for j in range(2))
            )
            ** 2
            for t in range(1)
        )
        self.assertEqual(
            evaluate(system.objective_difference, z),
            denominator**2 * (loss - F(1, 16)),
        )
        self.assertEqual(polynomial_degree(system.denominator), 1)
        self.assertEqual(
            {polynomial_degree(item) for item in system.weight_numerators},
            {1},
        )
        self.assertEqual(polynomial_degree(system.objective_difference), 2)
        for polynomial in (
            system.denominator,
            *system.weight_numerators,
            system.objective_difference,
        ):
            degree = polynomial_degree(polynomial)
            self.assertEqual({sum(exponent) for exponent in polynomial}, {degree})

    def test_incomplete_support_cover_cannot_certify_global_bound(self):
        target = as_vector([1, 0])
        donors = as_matrix([[0, 1], [0, 1]])
        variances = treated_sample_variances(target, donors)
        common = (
            target,
            donors,
            as_vector([0]),
            as_matrix([[0, 1]]),
            variances,
            F(1, 4),
        )
        for support in ((0,), (1,)):
            system = predictor_floor_support_system(*common, support, F(2))
            certificate = certify_semialgebraic_nonnegative(
                system.objective_difference,
                system.feasibility_constraints,
            )
            self.assertTrue(certificate.complete)
            self.assertEqual(
                {leaf.disposition for leaf in certificate.leaves},
                {"constraint-negative"},
            )
        full = predictor_floor_support_system(*common, (0, 1), F(2))
        z = (F(0), F(1))
        self.assertLess(evaluate(full.objective_difference, z), 0)

    def test_inactive_gradient_sign_rejects_false_singleton(self):
        target = as_vector([1, 0])
        donors = as_matrix([[0, 1], [0, 1]])
        variances = treated_sample_variances(target, donors)
        system = predictor_floor_support_system(
            target,
            donors,
            as_vector([0]),
            as_matrix([[0, 1]]),
            variances,
            F(1, 4),
            (0,),
            F(1, 16),
        )
        self.assertEqual(system.reduced_gradient_donors, (1,))
        reduced = system.reduced_gradient_numerators[0]
        for z in ((F(1), F(0)), (F(0), F(1)), (F(1, 2), F(1, 2))):
            self.assertLess(evaluate(reduced, z), 0)

    def test_geometric_simplex_with_interior_donor_has_nonunique_fiber(self):
        target = as_vector(["1/2", 1])
        donors = as_matrix([[0, 1, "1/2"], [0, 0, 0]])
        inner = simplex_qp_vertices(target, donors, as_vector([1, 1]))
        endpoint_mix = as_vector(["1/2", "1/2", 0])
        interior_donor = as_vector([0, 0, 1])
        self.assertEqual(set(inner.vertices), {endpoint_mix, interior_donor})
        validation = as_vector([0, 0, 1])
        self.assertEqual(
            {
                sum(value * weight for value, weight in zip(validation, item))
                for item in inner.vertices
            },
            {F(0), F(1)},
        )
        with self.assertRaisesRegex(ValueError, "affinely dependent"):
            predictor_floor_support_system(
                target,
                donors,
                as_vector([0]),
                as_matrix([[0, 0, 1]]),
                as_vector([1, 1]),
                F(1, 4),
                (0, 1, 2),
                F(0),
            )

    def test_duplicate_donors_are_outside_certificate_scope(self):
        target = as_vector([0, 1])
        donors = as_matrix([[0, 0], [0, 0]])
        inner = simplex_qp_vertices(target, donors, as_vector([1, 1]))
        self.assertEqual(
            set(inner.vertices),
            {as_vector([1, 0]), as_vector([0, 1])},
        )
        with self.assertRaisesRegex(ValueError, "affinely dependent"):
            predictor_floor_support_system(
                target,
                donors,
                as_vector([0]),
                as_matrix([[0, 1]]),
                as_vector([1, 1]),
                F(1, 4),
                (0, 1),
                F(0),
            )

    def test_zero_support_weight_is_covered_by_smaller_enumerated_support(self):
        target = as_vector([0, 1])
        donors = as_matrix([[0, 1], [0, 0]])
        common = (
            target,
            donors,
            as_vector([0]),
            as_matrix([[0, 1]]),
            as_vector([1, 1]),
            F(1, 4),
        )
        full = predictor_floor_support_system(*common, (0, 1), F(0))
        singleton = predictor_floor_support_system(*common, (0,), F(0))
        for vertex in (0, 1):
            self.assertEqual(
                support_weights_at_simplex_vertex(full, vertex),
                as_vector([1, 0]),
            )
            self.assertEqual(
                support_weights_at_simplex_vertex(singleton, vertex),
                as_vector([1]),
            )

    def test_metric_floor_requires_positive_scales_and_noncollapsed_domain(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            floor_metric_polynomials(as_vector([1, 0]), F(1, 4))
        for delta in (F(0), F(1, 2), F(3, 4)):
            with self.assertRaisesRegex(ValueError, "0 < K\*delta < 1"):
                floor_metric_polynomials(as_vector([1, 1]), delta)

    def test_floor_semantics_depend_on_declared_standardization(self):
        target = as_vector([1, 0])
        donors = as_matrix([[0, 1], [0, 1]])
        validation_target = as_vector([0])
        validation_donors = as_matrix([[0, 1]])
        z = (F(1, 2), F(1, 2))
        systems = tuple(
            predictor_floor_support_system(
                target,
                donors,
                validation_target,
                validation_donors,
                scales,
                F(1, 4),
                (0, 1),
                F(0),
            )
            for scales in (as_vector([1, 1]), as_vector([4, 1]))
        )
        fitted_weights = []
        for system in systems:
            denominator = evaluate(system.denominator, z)
            fitted_weights.append(
                tuple(
                    evaluate(item, z) / denominator
                    for item in system.weight_numerators
                )
            )
        self.assertEqual(fitted_weights[0], as_vector(["1/2", "1/2"]))
        self.assertEqual(fitted_weights[1], as_vector(["4/5", "1/5"]))


if __name__ == "__main__":
    unittest.main()
