import random
import unittest
from fractions import Fraction

from src.general_predictor import (
    branch_contains_weights,
    construct_general_predictor_graph,
    witness_predictor_weights,
)
from src.general_predictor_scalable import (
    certify_inner_attainability_on_predictor_support,
)
from src.scm_oracle import (
    as_matrix,
    as_vector,
    polytope_vertices,
    simplex_qp_vertices,
    squared_loss,
)


class GeneralPredictorGraphTests(unittest.TestCase):
    def setUp(self):
        self.target = as_vector([1, 1, 1])
        self.donors = as_matrix(
            [
                [0, 2, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 2],
            ]
        )

    def test_tetrahedron_face_lattice_and_branch_soundness(self):
        graph = construct_general_predictor_graph(self.target, self.donors)
        self.assertEqual(len(graph.facets), 4)
        self.assertEqual(len(graph.faces), 15)
        self.assertGreater(len(graph.branches), 3)

        checked = 0
        for branch in graph.branches:
            vertices = polytope_vertices(
                branch.polytope.equality_rows,
                branch.polytope.equality_rhs,
            )
            donor_vectors = [
                variables[: graph.donor_count] for variables in vertices
            ]
            if len(donor_vectors) > 1:
                donor_vectors.append(
                    tuple(
                        (left + right) / 2
                        for left, right in zip(donor_vectors[0], donor_vectors[1])
                    )
                )
            for donor_weights in donor_vectors:
                predictor_weights = witness_predictor_weights(
                    branch,
                    self.target,
                    self.donors,
                    donor_weights,
                )
                inner = simplex_qp_vertices(
                    self.target,
                    self.donors,
                    predictor_weights,
                )
                self.assertEqual(
                    squared_loss(
                        self.target,
                        self.donors,
                        predictor_weights,
                        donor_weights,
                    ),
                    inner.value,
                )
                checked += 1
        self.assertGreater(checked, 12)

    def test_exact_grid_inner_vertices_are_covered(self):
        graph = construct_general_predictor_graph(self.target, self.donors)
        for first in range(5):
            for second in range(5 - first):
                third = 4 - first - second
                predictor_weights = tuple(
                    Fraction(value, 4) for value in (first, second, third)
                )
                inner = simplex_qp_vertices(
                    self.target,
                    self.donors,
                    predictor_weights,
                )
                for donor_weights in inner.vertices:
                    self.assertTrue(
                        any(
                            branch_contains_weights(
                                branch,
                                self.target,
                                self.donors,
                                donor_weights,
                            )
                            for branch in graph.branches
                        )
                    )

    def test_declared_predictor_support_reconstructs_exact_inner_certificate(self):
        donor_weights = as_vector([0, "1/3", "1/3", "1/3"])
        certificate = certify_inner_attainability_on_predictor_support(
            self.target,
            self.donors,
            donor_weights,
            (0, 1, 2),
        )
        self.assertEqual(
            certificate.predictor_weights,
            as_vector(["1/3", "1/3", "1/3"]),
        )
        self.assertEqual(certificate.inner.optimizer, donor_weights)

    def test_lower_dimensional_hull_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "full-dimensional"):
            construct_general_predictor_graph(
                as_vector([0, 0, 0]),
                as_matrix(
                    [
                        [0, 1, 0, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                    ]
                ),
            )

    def test_random_k3_graph_soundness_and_grid_coverage(self):
        rng = random.Random(271828)
        accepted = 0
        for trial in range(8):
            donor_count = 4 + trial % 2
            donors = as_matrix(
                [
                    [rng.randint(-3, 3) for _ in range(donor_count)]
                    for _ in range(3)
                ]
            )
            target = as_vector(rng.randint(-3, 3) for _ in range(3))
            try:
                graph = construct_general_predictor_graph(target, donors)
            except ValueError:
                continue
            accepted += 1
            for branch in graph.branches:
                variables = polytope_vertices(
                    branch.polytope.equality_rows,
                    branch.polytope.equality_rhs,
                )[0]
                donor_weights = variables[:donor_count]
                predictor_weights = witness_predictor_weights(
                    branch,
                    target,
                    donors,
                    donor_weights,
                )
                inner = simplex_qp_vertices(target, donors, predictor_weights)
                self.assertEqual(
                    squared_loss(
                        target,
                        donors,
                        predictor_weights,
                        donor_weights,
                    ),
                    inner.value,
                )

            for first in range(3):
                for second in range(3 - first):
                    third = 2 - first - second
                    predictor_weights = tuple(
                        Fraction(value, 2) for value in (first, second, third)
                    )
                    inner = simplex_qp_vertices(target, donors, predictor_weights)
                    for donor_weights in inner.vertices:
                        self.assertTrue(
                            any(
                                branch_contains_weights(
                                    branch,
                                    target,
                                    donors,
                                    donor_weights,
                                )
                                for branch in graph.branches
                            )
                        )
        self.assertGreaterEqual(accepted, 5)


if __name__ == "__main__":
    unittest.main()
