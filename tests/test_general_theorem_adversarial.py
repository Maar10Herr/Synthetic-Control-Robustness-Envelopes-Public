import unittest
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Tuple

from src.general_theorem_audit import (
    direct_predictor_weight_witness,
    normal_projection_witness,
)
from src.scm_oracle import (
    Matrix,
    Vector,
    _dot,
    as_matrix,
    as_vector,
    simplex_qp_vertices,
    squared_loss,
)


F = Fraction


@dataclass(frozen=True)
class Case:
    name: str
    target: Vector
    donors: Matrix
    weights: Vector
    attainable: bool


def case(name, target, donors, weights, attainable=True):
    return Case(
        name,
        as_vector(target),
        as_matrix(donors),
        as_vector(weights),
        attainable,
    )


CASES: Tuple[Case, ...] = (
    case("K=1, one donor, full-face normal", [2], [[0]], [1]),
    case("K=1, treated inside interval, exact fit", ["1/2"], [[0, 1]], ["1/2", "1/2"]),
    case("K=1, treated outside interval, boundary optimum", [2], [[0, 1]], [0, 1]),
    case("K=1, declared nonoptimum", [2], [[0, 1]], [1, 0], False),
    case(
        "lower-dimensional hull, relative-interior full-face normal",
        [1, 0],
        [[0, 1], [0, 1]],
        ["1/2", "1/2"],
    ),
    case(
        "duplicate donors, nonunique representation A",
        [1, -1],
        [[0, 0, 1], [0, 0, 1]],
        [1, 0, 0],
    ),
    case(
        "duplicate donors, nonunique representation B",
        [1, -1],
        [[0, 0, 1], [0, 0, 1]],
        [0, 1, 0],
    ),
    case(
        "constant predictor coordinate",
        [2, 1],
        [[0, 1], [0, 0]],
        [0, 1],
    ),
    case(
        "treated point inside full-dimensional hull",
        ["1/2", "1/2"],
        [[0, 1, 0, 1], [0, 0, 1, 1]],
        ["1/2", 0, 0, "1/2"],
    ),
    case(
        "treated point on hull boundary",
        ["1/2", "1/2"],
        [[0, 1, 0], [0, 0, 1]],
        [0, "1/2", "1/2"],
    ),
    case(
        "treated point outside hull",
        [2, 2],
        [[0, 1, 0], [0, 0, 1]],
        [0, "1/2", "1/2"],
    ),
    case(
        "one zero residual",
        ["1/2", 2],
        [[0, 1, 0], [0, 0, 1]],
        ["1/2", "1/2", 0],
    ),
    case(
        "several zero residuals",
        ["1/2", "1/4", 2],
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        ["1/4", "1/2", "1/4", 0],
    ),
    case(
        "all predictor residuals zero and affine dependence",
        ["1/2", "1/2"],
        [[0, 1, 0, 1], [0, 0, 1, 1]],
        [0, "1/2", "1/2", 0],
    ),
    case(
        "simplex-boundary predictor weight route",
        ["1/2", 2],
        [[0, 1, 0], [0, 0, 1]],
        ["1/2", "1/2", 0],
    ),
    case(
        "multiple compatible faces at face intersection",
        [2, 2],
        [[0, 1, 0, 1], [0, 0, 1, 1]],
        [0, 0, 0, 1],
    ),
    case(
        "nonattainable interior point without coordinate match",
        [2, 3],
        [[0, 1, 0], [0, 0, 1]],
        ["1/3", "1/3", "1/3"],
        False,
    ),
)


class GeneralTheoremAdversarialTests(unittest.TestCase):
    def test_face_union_matches_direct_lower_optimality_on_exact_cases(self):
        for item in CASES:
            with self.subTest(item.name):
                direct = direct_predictor_weight_witness(
                    item.target, item.donors, item.weights
                )
                theorem = normal_projection_witness(
                    item.target, item.donors, item.weights
                )
                self.assertEqual(direct is not None, item.attainable)
                self.assertEqual(theorem is not None, item.attainable)
                self.assertEqual(theorem is not None, direct is not None)
                for witness in (direct, theorem):
                    if witness is None:
                        continue
                    inner = simplex_qp_vertices(
                        item.target, item.donors, witness.predictor_weights
                    )
                    self.assertEqual(
                        squared_loss(
                            item.target,
                            item.donors,
                            witness.predictor_weights,
                            item.weights,
                        ),
                        inner.value,
                    )

    def test_exact_donor_grid_crosscheck_on_every_adversarial_design(self):
        seen = set()
        for item in CASES:
            design = (item.target, item.donors)
            if design in seen:
                continue
            seen.add(design)
            donor_count = len(item.weights)
            for numerators in product(range(3), repeat=donor_count):
                if sum(numerators) != 2:
                    continue
                weights = tuple(F(value, 2) for value in numerators)
                with self.subTest(item.name, weights=weights):
                    direct = direct_predictor_weight_witness(
                        item.target, item.donors, weights
                    )
                    theorem = normal_projection_witness(
                        item.target, item.donors, weights
                    )
                    self.assertEqual(theorem is not None, direct is not None)

    def test_lower_dimensional_case_requires_full_hull_face(self):
        item = next(case for case in CASES if case.name.startswith("lower-dimensional"))
        witness = normal_projection_witness(item.target, item.donors, item.weights)
        self.assertIsNotNone(witness)
        assert witness is not None
        self.assertEqual(witness.route, "normal-face")
        self.assertNotEqual(witness.normal, (F(0), F(0)))
        donor_points = tuple(zip(*item.donors))
        self.assertEqual(
            {_dot(witness.normal, point) for point in donor_points},
            {F(0)},
        )

    def test_duplicate_donors_preserve_entire_optimal_fiber(self):
        first, second = CASES[5], CASES[6]
        self.assertEqual(
            tuple(sum(row[j] * first.weights[j] for j in range(3)) for row in first.donors),
            tuple(sum(row[j] * second.weights[j] for j in range(3)) for row in second.donors),
        )
        for item in (first, second):
            self.assertIsNotNone(
                direct_predictor_weight_witness(item.target, item.donors, item.weights)
            )

    def test_singular_validation_qp_has_rational_optimal_face(self):
        solution = simplex_qp_vertices(
            as_vector([0]),
            as_matrix([[0, 0, 1]]),
            as_vector([1]),
        )
        self.assertEqual(solution.value, F(0))
        self.assertEqual(
            set(solution.vertices),
            {as_vector([1, 0, 0]), as_vector([0, 1, 0])},
        )
        fitted = {
            sum(F(value) * weight for value, weight in zip((0, 0, 1), vertex))
            for vertex in solution.vertices
        }
        self.assertEqual(fitted, {F(0)})
        effects = {
            sum(F(value) * weight for value, weight in zip((2, -1, 7), vertex))
            for vertex in solution.vertices
        }
        self.assertEqual(effects, {F(-1), F(2)})


if __name__ == "__main__":
    unittest.main()
