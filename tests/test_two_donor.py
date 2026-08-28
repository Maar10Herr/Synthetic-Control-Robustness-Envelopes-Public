import json
import random
import unittest
from fractions import Fraction
from pathlib import Path

from src.scm_oracle import as_matrix, as_vector, exact_grid_envelope
from src.two_donor import QuadraticSurd, reduce_two_donor_problem


ROOT = Path(__file__).resolve().parents[1]


class TwoDonorReductionTests(unittest.TestCase):
    def load_interior_fixture(self):
        with (ROOT / "instances" / "tiny" / "interior_v_optimum.json").open() as handle:
            return json.load(handle)

    def make_interior_result(self):
        data = self.load_interior_fixture()
        return reduce_two_donor_problem(
            as_vector(data["predictor_target"]),
            as_matrix(data["predictor_donors"]),
            as_vector(data["validation_target"]),
            as_matrix(data["validation_donors"]),
            Fraction(data["treated_post"]),
            as_vector(data["donor_post"]),
        )

    def test_continuous_v_reduction_recovers_interior_optimum(self):
        result = self.make_interior_result()
        self.assertEqual(result.attainable_t, (Fraction(0), Fraction(1)))
        self.assertEqual(result.outer_optimizer_t, (Fraction(1, 2), Fraction(1, 2)))
        self.assertEqual(result.outer_optimum, Fraction(0))
        self.assertEqual(result.sign_breakdown_epsilon, Fraction(0))

    def test_exact_rational_near_interval(self):
        result = self.make_interior_result()
        interval = result.near_t_interval(Fraction(1, 4))
        self.assertEqual(interval.lower, QuadraticSurd.make(Fraction(1, 4)))
        self.assertEqual(interval.upper, QuadraticSurd.make(Fraction(3, 4)))
        effects = result.near_effect_interval(Fraction(1, 4))
        self.assertEqual(effects.lower, QuadraticSurd.make(Fraction(-1, 2)))
        self.assertEqual(effects.upper, QuadraticSurd.make(Fraction(1, 2)))

    def test_zero_difference_predictor_makes_all_weights_attainable(self):
        result = reduce_two_donor_problem(
            as_vector([3]),
            as_matrix([[2, 2]]),
            as_vector([1]),
            as_matrix([[0, 1]]),
            Fraction(0),
            as_vector([0, 2]),
        )
        self.assertEqual(result.attainable_t, (Fraction(0), Fraction(1)))
        self.assertEqual(result.sign_breakdown_epsilon, Fraction(1))
        effects = result.near_effect_interval(Fraction(1))
        self.assertEqual(effects.lower, QuadraticSurd.make(Fraction(-2)))
        self.assertEqual(effects.upper, QuadraticSurd.make(Fraction(0)))

    def test_irrational_endpoint_is_kept_as_exact_surd(self):
        result = self.make_interior_result()
        interval = result.near_t_interval(Fraction(1, 3))
        self.assertEqual(interval.lower.offset, Fraction(1, 2))
        self.assertEqual(interval.lower.coefficient, Fraction(-1))
        self.assertEqual(interval.lower.radicand, Fraction(1, 12))

    def test_random_reductions_replay_in_general_exact_oracle(self):
        rng = random.Random(8675309)
        for _ in range(50):
            predictor_target = as_vector(rng.randint(-2, 2) for _ in range(3))
            predictor_donors = as_matrix(
                (rng.randint(-2, 2), rng.randint(-2, 2)) for _ in range(3)
            )
            validation_target = as_vector(rng.randint(-2, 2) for _ in range(2))
            validation_donors = as_matrix(
                (rng.randint(-2, 2), rng.randint(-2, 2)) for _ in range(2)
            )
            donor_post = as_vector([rng.randint(-2, 2), rng.randint(-2, 2)])
            result = reduce_two_donor_problem(
                predictor_target,
                predictor_donors,
                validation_target,
                validation_donors,
                Fraction(0),
                donor_post,
            )
            target_t = result.outer_optimizer_t[0]
            differences = tuple(row[0] - row[1] for row in predictor_donors)
            offsets = tuple(x - row[1] for x, row in zip(predictor_target, predictor_donors))

            zero_row = next((i for i, difference in enumerate(differences) if difference == 0), None)
            if zero_row is not None:
                v = tuple(Fraction(int(i == zero_row)) for i in range(3))
            else:
                ratios = tuple(offset / difference for offset, difference in zip(offsets, differences))
                exact_row = next((i for i, ratio in enumerate(ratios) if ratio == target_t), None)
                if exact_row is not None:
                    v = tuple(Fraction(int(i == exact_row)) for i in range(3))
                else:
                    below = [i for i, ratio in enumerate(ratios) if ratio < target_t]
                    above = [i for i, ratio in enumerate(ratios) if ratio > target_t]
                    if below and above:
                        i = max(below, key=lambda index: ratios[index])
                        j = min(above, key=lambda index: ratios[index])
                        theta_i = (ratios[j] - target_t) / (ratios[j] - ratios[i])
                        theta_j = 1 - theta_i
                        raw = [Fraction(0), Fraction(0), Fraction(0)]
                        raw[i] = theta_i / (differences[i] * differences[i])
                        raw[j] = theta_j / (differences[j] * differences[j])
                        total = sum(raw, Fraction(0))
                        v = tuple(value / total for value in raw)
                    else:
                        index = min(range(3), key=lambda i: ratios[i]) if not below else max(
                            range(3), key=lambda i: ratios[i]
                        )
                        v = tuple(Fraction(int(i == index)) for i in range(3))

            replay = exact_grid_envelope(
                predictor_target,
                predictor_donors,
                validation_target,
                validation_donors,
                (v,),
                Fraction(0),
                donor_post,
            )
            self.assertEqual(replay.outer_optimum, result.outer_optimum)


if __name__ == "__main__":
    unittest.main()
