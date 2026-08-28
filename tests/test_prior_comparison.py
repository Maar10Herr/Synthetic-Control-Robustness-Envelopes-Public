import importlib.util
import unittest
from fractions import Fraction

from src.prior_comparison import (
    koo_guo_moment_envelope,
    minimum_koo_guo_moment_radius,
    sample_moments,
)
from src.scm_oracle import as_matrix, as_vector


SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(SCIPY_AVAILABLE, "optional scipy discovery dependency is absent")
class PriorComparisonTests(unittest.TestCase):
    def test_moment_class_radius_and_effect_endpoints_are_exact(self):
        target = as_vector([0])
        donors = as_matrix([[0, 1]])
        donor_post = as_vector([1, -1])
        sigma, gamma = sample_moments(target, donors)
        self.assertEqual(sigma, as_matrix([[0, 0], [0, 1]]))
        self.assertEqual(gamma, as_vector([0, 0]))

        feasible = minimum_koo_guo_moment_radius(target, donors)
        zero = minimum_koo_guo_moment_radius(
            target,
            donors,
            donor_post,
            Fraction(0),
        )
        self.assertEqual(feasible.radius, 0)
        self.assertEqual(zero.radius, Fraction(1, 2))

        envelope = koo_guo_moment_envelope(
            target,
            donors,
            Fraction(0),
            donor_post,
            Fraction(1, 2),
        )
        self.assertEqual(envelope.lower_effect, -1)
        self.assertEqual(envelope.upper_effect, 0)
        self.assertEqual(envelope.zero_robust_effect, 0)


if __name__ == "__main__":
    unittest.main()
