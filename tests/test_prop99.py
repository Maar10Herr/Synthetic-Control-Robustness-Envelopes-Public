import sys
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prop99 import load_prop99_full  # noqa: E402


class Prop99LoaderTests(unittest.TestCase):
    def test_full_fixture_schema_and_exact_values(self):
        design = load_prop99_full(
            ROOT / "instances" / "empirical" / "prop99_full.csv"
        )
        self.assertEqual(design.treated_label, "California")
        self.assertEqual(len(design.donor_labels), 38)
        self.assertEqual(len(design.predictor_target), 7)
        self.assertEqual(len(design.validation_target), 19)
        self.assertEqual(design.validation_names[5], "cigsale_1975")
        self.assertEqual(design.predictor_names[4], "cigsale_1975")
        self.assertEqual(design.predictor_target[4], design.validation_target[5])
        self.assertEqual(design.treated_post, Fraction(1207, 20))


if __name__ == "__main__":
    unittest.main()
