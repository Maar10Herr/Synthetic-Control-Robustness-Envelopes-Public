import sys
import tempfile
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

    def test_duplicate_control_labels_are_rejected(self):
        source = ROOT / "instances" / "empirical" / "prop99_full.csv"
        rows = source.read_text().splitlines()
        first_control = rows[2].split(",", 1)[0]
        second_control = rows[3].split(",", 1)[1]
        rows[3] = first_control + "," + second_control
        with tempfile.TemporaryDirectory(prefix="prop99-loader-") as temporary:
            path = Path(temporary) / "duplicate.csv"
            path.write_text("\n".join(rows) + "\n")
            with self.assertRaisesRegex(ValueError, "unique nonempty"):
                load_prop99_full(path)


if __name__ == "__main__":
    unittest.main()
