#!/usr/bin/env python3
"""Replay the public Synth demo slice's exact and endpoint-box certificates."""

import json
import sys
from fractions import Fraction
from pathlib import Path

if not __debug__:
    raise SystemExit("verification requires assertions; run Python without -O")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scm_oracle import as_matrix, as_vector  # noqa: E402
from src.two_predictor import reduce_two_predictor_problem  # noqa: E402


def _fraction(value):
    return Fraction(value)


def main():
    certificate_path = ROOT / "certificates" / "synth_demo_k2.json"
    with certificate_path.open() as handle:
        certificate = json.load(handle)
    with (ROOT / certificate["instance"]).open() as handle:
        data = json.load(handle)

    result = reduce_two_predictor_problem(
        as_vector(data["predictor_target"]),
        as_matrix(data["predictor_donors"]),
        as_vector(data["validation_target"]),
        as_matrix(data["validation_donors"]),
        _fraction(data["treated_post"]),
        as_vector(data["donor_post"]),
    )
    expected = certificate["expected"]
    assert len(result.hull) == expected["hull_vertex_count"]
    assert len(result.branches) == expected["graph_branch_count"]
    assert result.outer_optimum == _fraction(expected["outer_optimum"])
    assert result.exact_optimum_effects == (
        _fraction(expected["exact_lower_effect"]),
        _fraction(expected["exact_upper_effect"]),
    )
    assert result.attainable_effects == (
        _fraction(expected["attainable_lower_effect"]),
        _fraction(expected["attainable_upper_effect"]),
    )
    assert result.sign_breakdown_epsilon == _fraction(expected["sign_breakdown_epsilon"])
    assert result.attained_zero_epsilon == _fraction(expected["attained_zero_epsilon"])
    assert (
        result.sign_breakdown_epsilon / result.outer_optimum
        == _fraction(expected["relative_sign_breakdown"])
    )

    for item in certificate["near_envelopes"]:
        epsilon = _fraction(item["epsilon"])
        threshold = result.outer_optimum + epsilon
        width = Fraction(1, 10 ** item["digits"])
        lower_lo, lower_hi = map(_fraction, item["lower_effect_box"])
        upper_lo, upper_hi = map(_fraction, item["upper_effect_box"])
        assert 0 <= lower_hi - lower_lo <= width
        assert 0 <= upper_hi - upper_lo <= width

        lower_lo_loss = result.minimum_loss_with_effect_bound(lower_lo, "at_most")
        lower_hi_loss = result.minimum_loss_with_effect_bound(lower_hi, "at_most")
        upper_lo_loss = result.minimum_loss_with_effect_bound(upper_lo, "at_least")
        upper_hi_loss = result.minimum_loss_with_effect_bound(upper_hi, "at_least")
        assert lower_lo_loss is None or lower_lo_loss > threshold
        assert lower_hi_loss is not None and lower_hi_loss <= threshold
        assert upper_lo_loss is not None and upper_lo_loss <= threshold
        assert upper_hi_loss is None or upper_hi_loss > threshold

    print(
        "verified synth_demo_k2: exact tau=%.9f; "
        "relative sign breakdown=%.6f%%; 3 positive-gap endpoint boxes replayed"
        % (
            float(result.exact_optimum_effects[0]),
            100 * float(result.sign_breakdown_epsilon / result.outer_optimum),
        )
    )


if __name__ == "__main__":
    main()
