#!/usr/bin/env python3
"""Replay the real-data Basque K=2 support/KKT and endpoint certificates."""

import csv
import json
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scm_oracle import _dot, squared_loss  # noqa: E402
from src.two_predictor_scalable import reduce_two_predictor_problem_scalable  # noqa: E402


def _load_design(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    treated, donors = rows[0], rows[1:]
    predictor_names = ("mean_gdpcap_1960_1969", "mean_invest_1964_1969")
    predictor_target = tuple(Fraction(treated[name]) for name in predictor_names)
    predictor_donors = tuple(
        tuple(Fraction(donor[name]) for donor in donors) for name in predictor_names
    )
    validation_target = tuple(
        Fraction(treated[f"gdpcap_{year}"]) for year in range(1960, 1970)
    )
    validation_donors = tuple(
        tuple(Fraction(donor[f"gdpcap_{year}"]) for donor in donors)
        for year in range(1960, 1970)
    )
    treated_post = Fraction(treated["mean_gdpcap_1970_1975"])
    donor_post = tuple(Fraction(donor["mean_gdpcap_1970_1975"]) for donor in donors)
    return (
        predictor_target,
        predictor_donors,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )


def main():
    with (ROOT / "certificates" / "basque_k2.json").open() as handle:
        saved = json.load(handle)
    result = reduce_two_predictor_problem_scalable(*_load_design(ROOT / saved["instance"]))
    expected = saved["expected"]
    assert len(result.hull) == expected["hull_vertex_count"]
    assert len(result.branches) == expected["graph_branch_count"]
    assert result.outer_optimum == Fraction(expected["outer_optimum"])
    assert result.exact_optimum_effects == (
        Fraction(expected["exact_lower_effect"]),
        Fraction(expected["exact_upper_effect"]),
    )
    assert result.attainable_effects == (
        Fraction(expected["attainable_lower_effect"]),
        Fraction(expected["attainable_upper_effect"]),
    )
    assert result.sign_breakdown_epsilon == Fraction(expected["sign_breakdown_epsilon"])
    assert result.attained_zero_epsilon == Fraction(expected["attained_zero_epsilon"])
    assert (
        result.sign_breakdown_epsilon / result.outer_optimum
        == Fraction(expected["relative_sign_breakdown"])
    )
    for item in saved["near_envelopes"]:
        envelope = result.near_effect_envelope(
            Fraction(item["epsilon"]),
            digits=item["digits"],
        )
        assert (envelope.lower_effect.lower, envelope.lower_effect.upper) == tuple(
            map(Fraction, item["lower_effect_box"])
        )
        assert (envelope.upper_effect.lower, envelope.upper_effect.upper) == tuple(
            map(Fraction, item["upper_effect_box"])
        )
    assert all(
        branch.certificate.exact.value >= result.outer_optimum
        for branch in result.branch_results
    )

    with (ROOT / saved["instance"]).open() as handle:
        donor_regions = tuple(
            row["regionno"] for row in list(csv.DictReader(handle))[1:]
        )
    global_witness = saved["witnesses"]["global_optimum"]
    global_result = next(
        branch
        for branch in result.branch_results
        if branch.branch.name == global_witness["branch"]
        and branch.certificate.exact.value == result.outer_optimum
    )
    expected_weights = tuple(
        Fraction(global_witness["nonzero_donor_weights_by_region"].get(region, 0))
        for region in donor_regions
    )
    assert global_result.certificate.exact.support == tuple(
        global_witness["zero_based_support"]
    )
    assert (
        global_result.certificate.exact.optimizer[: len(donor_regions)]
        == expected_weights
    )

    zero_witness = saved["witnesses"]["first_attained_zero"]
    zero_weights = tuple(
        Fraction(zero_witness["nonzero_donor_weights_by_region"].get(region, 0))
        for region in donor_regions
    )
    zero_branch = next(
        branch for branch in result.branches if branch.name == zero_witness["branch"]
    )
    assert tuple(index for index, value in enumerate(zero_weights) if value) == tuple(
        zero_witness["zero_based_support"]
    )
    assert all(value >= 0 for value in zero_weights)
    assert all(
        _dot(row, zero_weights) == value
        for row, value in zip(zero_branch.equality_rows, zero_branch.equality_rhs)
    )
    assert result.treated_post - _dot(result.donor_post, zero_weights) == 0
    validation_weights = tuple(
        Fraction(1, len(result.validation_target)) for _ in result.validation_target
    )
    assert squared_loss(
        result.validation_target,
        result.validation_donors,
        validation_weights,
        zero_weights,
    ) == result.outer_optimum + result.attained_zero_epsilon
    print(
        "verified basque_k2: exact tau=%.9f; 1%% and 5%% boxes negative; "
        "relative sign breakdown=%.6f%%"
        % (
            float(result.exact_optimum_effects[0]),
            100 * float(result.sign_breakdown_epsilon / result.outer_optimum),
        )
    )


if __name__ == "__main__":
    main()
