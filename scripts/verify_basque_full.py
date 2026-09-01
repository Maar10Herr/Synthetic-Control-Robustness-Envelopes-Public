#!/usr/bin/env python3
"""Replay the full K=13 Basque relaxation/inner-KKT certificates."""

import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

if not __debug__:
    raise SystemExit("verification requires assertions; run Python without -O")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.general_predictor import _donor_points, _faces, _facets  # noqa: E402
from src.general_predictor_scalable import (  # noqa: E402
    certify_inner_attainability_on_predictor_support,
)
from src.scm_oracle import _rank  # noqa: E402
from src.simplex_relaxation import reduce_to_simplex_relaxation  # noqa: E402
from src.two_predictor_scalable import (  # noqa: E402
    reduce_two_predictor_problem_scalable,
)


def _load_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _load_design(certificate):
    predictor_rows = _load_rows(ROOT / certificate["predictor_instance"])
    outcome_rows = _load_rows(ROOT / certificate["outcome_instance"])
    assert [row["regionno"] for row in predictor_rows] == [
        row["regionno"] for row in outcome_rows
    ]
    predictor_names = tuple(list(predictor_rows[0])[2:])
    predictor_target = tuple(
        Fraction(predictor_rows[0][name]) for name in predictor_names
    )
    predictor_donors = tuple(
        tuple(Fraction(row[name]) for row in predictor_rows[1:])
        for name in predictor_names
    )
    validation_target = tuple(
        Fraction(outcome_rows[0][f"gdpcap_{year}"])
        for year in range(1960, 1970)
    )
    validation_donors = tuple(
        tuple(Fraction(row[f"gdpcap_{year}"]) for row in outcome_rows[1:])
        for year in range(1960, 1970)
    )
    treated_post = Fraction(outcome_rows[0]["mean_gdpcap_1970_1975"])
    donor_post = tuple(
        Fraction(row["mean_gdpcap_1970_1975"]) for row in outcome_rows[1:]
    )
    donor_regions = tuple(row["regionno"] for row in predictor_rows[1:])
    return (
        predictor_names,
        predictor_target,
        predictor_donors,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
        donor_regions,
    )


def _expected_weights(witness, donor_regions):
    weights = witness["nonzero_donor_weights_by_region"]
    return tuple(Fraction(weights.get(region, 0)) for region in donor_regions)


def main():
    with (ROOT / "certificates" / "basque_full.json").open() as handle:
        saved = json.load(handle)
    (
        predictor_names,
        predictor_target,
        predictor_donors,
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
        donor_regions,
    ) = _load_design(saved)

    geometry = saved["geometry"]
    points = _donor_points(predictor_donors)
    differences = tuple(
        tuple(point[k] - points[0][k] for k in range(len(predictor_target)))
        for point in points[1:]
    )
    assert _rank(differences) == geometry["affine_rank"]
    facet_normals = _facets(points)
    faces = _faces(len(points), tuple(facet_normals))
    assert len(facet_normals) == geometry["facet_count"]
    assert len(faces) == geometry["nonempty_face_count_including_full_hull"]
    assert sum(len(face) == 1 for face in faces) == geometry["hull_vertex_count"]
    assert (len(faces) - 1) * 2 ** len(predictor_target) == geometry[
        "naive_proper_face_orthant_pairs"
    ]

    relaxation = reduce_to_simplex_relaxation(
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
    expected = saved["expected"]
    assert relaxation.outer_optimum == Fraction(expected["outer_optimum"])
    assert relaxation.exact_optimum_effects == (
        Fraction(expected["exact_lower_effect"]),
        Fraction(expected["exact_upper_effect"]),
    )
    assert relaxation.attainable_effects == (
        Fraction(expected["simplex_attainable_lower_effect"]),
        Fraction(expected["simplex_attainable_upper_effect"]),
    )
    assert relaxation.sign_breakdown_epsilon == Fraction(
        expected["sign_breakdown_epsilon"]
    )
    assert relaxation.attained_zero_epsilon == Fraction(
        expected["attained_zero_epsilon"]
    )
    assert (
        relaxation.sign_breakdown_epsilon / relaxation.outer_optimum
        == Fraction(expected["relative_sign_breakdown"])
    )

    global_witness = saved["witnesses"]["global_optimum"]
    global_weights = _expected_weights(global_witness, donor_regions)
    assert relaxation.optimum.exact.optimizer == global_weights
    assert relaxation.optimum.exact.support == tuple(
        global_witness["zero_based_donor_support"]
    )
    global_inner = certify_inner_attainability_on_predictor_support(
        predictor_target,
        predictor_donors,
        global_weights,
        global_witness["zero_based_predictor_support"],
    )
    assert global_inner.inner.optimizer == global_weights

    zero_witness = saved["witnesses"]["first_attained_zero"]
    zero_certificate = relaxation.effect_bound_certificate(Fraction(0), "equal")
    assert zero_certificate is not None
    zero_weights = zero_certificate.exact.optimizer[: len(donor_regions)]
    assert zero_certificate.exact.value == Fraction(expected["first_zero_loss"])
    assert zero_weights == _expected_weights(zero_witness, donor_regions)
    assert tuple(index for index, value in enumerate(zero_weights) if value) == tuple(
        zero_witness["zero_based_donor_support"]
    )
    zero_inner = certify_inner_attainability_on_predictor_support(
        predictor_target,
        predictor_donors,
        zero_weights,
        zero_witness["zero_based_predictor_support"],
    )
    assert zero_inner.inner.optimizer == zero_weights

    subgraph = saved["inner_lower_bound_subgraph"]
    first, second = subgraph["zero_based_predictor_indices"]
    assert [predictor_names[first], predictor_names[second]] == subgraph["predictors"]
    two_predictor = reduce_two_predictor_problem_scalable(
        (predictor_target[first], predictor_target[second]),
        (predictor_donors[first], predictor_donors[second]),
        validation_target,
        validation_donors,
        treated_post,
        donor_post,
    )
    for item in saved["near_envelopes"]:
        epsilon = Fraction(item["epsilon"])
        threshold = relaxation.outer_optimum + epsilon
        outer = relaxation.near_effect_envelope(epsilon, digits=item["digits"])
        assert (outer.lower_effect.lower, outer.lower_effect.upper) == tuple(
            map(Fraction, item["simplex_lower_box"])
        )
        assert (outer.upper_effect.lower, outer.upper_effect.upper) == tuple(
            map(Fraction, item["certified_full_upper_box"])
        )

        lower_infeasible = relaxation.minimum_loss_with_effect_bound(
            outer.lower_effect.lower,
            "at_most",
        )
        upper_infeasible = relaxation.minimum_loss_with_effect_bound(
            outer.upper_effect.upper,
            "at_least",
        )
        assert lower_infeasible is not None and lower_infeasible > threshold
        assert upper_infeasible is not None and upper_infeasible > threshold

        subgraph_epsilon = Fraction(item["subgraph_epsilon"])
        assert (
            two_predictor.outer_optimum + subgraph_epsilon
            == relaxation.outer_optimum + epsilon
        )
        inner = two_predictor.near_effect_envelope(
            subgraph_epsilon,
            digits=item["digits"],
        )
        assert (inner.lower_effect.lower, inner.lower_effect.upper) == tuple(
            map(Fraction, item["subgraph_lower_box"])
        )
        assert (
            outer.lower_effect.lower,
            inner.lower_effect.upper,
        ) == tuple(map(Fraction, item["certified_full_lower_box"]))
        inner_feasible = two_predictor.minimum_loss_with_effect_bound(
            inner.lower_effect.upper,
            "at_most",
        )
        assert inner_feasible is not None and inner_feasible <= threshold

        upper_feasible = relaxation.effect_bound_certificate(
            outer.upper_effect.lower,
            "at_least",
        )
        assert upper_feasible is not None and upper_feasible.exact.value <= threshold
        upper_weights = upper_feasible.exact.optimizer[: len(donor_regions)]
        upper_inner = certify_inner_attainability_on_predictor_support(
            predictor_target,
            predictor_donors,
            upper_weights,
            item["upper_feasible_zero_based_predictor_support"],
        )
        assert upper_inner.inner.optimizer == upper_weights

    print(
        "verified basque_full: K=13 exact tau=%.9f; relative sign "
        "breakdown=%.6f%%; 1%% and 5%% endpoint boxes replayed"
        % (
            float(relaxation.exact_optimum_effects[0]),
            100 * float(relaxation.sign_breakdown_epsilon / relaxation.outer_optimum),
        )
    )


if __name__ == "__main__":
    main()
