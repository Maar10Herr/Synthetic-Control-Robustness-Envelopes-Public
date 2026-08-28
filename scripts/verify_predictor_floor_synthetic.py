#!/usr/bin/env python3
"""Independently replay the exact public synthetic predictor-floor proof."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictor_floor import (  # noqa: E402
    predictor_floor_support_system,
    treated_sample_variances,
)
from src.scm_oracle import as_matrix, as_vector, certify_qp_support  # noqa: E402
from src.simplex_polynomial import (  # noqa: E402
    SimplexCertificateLeaf,
    SimplexCoefficientCertificate,
    verify_semialgebraic_certificate,
)


FIXTURE = ROOT / "instances" / "tiny" / "predictor_floor_synthetic.json"
CERTIFICATE = ROOT / "certificates" / "predictor_floor_synthetic.json"


def main():
    data = json.loads(FIXTURE.read_text())
    payload = json.loads(CERTIFICATE.read_text())
    if payload["status"] != "exact rational synthetic predictor-floor certificate":
        raise AssertionError("certificate status changed")
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != payload["fixture_sha256"]:
        raise AssertionError("fixture hash changed")

    predictor_target = as_vector(data["predictor_target"])
    predictor_donors = as_matrix(data["predictor_donors"])
    validation_target = as_vector(data["validation_target"])
    validation_donors = as_matrix(data["validation_donors"])
    delta = Fraction(data["predictor_floor"])
    candidate_loss = Fraction(data["expected"]["outer_loss"])
    variances = treated_sample_variances(predictor_target, predictor_donors)
    if delta != Fraction(payload["predictor_floor"]):
        raise AssertionError("predictor floor changed")
    if candidate_loss != Fraction(payload["candidate_loss"]):
        raise AssertionError("candidate loss changed")
    if variances != as_vector(payload["sample_variances"]):
        raise AssertionError("standardization changed")

    standardized_weights = as_vector(
        data["expected"]["standardized_predictor_weights"]
    )
    metric = tuple(
        weight / variance for weight, variance in zip(standardized_weights, variances)
    )
    exact_qp = certify_qp_support(
        predictor_target,
        predictor_donors,
        metric,
        (as_vector([1, 1]),),
        as_vector([1]),
        (0, 1),
    )
    if exact_qp.optimizer != as_vector(data["expected"]["donor_weights"]):
        raise AssertionError("candidate donor weights changed")

    supports = tuple(
        support
        for size in range(1, 3)
        for support in itertools.combinations(range(2), size)
    )
    records = payload["support_certificates"]
    if tuple(tuple(record["support"]) for record in records) != supports:
        raise AssertionError("support traversal changed")
    for support, record in zip(supports, records):
        system = predictor_floor_support_system(
            predictor_target,
            predictor_donors,
            validation_target,
            validation_donors,
            variances,
            delta,
            support,
            candidate_loss,
        )
        labels = [
            *(f"weight:{position}" for position in range(len(system.weight_numerators))),
            *(f"reduced-gradient:{donor}" for donor in system.reduced_gradient_donors),
        ]
        if labels != record["constraint_labels"]:
            raise AssertionError("constraint traversal changed")
        certificate = SimplexCoefficientCertificate(
            len(variances),
            tuple(
                SimplexCertificateLeaf(
                    tuple(leaf["path"]),
                    leaf["disposition"],
                    leaf["constraint_index"],
                )
                for leaf in record["leaves"]
            ),
            True,
            (),
        )
        verify_semialgebraic_certificate(
            system.objective_difference,
            system.feasibility_constraints,
            certificate,
        )
        if certificate.max_depth != record["max_depth"]:
            raise AssertionError("subdivision depth changed")
    print(
        "verified synthetic predictor floor: exact loss=1/16; "
        "3 support certificates"
    )


if __name__ == "__main__":
    main()
