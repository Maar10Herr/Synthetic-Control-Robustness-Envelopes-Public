#!/usr/bin/env python3
"""Generate the exact coordinate-sandwich certificate for Proposition 99."""

from __future__ import annotations

import json
import platform
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coordinate_sandwich import reduce_with_coordinate_sandwich  # noqa: E402
from src.prop99 import load_prop99_full  # noqa: E402


def _interval(interval):
    return [str(interval.lower), str(interval.upper)]


def _sparse_weights(values, labels):
    return {
        label: str(value)
        for label, value in zip(labels, values)
        if value != 0
    }


def _candidate_payload(candidate, design):
    zero = candidate.zero_effect_optimum
    return {
        "zero_based_coordinate": candidate.coordinate,
        "predictor": design.predictor_names[candidate.coordinate],
        "outer_loss": str(candidate.optimum.exact.value),
        "outer_effect_range": [str(value) for value in candidate.exact_optimum_effects],
        "attainable_effect_range": [
            str(value) for value in candidate.attainable_effects
        ],
        "outer_weights": _sparse_weights(
            candidate.optimum.exact.optimizer[: len(design.donor_labels)],
            design.donor_labels,
        ),
        "zero_effect_loss": str(zero.exact.value) if zero is not None else None,
        "zero_effect_weights": (
            _sparse_weights(
                zero.exact.optimizer[: len(design.donor_labels)],
                design.donor_labels,
            )
            if zero is not None
            else None
        ),
    }


def build_payload():
    design = load_prop99_full(
        ROOT / "instances" / "empirical" / "prop99_full.csv"
    )
    result = reduce_with_coordinate_sandwich(
        design.predictor_target,
        design.predictor_donors,
        design.validation_target,
        design.validation_donors,
        design.treated_post,
        design.donor_post,
    )
    loss_lower, loss_upper = result.optimum_loss_bounds
    zero_bounds = result.zero_loss_bounds
    absolute_zero = result.absolute_zero_gap_bounds
    relative_zero = result.relative_zero_gap_bounds
    if zero_bounds is None or absolute_zero is None or relative_zero is None:
        raise AssertionError("the Proposition 99 zero-effect sandwich did not close")
    optimum_effect_box = result.optimum_effect_outer_box(digits=6)

    near = []
    for label, relative_gap in (
        ("one_percent_relative_outer_gap", Fraction(1, 100)),
        ("five_percent_relative_outer_gap", Fraction(5, 100)),
    ):
        envelope = result.relative_envelope_sandwich(relative_gap, digits=6)
        near.append(
            {
                "label": label,
                "relative_gap": str(relative_gap),
                "digits": 6,
                "relaxation_epsilon": str(envelope.outer_epsilon),
                "coordinate_epsilon": str(envelope.inner_epsilon),
                "relaxation_lower_box": _interval(
                    envelope.relaxation_envelope.lower_effect
                ),
                "relaxation_upper_box": _interval(
                    envelope.relaxation_envelope.upper_effect
                ),
                "coordinate_lower_box": _interval(
                    envelope.coordinate_envelope.lower_effect
                ),
                "coordinate_upper_box": _interval(
                    envelope.coordinate_envelope.upper_effect
                ),
                "certified_full_lower_box": _interval(envelope.full_lower_effect),
                "certified_full_upper_box": _interval(envelope.full_upper_effect),
            }
        )

    try:
        import numpy
        import scipy

        discovery = {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        }
    except ImportError:
        discovery = {"python": platform.python_version()}

    best = result.best_candidate
    best_zero = result.best_zero_candidate
    if best_zero is None:
        raise AssertionError("missing coordinate zero-effect witness")
    return {
        "schema_version": 1,
        "instance": "instances/empirical/prop99_full.csv",
        "metadata": "instances/empirical/prop99_full_metadata.json",
        "status": (
            "exact rational global bounds conditional on the six-decimal fixture; "
            "unmatched bounds are reported as outward intervals, not sharp points"
        ),
        "method": (
            "donor-simplex relaxation lower bounds plus exact ADH coordinate-match "
            "branches V=e_k; numerical LP/QP supports accepted only by rational KKT"
        ),
        "discovery_environment": discovery,
        "design": {
            "treated_unit": design.treated_label,
            "predictor_count": len(design.predictor_names),
            "donor_count": len(design.donor_labels),
            "validation_period_count": len(design.validation_names),
            "post_summary": "mean cigsale 1989-2000",
            "effect_direction": "treated minus synthetic",
        },
        "relaxation": {
            "loss": str(result.relaxation.outer_optimum),
            "exact_optimum_effect_range": [
                str(value) for value in result.relaxation.exact_optimum_effects
            ],
            "weights": _sparse_weights(
                result.relaxation.optimum.exact.optimizer,
                design.donor_labels,
            ),
        },
        "coordinate_candidates": [
            _candidate_payload(candidate, design) for candidate in result.candidates
        ],
        "best_feasible_candidate": {
            "zero_based_coordinate": best.coordinate,
            "predictor": design.predictor_names[best.coordinate],
            "loss": str(best.optimum.exact.value),
            "effect_range": [str(value) for value in best.exact_optimum_effects],
            "weights": _sparse_weights(
                best.optimum.exact.optimizer,
                design.donor_labels,
            ),
        },
        "best_zero_effect_candidate": {
            "zero_based_coordinate": best_zero.coordinate,
            "predictor": design.predictor_names[best_zero.coordinate],
            "loss": str(best_zero.zero_effect_optimum.exact.value),
            "weights": _sparse_weights(
                best_zero.zero_effect_optimum.exact.optimizer,
                design.donor_labels,
            ),
        },
        "certified_global_bounds": {
            "optimum_loss": [str(loss_lower), str(loss_upper)],
            "relative_optimum_loss_width": str(loss_upper / loss_lower - 1),
            "optimum_effect_outer_box": _interval(optimum_effect_box),
            "first_zero_effect_loss": [str(value) for value in zero_bounds],
            "absolute_sign_or_zero_gap": [str(value) for value in absolute_zero],
            "relative_sign_or_zero_gap": [str(value) for value in relative_zero],
            "interpretation": (
                "the same outward bracket contains both first nonnegative-effect "
                "and first attained-zero thresholds; equality is not claimed"
            ),
        },
        "near_envelopes": near,
    }


def main():
    output = json.dumps(build_payload(), indent=2, sort_keys=True) + "\n"
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(output)
    elif len(sys.argv) == 1:
        sys.stdout.write(output)
    else:
        raise SystemExit("usage: certify_prop99_full.py [output.json]")


if __name__ == "__main__":
    main()
