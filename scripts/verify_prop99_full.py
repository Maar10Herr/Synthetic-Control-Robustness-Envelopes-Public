#!/usr/bin/env python3
"""Replay the full Proposition 99 coordinate-sandwich certificate."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.certify_prop99_full import build_payload  # noqa: E402


def main():
    path = ROOT / "certificates" / "prop99_full.json"
    with path.open() as handle:
        saved = json.load(handle)
    replayed = build_payload()
    # Runtime versions document discovery provenance but are not mathematical
    # certificate content.  A clean runner may use a different Python patch
    # release while reproducing the same exact rational payload.
    saved_certificate = {
        key: value for key, value in saved.items() if key != "discovery_environment"
    }
    replayed_certificate = {
        key: value for key, value in replayed.items() if key != "discovery_environment"
    }
    if replayed_certificate != saved_certificate:
        raise AssertionError("Proposition 99 coordinate sandwich changed")

    bounds = saved["certified_global_bounds"]
    loss_lower, loss_upper = map(Fraction, bounds["optimum_loss"])
    effect_lower, effect_upper = map(Fraction, bounds["optimum_effect_outer_box"])
    zero_lower, zero_upper = map(Fraction, bounds["first_zero_effect_loss"])
    relative_lower, relative_upper = map(
        Fraction, bounds["relative_sign_or_zero_gap"]
    )
    if not 0 < loss_lower <= loss_upper < zero_lower <= zero_upper:
        raise AssertionError("global loss/zero sandwich ordering changed")
    if not effect_lower <= effect_upper < 0:
        raise AssertionError("globally optimal effect is no longer certified negative")
    if not 300 < relative_lower <= relative_upper:
        raise AssertionError("large relative sign-robustness bracket changed")

    candidates = saved["coordinate_candidates"]
    best = min(candidates, key=lambda item: Fraction(item["outer_loss"]))
    if best["predictor"] != "cigsale_1980":
        raise AssertionError("best coordinate branch changed")
    zero_candidates = [item for item in candidates if item["zero_effect_loss"]]
    best_zero = min(
        zero_candidates,
        key=lambda item: Fraction(item["zero_effect_loss"]),
    )
    if best_zero["predictor"] != "mean_retprice_1980_1988":
        raise AssertionError("best zero-effect coordinate branch changed")

    for envelope in saved["near_envelopes"]:
        lower_box = tuple(map(Fraction, envelope["certified_full_lower_box"]))
        upper_box = tuple(map(Fraction, envelope["certified_full_upper_box"]))
        if not lower_box[0] <= lower_box[1] < upper_box[0] <= upper_box[1] < 0:
            raise AssertionError("near-optimal endpoint sign/order changed")

    print(
        "verified prop99_full: exact MSE bracket [%.9f, %.9f]; "
        "globally optimal and 1%%/5%% endpoint boxes remain negative"
        % (float(loss_lower), float(loss_upper))
    )


if __name__ == "__main__":
    main()
