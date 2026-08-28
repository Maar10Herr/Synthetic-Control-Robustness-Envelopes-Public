#!/usr/bin/env python3
"""Replay the certified Ferguson--Ross/Koo--Guo empirical comparison."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.certify_prior_comparison import build_payload  # noqa: E402


def main():
    path = ROOT / "certificates" / "prior_feasible_set_comparison.json"
    with path.open() as handle:
        saved = json.load(handle)
    replayed = build_payload()
    if replayed != saved:
        raise AssertionError("prior feasible-set comparison changed")

    basque = saved["designs"]["basque"]
    basque_kg = basque["koo_guo_sample_moment_class"]
    if basque_kg["first_zero_lambda_on_authors_0.001_grid_through_0.06"] != "21/1000":
        raise AssertionError("Basque Koo-Guo grid breakdown changed")
    for design in (basque,):
        fr = design["ferguson_ross_constrained_error"]
        if Fraction(fr["minimum_mse"]) <= 0:
            raise AssertionError("Ferguson-Ross relative metric must be defined")
        kg = design["koo_guo_sample_moment_class"]
        if Fraction(kg["minimum_zero_effect_radius"]) < Fraction(
            kg["minimum_feasible_radius"]
        ):
            raise AssertionError("zero-effect moment radius is below feasibility floor")
        for row in kg["author_default_style_path"]:
            lower = Fraction(row["lower_effect"])
            upper = Fraction(row["upper_effect"])
            robust = Fraction(row["zero_robust_effect"])
            expected = lower if lower > 0 else upper if upper < 0 else Fraction(0)
            if robust != expected:
                raise AssertionError("Koo-Guo zero projection changed")

    print(
        "prior feasible-set comparison verified: exact LP/QP KKT replay; "
        "Basque Koo-Guo zero grid lambda=0.021"
    )


if __name__ == "__main__":
    main()
