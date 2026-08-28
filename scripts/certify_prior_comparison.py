#!/usr/bin/env python3
"""Certify closest-prior feasible-set comparisons on frozen empirical designs."""

from __future__ import annotations

import csv
import json
import math
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prior_comparison import (  # noqa: E402
    ferguson_ross_constrained_error_envelope,
    koo_guo_author_default_radius,
    koo_guo_moment_envelope,
    minimum_koo_guo_moment_radius,
)
from src.simplex_relaxation import reduce_to_simplex_relaxation  # noqa: E402


def _rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _load_basque():
    rows = _rows(ROOT / "instances" / "empirical" / "basque_k2.csv")
    validation = tuple(f"gdpcap_{year}" for year in range(1960, 1970))
    return {
        "label": "frozen Basque full-design outer criterion",
        "pre_periods": list(range(1960, 1970)),
        "post_summary": "mean_gdpcap_1970_1975",
        "pre_target": tuple(Fraction(rows[0][name]) for name in validation),
        "pre_donors": tuple(
            tuple(Fraction(row[name]) for row in rows[1:]) for name in validation
        ),
        "treated_post": Fraction(rows[0]["mean_gdpcap_1970_1975"]),
        "donor_post": tuple(
            Fraction(row["mean_gdpcap_1970_1975"]) for row in rows[1:]
        ),
        "donors": [row["regionname"] for row in rows[1:]],
        "instance": "instances/empirical/basque_k2.csv",
    }


def _pair(interval):
    return [str(interval.lower), str(interval.upper)]


def _fraction_vector(values):
    return [str(value) for value in values]


def _reference_adh():
    with (ROOT / "certificates" / "basque_full.json").open() as handle:
        expected = json.load(handle)["expected"]
    return {
        "certificate": "certificates/basque_full.json",
        "outer_optimum_mse": expected["outer_optimum"],
        "exact_effect_interval": [
            expected["exact_lower_effect"],
            expected["exact_upper_effect"],
        ],
        "zero_effect_squared_loss_gap": expected["attained_zero_epsilon"],
    }


def _ferguson_ross(one_level):
    radii = []
    for radius in (Fraction(0), Fraction(1, 100), Fraction(1, 20)):
        result = ferguson_ross_constrained_error_envelope(
            one_level,
            radius,
            digits=7,
        )
        radii.append(
            {
                "B": str(radius),
                "equivalent_squared_loss_gap": str(result.squared_loss_gap),
                "lower_effect_box": _pair(result.effect_envelope.lower_effect),
                "upper_effect_box": _pair(result.effect_envelope.upper_effect),
            }
        )
    matched_squared_loss_gaps = []
    for relative_gap in (Fraction(1, 100), Fraction(1, 20)):
        epsilon = relative_gap * one_level.outer_optimum
        envelope = one_level.near_effect_envelope(epsilon, digits=7)
        matched_squared_loss_gaps.append(
            {
                "relative_squared_loss_gap": str(relative_gap),
                "equivalent_B_approx": math.sqrt(1 + float(relative_gap)) - 1,
                "squared_loss_gap": str(epsilon),
                "lower_effect_box": _pair(envelope.lower_effect),
                "upper_effect_box": _pair(envelope.upper_effect),
            }
        )
    zero_gap = one_level.attained_zero_epsilon
    if zero_gap is None:
        zero = None
    else:
        relative_squared_gap = zero_gap / one_level.outer_optimum
        zero = {
            "squared_loss_gap": str(zero_gap),
            "relative_squared_loss_gap": str(relative_squared_gap),
            "B_approx": math.sqrt(1 + float(relative_squared_gap)) - 1,
        }
    return {
        "minimum_mse": str(one_level.outer_optimum),
        "minimum_effect_interval": [
            str(one_level.exact_optimum_effects[0]),
            str(one_level.exact_optimum_effects[1]),
        ],
        "zero_effect_breakdown": zero,
        "radii": radii,
        "matched_squared_loss_relative_gaps": matched_squared_loss_gaps,
    }


def _author_radius(one_level, design, nominal_lambda, feasible_radius):
    return koo_guo_author_default_radius(
        one_level,
        design["pre_target"],
        design["pre_donors"],
        nominal_lambda,
        feasible_radius,
    )


def _koo_guo(one_level, design):
    feasible = minimum_koo_guo_moment_radius(
        design["pre_target"],
        design["pre_donors"],
    )
    zero = minimum_koo_guo_moment_radius(
        design["pre_target"],
        design["pre_donors"],
        design["donor_post"],
        design["treated_post"],
    )
    floor_envelope = koo_guo_moment_envelope(
        design["pre_target"],
        design["pre_donors"],
        design["treated_post"],
        design["donor_post"],
        feasible.radius,
    )

    first_zero_grid = None
    for numerator in range(61):
        nominal = Fraction(numerator, 1000)
        radius = _author_radius(one_level, design, nominal, feasible.radius)
        if radius.effective_radius >= zero.radius:
            first_zero_grid = nominal
            break

    selected = [Fraction(0), Fraction(15, 1000), Fraction(30, 1000)]
    selected.extend((Fraction(45, 1000), Fraction(60, 1000)))
    if first_zero_grid is not None and first_zero_grid not in selected:
        selected.append(first_zero_grid)
    path = []
    for nominal in sorted(selected):
        radius = _author_radius(one_level, design, nominal, feasible.radius)
        envelope = koo_guo_moment_envelope(
            design["pre_target"],
            design["pre_donors"],
            design["treated_post"],
            design["donor_post"],
            radius.effective_radius,
        )
        path.append(
            {
                "lambda": str(nominal),
                "effective_radius": str(radius.effective_radius),
                "nu_used": str(radius.nu_used),
                "feasibility_attempts": radius.feasibility_attempts,
                "lower_effect": str(envelope.lower_effect),
                "upper_effect": str(envelope.upper_effect),
                "zero_robust_effect": str(envelope.zero_robust_effect),
            }
        )
    radius_zero_ratio = zero.radius / feasible.radius if feasible.radius else None
    return {
        "minimum_feasible_radius": str(feasible.radius),
        "minimum_feasible_weights": _fraction_vector(feasible.donor_weights),
        "floor_effect_interval": [
            str(floor_envelope.lower_effect),
            str(floor_envelope.upper_effect),
        ],
        "minimum_zero_effect_radius": str(zero.radius),
        "minimum_zero_effect_weights": _fraction_vector(zero.donor_weights),
        "zero_to_feasibility_radius_ratio": (
            str(radius_zero_ratio) if radius_zero_ratio is not None else None
        ),
        "first_zero_lambda_on_authors_0.001_grid_through_0.06": (
            str(first_zero_grid) if first_zero_grid is not None else None
        ),
        "author_default_style_path": path,
        "author_radius_diagnostics": {
            "residual_sample_variance": str(
                _author_radius(
                    one_level, design, Fraction(0), feasible.radius
                ).residual_sample_variance
            ),
            "maximum_donor_mean_square": str(
                _author_radius(
                    one_level, design, Fraction(0), feasible.radius
                ).maximum_donor_mean_square
            ),
        },
    }


def _certify_design(design):
    one_level = reduce_to_simplex_relaxation(
        design["pre_target"],
        design["pre_donors"],
        design["treated_post"],
        design["donor_post"],
    )
    return {
        "label": design["label"],
        "instance": design["instance"],
        "pre_periods": design["pre_periods"],
        "post_summary": design["post_summary"],
        "donor_count": len(design["donors"]),
        "donors": design["donors"],
        "adh_bilevel_reference": _reference_adh(),
        "ferguson_ross_constrained_error": _ferguson_ross(one_level),
        "koo_guo_sample_moment_class": _koo_guo(one_level, design),
    }


def build_payload():
    return {
        "schema_version": 1,
        "status": (
            "exact rational endpoint certificates conditional on frozen inputs; "
            "Koo-Guo author-style transcendental radii serialized from binary64"
        ),
        "sources": {
            "ferguson_ross": {
                "arxiv": "2012.15367v3",
                "pdf_sha256": (
                    "14f712b416ef2c5c097d16813a3d1a51a2a63410125028a6e7a3570ffd0689df"
                ),
                "definition": "equation 16 constrained error space",
            },
            "koo_guo": {
                "arxiv": "2511.02632v3",
                "pdf_sha256": (
                    "9c47f9f780f827539e9f9355f3112888a7820eb1a00959182d931f183fc657d5"
                ),
                "definition": "equations 13-16 and author point-estimation code",
                "author_code_commit": "e95c713c163e2c4d3b0906e38266d4f3e934f98c",
                "helpers_R_sha256": (
                    "29368791b2c377269fcfbefd4cd21223c57a17ee1092c831a9b7912b55d3ff39"
                ),
            },
        },
        "comparison_scope": {
            "post_effect": "treated minus simplex-weighted donor post summary",
            "ferguson_ross": (
                "outcome-only simplex Euclidean pre-fit sublevel; no placebo calibration"
            ),
            "koo_guo": (
                "sample analogue of the moment class with author default-style "
                "point-estimation radius; no perturbation inference"
            ),
            "not_claimed": (
                "not a replication of either paper's empirical analysis and not a "
                "statistical comparison"
            ),
        },
        "designs": {
            "basque": _certify_design(_load_basque()),
        },
    }


def main():
    payload = build_payload()
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(output)
    elif len(sys.argv) == 1:
        sys.stdout.write(output)
    else:
        raise SystemExit("usage: certify_prior_comparison.py [output.json]")


if __name__ == "__main__":
    main()
