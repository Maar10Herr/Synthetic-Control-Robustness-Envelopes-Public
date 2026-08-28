#!/usr/bin/env python3
"""Independent exact verifier for the permanent sign-flip certificate.

This script intentionally does not import the oracle. It checks the affine
parameterization, proves it is the entire zero-loss feasible set by rank/nullity,
derives its nonnegative parameter interval, and evaluates the affine effect at
both endpoints using only ``fractions.Fraction``.
"""

import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def f(value):
    return Fraction(value)


def dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def rank(rows):
    work = [list(row) for row in rows]
    pivot_row = 0
    for col in range(len(work[0])):
        pivot = next((r for r in range(pivot_row, len(work)) if work[r][col]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][col]
        work[pivot_row] = [x / scale for x in work[pivot_row]]
        for row in range(len(work)):
            if row != pivot_row and work[row][col]:
                scale = work[row][col]
                work[row] = [a - scale * b for a, b in zip(work[row], work[pivot_row])]
        pivot_row += 1
    return pivot_row


def nonnegative_interval(base, direction):
    lower = None
    upper = None
    for intercept, slope in zip(base, direction):
        if slope > 0:
            bound = -intercept / slope
            lower = bound if lower is None else max(lower, bound)
        elif slope < 0:
            bound = -intercept / slope
            upper = bound if upper is None else min(upper, bound)
        elif intercept < 0:
            raise AssertionError("empty affine nonnegative set")
    if lower is None or upper is None:
        raise AssertionError("certificate must give a bounded segment")
    return lower, upper


def main():
    with (ROOT / "instances" / "tiny" / "exact_sign_flip.json").open() as handle:
        instance = json.load(handle)
    with (ROOT / "certificates" / "exact_sign_flip.json").open() as handle:
        certificate = json.load(handle)

    base = tuple(map(f, certificate["weights"]["base"]))
    direction = tuple(map(f, certificate["weights"]["direction"]))
    predictor = tuple(map(f, instance["predictor_donors"][0]))
    validation = tuple(map(f, instance["validation_donors"][0]))
    ones = tuple(Fraction(1) for _ in base)

    assert dot(ones, base) == 1 and dot(ones, direction) == 0
    assert dot(predictor, base) == f(instance["predictor_target"][0])
    assert dot(predictor, direction) == 0
    assert dot(validation, base) == f(instance["validation_target"][0])
    assert dot(validation, direction) == 0
    assert rank((ones, predictor)) == 2
    assert len(base) - rank((ones, predictor)) == 1

    lower_t, upper_t = nonnegative_interval(base, direction)
    assert lower_t == f(certificate["weights"]["lower"])
    assert upper_t == f(certificate["weights"]["upper"])

    donor_post = tuple(map(f, instance["donor_post"]))
    treated_post = f(instance["treated_post"])
    effect_intercept = treated_post - dot(donor_post, base)
    effect_slope = -dot(donor_post, direction)
    endpoint_effects = {
        effect_intercept + effect_slope * lower_t,
        effect_intercept + effect_slope * upper_t,
    }
    assert min(endpoint_effects) == f(certificate["lower_effect"])
    assert max(endpoint_effects) == f(certificate["upper_effect"])
    assert f(certificate["outer_optimum"]) == 0
    print("verified exact_sign_flip: L*=0, tau envelope=[-2, 2]")


if __name__ == "__main__":
    main()

