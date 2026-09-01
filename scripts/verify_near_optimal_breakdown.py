#!/usr/bin/env python3
"""Independent rational verification of the near-optimal breakdown fixture."""

import json
from fractions import Fraction as F
from pathlib import Path

if not __debug__:
    raise SystemExit("verification requires assertions; run Python without -O")


ROOT = Path(__file__).resolve().parents[1]


def dot(left, right):
    return sum((a * b for a, b in zip(left, right)), F(0))


def outer_loss(target, donors, weights):
    residual = [target[k] - dot(donors[k], weights) for k in range(len(target))]
    return sum((value * value for value in residual), F(0)) / len(target)


def effect(treated, donor_post, weights):
    return treated - dot(donor_post, weights)


def main():
    path = ROOT / "instances" / "adversarial" / "near_optimal_sign_breakdown.json"
    with path.open() as handle:
        data = json.load(handle)

    z = tuple(map(F, data["validation_target"]))
    Z = tuple(tuple(map(F, row)) for row in data["validation_donors"])
    donor_post = tuple(map(F, data["donor_post"]))
    treated = F(data["treated_post"])
    expected = data["expected"]

    global_item = expected["unique_global_optimizer"]
    opposite_item = expected["first_opposite_sign_solution"]
    zero_item = expected["first_attained_zero"]
    w_a = tuple(map(F, global_item["donor_weights"]))
    w_b = tuple(map(F, opposite_item["donor_weights"]))
    w_zero = tuple(map(F, zero_item["donor_weights"]))

    loss_a = outer_loss(z, Z, w_a)
    loss_b = outer_loss(z, Z, w_b)
    loss_zero = outer_loss(z, Z, w_zero)
    assert loss_a == F(global_item["outer_loss"])
    assert loss_b == F(opposite_item["outer_loss"])
    assert effect(treated, donor_post, w_a) == F(global_item["effect"]) == F(-1, 3)
    assert effect(treated, donor_post, w_b) == F(opposite_item["effect"]) == F(1, 3)
    assert effect(treated, donor_post, w_zero) == 0

    epsilon_span = loss_b - loss_a
    epsilon_attained = loss_zero - loss_a
    assert epsilon_span == F(expected["envelope_sign_breakdown_epsilon"]) == F(1, 1000)
    assert F(expected["relative_outer_loss_gap"]) == epsilon_span / loss_a
    assert epsilon_attained == F(zero_item["epsilon"]) == F(26081, 81000)
    assert epsilon_attained - epsilon_span == F(26, 81)
    assert loss_b < 3 < 5
    print(
        "verified near_optimal_breakdown: unique optimum; "
        "span epsilon*=1/1000; attained-zero epsilon=26081/81000"
    )


if __name__ == "__main__":
    main()
