#!/usr/bin/env python3
"""Oracle-independent exact checks for the continuous-V adversarial proof."""

import json
from fractions import Fraction as F
from pathlib import Path

if not __debug__:
    raise SystemExit("verification requires assertions; run Python without -O")


ROOT = Path(__file__).resolve().parents[1]


def dot(left, right):
    return sum((a * b for a, b in zip(left, right)), F(0))


def loss(target, donors, weights):
    residual = [target[k] - dot(donors[k], weights) for k in range(len(target))]
    return sum((value * value for value in residual), F(0)) / len(target)


def main():
    path = ROOT / "instances" / "adversarial" / "two_outer_optima_sign_flip.json"
    with path.open() as handle:
        data = json.load(handle)

    x = tuple(map(F, data["predictor_target"]))
    X = tuple(tuple(map(F, row)) for row in data["predictor_donors"])
    z = tuple(map(F, data["validation_target"]))
    Z = tuple(tuple(map(F, row)) for row in data["validation_donors"])
    post = tuple(map(F, data["donor_post"]))
    treated_post = F(data["treated_post"])
    expected = data["expected"]

    assert len({tuple(X[k][j] for k in range(2)) for j in range(3)}) == 3
    assert len({tuple(Z[k][j] for k in range(2)) for j in range(3)}) == 3

    endpoints = []
    for item in expected["global_optimizers"]:
        a = F(item["predictor_weights"][0])
        w = tuple(map(F, item["donor_weights"]))
        assert all(value >= 0 for value in w) and sum(w, F(0)) == 1
        q = a * (x[0] - dot(X[0], w)) ** 2 + (1 - a) * (x[1] - dot(X[1], w)) ** 2
        assert q == 0
        outer = loss(z, Z, w)
        effect = treated_post - dot(post, w)
        assert outer == F(expected["continuous_v_outer_optimum"])
        assert effect == F(item["effect"])
        endpoints.append(effect)

    # Replay symbolic numerators used by the continuum proof. With
    # D=9-5a, r1*D=3(a-1), r2*D=2a. Their t-FOC numerator is identically zero.
    # Polynomial coefficients are stored low degree first.
    def poly_add(p, q):
        n = max(len(p), len(q))
        return tuple((p[i] if i < len(p) else F(0)) + (q[i] if i < len(q) else F(0)) for i in range(n))

    def poly_mul(p, q):
        out = [F(0)] * (len(p) + len(q) - 1)
        for i, left in enumerate(p):
            for j, right in enumerate(q):
                out[i + j] += left * right
        return tuple(out)

    a_poly = (F(0), F(1))
    one_minus_a = (F(1), F(-1))
    r1_num = (F(-3), F(3))
    r2_num = (F(0), F(2))
    foc_num = poly_add(
        tuple(2 * value for value in poly_mul(a_poly, r1_num)),
        tuple(3 * value for value in poly_mul(one_minus_a, r2_num)),
    )
    assert all(value == 0 for value in foc_num)

    assert F(expected["continuous_v_outer_optimum"]) < 5
    assert min(endpoints) == F(expected["lower_effect"])
    assert max(endpoints) == F(expected["upper_effect"])
    print("verified two_outer_optima: continuous V, L*=41/18, tau envelope=[-1/3, 1/3]")


if __name__ == "__main__":
    main()
