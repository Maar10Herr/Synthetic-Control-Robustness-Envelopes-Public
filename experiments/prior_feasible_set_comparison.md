# Closest-prior feasible sets on the frozen Basque design

Date checked: 2026-08-27.

## Question and scope

This comparison asks how the complete ADH lower graph differs from two nearby
convex robustness objects. It implements:

- Ferguson and Ross's equation-16 constrained-error metric over outcome-only
  simplex weights;
- the sample analogue of Koo and Guo's moment class in equations 13--16,
  including the point-estimation radius loop in their author code.

It uses the same rational pre-outcome matrix and declared scalar post summary
as the certified Basque analysis. It does not reproduce either paper's
empirical application, run Ferguson--Ross placebo calibration, or run
Koo--Guo perturbation inference. The comparison is optimization/procedural,
not statistical.

Primary sources:

- Billy Ferguson and Brad Ross, “Assessing the Sensitivity of Synthetic Control
  Treatment Effect Estimates to Misspecification Error,”
  [arXiv:2012.15367v3](https://arxiv.org/abs/2012.15367).
- Taehyeon Koo and Zijian Guo, “Synthetic Control with Weight Uncertainty:
  Robust Identification and Statistical Inference,”
  [arXiv:2511.02632v3](https://arxiv.org/abs/2511.02632), and author code at
  [taehyeonkoo/DRoSC](https://github.com/taehyeonkoo/DRoSC), commit
  `e95c713c163e2c4d3b0906e38266d4f3e934f98c`.

PDF and source-code hashes are frozen in
`certificates/prior_feasible_set_comparison.json`.

## Implemented objects

For pre-outcome target `y`, donor matrix `X`, and donor simplex `Delta`, the
Ferguson--Ross constrained-error class at radius `B` is

```text
F(B) = {w in Delta : ||y-Xw|| <= (1+B) min_{q in Delta} ||y-Xq||}.
```

When the minimum squared error `L_FR` is positive, this is the squared-loss
sublevel

```text
L(w) <= L_FR + ((1+B)^2-1)L_FR.
```

Its effect image is an attained interval because `F(B)` is convex. The saved
certificate reports rational `B` values and matched 1%/5% relative squared-loss
gaps.

For Koo--Guo, the exact sample moment class is

```text
Sigma_hat = X'X/T0,
gamma_hat = X'y/T0,
Omega_hat(q) = {w in Delta : ||gamma_hat-Sigma_hat w||_infinity <= q}.
```

The code separately certifies the minimum feasible radius, the minimum radius
admitting exact zero effect, and both affine-effect endpoints at a rational
radius. It also replays the authors' default point-estimation loop. Square
roots and logarithms follow binary64 implementation arithmetic; resulting
radii are serialized with 17 significant digits, after which endpoint LPs are
checked exactly conditional on those radii.

These classes are convex donor-weight sets. The ADH object is instead the
generally nonconvex lower graph selected by some predictor metric, intersected
with an outer validation-loss sublevel. The exact disconnected three-donor
example proves that the distinction is structural: its ADH effect image can
span zero without attaining zero, which cannot happen for a nonempty convex
donor-weight class under an affine effect.

## Certified public-data result

The frozen design uses GDP per capita in 1960--1969 and a declared mean
1970--1975 effect over 16 donors.

```text
ADH bilevel optimum MSE                 0.0041263737767375215
ADH exact effect                        -0.2682004674583837
ADH zero relative squared-loss gap      109.3716023221171%

Ferguson--Ross one-level minimum MSE    0.0041263737767375215
Ferguson--Ross minimum effect           -0.2682004674583837
Ferguson--Ross zero B                    0.4469678722145738
```

At matched relative squared-loss gaps:

| Relative MSE gap | Equivalent `B` | Ferguson--Ross effect interval |
| ---: | ---: | ---: |
| 1% | `0.0049875621` | `[-0.2948300, -0.2395807]` |
| 5% | `0.0246950766` | `[-0.3274251, -0.1873163]` |

These endpoints agree with the narrow Basque ADH endpoint boxes to reported
precision. Here the donor-simplex outer relaxation is empirically nearly
indistinguishable from the certified bilevel envelope at 1% and 5%, and its
exact optimum and zero threshold are inner-attainable. This is a negative
heterogeneity result, not evidence that the feasible sets are identical.

For the Koo--Guo moment class:

```text
minimum feasible q                       0.002555054378565246
effect at minimum feasible q             -0.2829191079589833
minimum zero-effect q                    0.02253914330420868
q_zero / q_min                           8.821394759067793
author-style lambda=0 effective q        0.002667683071163349
author-style lambda=0 interval           [-0.3118734, -0.2225259]
first zero on lambda grid 0,.001,...,.06 0.021
```

At nominal `lambda=0.021`, the exact interval at the serialized author-style
radius is `[-0.5206344, 0.0044664]`, so the Koo--Guo robust point is zero. The
tuning scale and uncertainty source differ from the ADH loss-gap envelope, so
this is not a tolerance reparameterization.

## Conclusion

The closest-prior sets and the ADH graph are different mathematical objects.
The disconnected exact example proves a qualitative distinction. The public
Basque calculation supplies no dramatic empirical separation: the
Ferguson--Ross and ADH endpoints nearly agree at the reported tolerances, while
the Koo--Guo class reaches zero on its own, non-comparable radius scale. The
paper therefore claims neither a first robustness set nor universal empirical
fragility; its contribution is the complete classical-ADH graph and certified
effect image.

## Replay

```bash
.work/venv/bin/python scripts/verify_prior_comparison.py
```

The verifier reconstructs the frozen design, recomputes the one-level QPs and
moment-class LPs, checks proposed supports with exact rational KKT conditions,
and compares the complete result object byte-for-byte after JSON parsing.
