# Full Basque predictor design

## Scope and provenance

This experiment fixes the treated Basque region (`regionno=17`), the 16
controls, the 1960--1969 GDP validation outcomes, and all 13 transformed
predictors in the official `Synth` package example. The post-treatment scalar
is the declared mean treated-minus-synthetic GDP per capita over 1970--1975.
The package example plots the full path, so this is a full-predictor-design
robustness analysis, not a replication of one published scalar estimate.

The source is CRAN `Synth` 1.1-10, SHA-256
`41a33d869b88f218d6c7a0a88f3a1e7b2730cb83850695be970d9b2c8effef93`,
licensed GPL (>= 2). `scripts/extract_basque_full.R` reproduces the 13-predictor
fixture from the package `basque.RData`. Exact claims below are conditional on
the documented six-decimal export; source floating-point and rounding
uncertainty are separate from mathematical `epsilon`.

## Certificate strategy

The exact arbitrary-`K` graph theorem makes the continuous-`V` lower graph a
finite union of rational polytopes, but direct enumeration is not credible
here. The 13-dimensional donor hull has affine rank 13, all 16 donors as
vertices, 136 facets, and 58,671 nonempty faces. A naive face-by-orthant pass
would inspect 480,624,640 pairs.

Instead, the certificate sandwiches the bilevel graph between two exact
bounds:

1. minimizing validation loss over the entire donor simplex is a global lower
   bound because that simplex contains every inner-attainable donor vector;
2. an exact reconstructed predictor-weight vector and exact inner QP KKT
   certificate prove that a proposed donor vector belongs to the bilevel graph.

When both bounds agree, the result is globally sharp without enumerating the
graph. SciPy proposes active supports; rational primal-dual checks alone accept
them.

## Certified results

Conditional on the frozen fixture,

```text
global validation loss L*      = 0.0041263737767375215
exact global effect            = -0.26820046745838366
absolute sign/zero gap         = 0.004513081117417488
relative sign/zero gap         = 109.37160232211713%
first zero-effect loss         = 0.00863945489415501
```

The donor-simplex optimizer is inner-attainable with predictor support
`{school.illit, mean GDP 1960--1969, construction}`. Its donor support is
regions 5, 14, and 18. The effect is constant over the complete
donor-simplex validation-optimizer polytope, so the exact bilevel envelope at
`epsilon=0` is the singleton shown above.

The first zero-effect donor vector is inner-attainable with predictor support
`{school.illit, mean GDP 1960--1969, non-traded services}` and donor support
regions 5, 10, and 14. Therefore the donor-simplex lower bound closes exactly:
the sign-spanning and attained-zero thresholds coincide.

Positive-gap endpoints are certified by an outer donor-simplex relaxation and
inner bilevel witnesses. The table gives outward rational boxes, displayed in
decimal form; an endpoint is guaranteed to lie inside its box.

| Relative gap | Lower endpoint box | Upper endpoint box |
| --- | --- | --- |
| 1% | `[-0.294830107, -0.294772001]` | `[-0.239581035, -0.239580058]` |
| 5% | `[-0.327425356, -0.325497806]` | `[-0.187317253, -0.187316276]` |

The upper boxes are narrower than `1e-6`. For the lower side, the feasible
inner bound comes from the exact two-predictor subgraph using primary schooling
and mean pre-treatment GDP; the remaining widths are about `5.81e-5` at 1%
and `1.93e-3` at 5%. Both entire endpoint boxes are negative, so the sign
conclusion is certified despite the unresolved point within each box.

## Conventional local `Synth` comparison

`scripts/run_basque_full_synth.R` runs the official `dataprep` transformation
and default local `synth()` procedure. With R 4.6.1, Synth 1.1.10, kernlab
0.9-33, optimx 2025.4.9, and rgenoud 5.9-0.11, two runs produced the same saved
output. This is numerical procedure output, not a global certificate.

On the source package's full-precision data, it reports

```text
loss.v                         = 0.008864749086073652
1970--1975 effect              = 0.0003080552308540874
```

Evaluating those same donor weights on the exact six-decimal fixture gives

```text
validation loss                = 0.008864726348532911
1970--1975 effect              = 0.0003083613218679560
relative loss above global     = 114.83091033846483%
```

Thus the conventional local run lands just beyond the exact 109.3716%
sign-breakdown threshold and reports an approximately zero positive effect,
while the globally optimal bilevel fit has a materially negative effect. This
is evidence of optimizer/procedure sensitivity in this fixed design; it is not
sampling uncertainty, a confidence interval, or a claim that every `Synth`
run behaves this way.

## Closest-prior feasible-set comparison

The closest-prior feasible-set comparison is reported in
`experiments/prior_feasible_set_comparison.md`. On this frozen Basque outer
design, the Ferguson--Ross outcome-only class agrees with the ADH endpoints to
reported precision at matched 1%/5% squared-loss gaps. The Koo--Guo
author-default-style moment class remains negative at nominal `lambda=0` and
first spans zero on its 0.001 grid at `lambda=0.021`.

## Certificate replay

With the repository-local optional Python environment:

```bash
.work/venv/bin/python scripts/verify_basque_full.py
```

This recomputes the exact hull counts, donor-simplex optimum and effect range,
sign/zero threshold, both inner-attainability witnesses, and the 1% and 5%
endpoint boxes. `scripts/check_optional_scalable.sh` also re-extracts both
Basque fixtures byte-for-byte when the source `RData` is present and replays the
saved local `Synth` output when its repository-local R dependencies are
available.
