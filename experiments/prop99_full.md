# Full Proposition 99 predictor design

## Scope and provenance

This experiment uses the canonical California Proposition 99 panel distributed
with CRAN `tidysynth` 0.2.1. It follows that package README's full design:
California is treated in 1988, the donor pool contains 38 control states, seven
predictors are formed over the declared 1975--1988 windows, and cigarette sales
over 1970--1988 supply the outer validation loss. The scalar effect is the
declared treated-minus-synthetic mean cigarette sales over 1989--2000.

The source archive is MIT-licensed and has SHA-256
`046b416b437dd2f3d5610ac6a594db1c941c3cf9c208becfdfed0a76ce553281`.
Package documentation attributes the panel to the Abadie--Diamond--Hainmueller
2010 Proposition 99 data. `scripts/extract_prop99_full.R` reproduces the frozen
fixture; its SHA-256 is
`7e0ec4fbf09c8357aee35e36bd911dbcb6cf95b1814669c1681c35c6263736cf`.
Exact claims are conditional on this documented six-decimal rational export.
This is a full package-example predictor design, not a claim to reproduce every
implementation detail or scalar in the original paper.

Malo et al. (2024) already analyze this California design as a bilevel point-
optimization benchmark. On source-precision data they report a corner solution
placing all predictor weight on 1980 cigarette sales, with outer loss `2.74366`,
and show that `Synth` is suboptimal. The contribution here is not rediscovery of
that point. It is the frozen exact-arithmetic replay, a rigorous global loss/effect
sandwich, treatment-effect endpoint boxes, and sign/attained-zero bounds over the
complete admissible graph.

## Certificate strategy

Exhaustive enumeration of a seven-dimensional, 38-donor lower graph is not a
credible default. The certificate instead uses an exact two-sided sandwich.

The donor simplex contains every ADH lower-level solution, so its validation
minimum is a global lower bound. For the other side, put all predictor weight
on one coordinate, `V=e_k`. Whenever the treated coordinate can be matched,
every matching donor vector has zero inner loss and is therefore exactly
inner-optimal. Optimizing validation loss over each such coordinate-match
polytope gives valid ADH candidates without local optimization over `V`.

SciPy proposes active LP/QP supports. Exact rational primal, dual, and KKT
checks accept the results; numerical tolerances are not certificates. Because
the bounds do not coincide, the result is reported as outward intervals rather
than a sharp point.

## Certified results

Conditional on the frozen fixture,

```text
global validation MSE L*      in [2.7436616406, 2.7440893158]
relative width of loss box    = 0.0155878%
globally optimal effect       in [-19.570993, -19.456267]
first zero-effect loss        in [866.0589222, 892.4243608]
relative sign/zero gap        in [31,460.8868%, 32,426.7646%]
```

The upper loss witness places all predictor weight on 1980 cigarette sales and
has effect `-19.475440435`. The donor-simplex optimum supplies the lower loss
bound. For the zero threshold, the donor-simplex constrained optimum is the
lower bound and the mean-retail-price coordinate branch supplies an exact
zero-effect ADH witness for the upper bound. The same bracket contains both the
first nonnegative-effect and first attained-zero thresholds; it does not assert
that those two thresholds are equal.

Rigorous relative-gap endpoint boxes are:

| Relative gap | Lower endpoint box | Upper endpoint box |
| --- | --- | --- |
| 1% | `[-19.925121, -19.724084]` | `[-19.135304, -19.050582]` |
| 5% | `[-20.367147, -20.178410]` | `[-18.710069, -18.484600]` |

The outer side of each box uses the donor-simplex threshold based on the loss
upper bound. The inner side uses the best coordinate branch at the threshold
based on the loss lower bound. Both complete endpoint boxes remain negative.
This design is therefore very sign-robust under the declared ADH envelope.
Together with the negative Basque envelopes, it argues against treating
optimization fragility as automatic in every empirical design; exact synthetic
examples establish possibility without selecting an empirical failure.

## Conventional local `Synth` comparison

`scripts/run_prop99_full_synth.R` applies the same source transformation and
the default local `synth()` procedure. With the saved repository-local R
environment it reports:

```text
source loss.v                 = 3.2090782638
source effect                 = -18.7194758211
frozen-fixture loss.v         = 3.2086219108
frozen-fixture effect         = -18.7197252017
```

The frozen local loss is between 16.928% and 16.947% above the globally
bracketed loss. It is not inside the certified 1% or 5% sets, but its effect has
the same negative sign. This is numerical procedure output, not a global
certificate or a statistical comparison.

## Replay

With the repository-local optional Python environment:

```bash
.work/venv/bin/python scripts/verify_prop99_full.py
```

This reloads the positional CSV schema, recomputes the simplex and all seven
coordinate branches, replays every rational LP/QP certificate, and verifies
the global, zero-effect, and 1%/5% endpoint boxes. When the CRAN source object
and local R packages are present, `scripts/check_optional_scalable.sh` also
re-extracts the fixture and compares the conventional `Synth` output
byte-for-byte.
