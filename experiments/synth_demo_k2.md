# Public `Synth` demo: declared two-predictor slice

**Status: globally certified exact rational result conditional on the declared
decimal export. This is an artificial public demo and not a canonical empirical
replication.**

## Design and provenance

The source is CRAN package `Synth` version `1.1-10`, object `synth.data`, under
`GPL (>=2)`. The downloaded source tarball had SHA-256
`41a33d869b88f218d6c7a0a88f3a1e7b2730cb83850695be970d9b2c8effef93`.
The frozen transformed data and full provenance are in
`instances/empirical/synth_demo_k2.json`.
`scripts/extract_synth_demo_k2.R` replays the rounding from the package's
`synth.data.RData` object.

The official package example uses three regular predictors plus three special
outcome predictors. The certified solver currently requires two predictors, so
this experiment deliberately fixes:

- treated unit 7 and the example's six controls `29,2,13,17,32,38`;
- predictors `X1` and `X2`, averaged over 1984--1989;
- validation outcome `Y` over 1984--1990;
- the 1991 `Y` gap as the scalar post-treatment effect.

It therefore changes the official example's fixed predictor list and is not a
replication of the package estimate. Predictor means were exported from R and
rounded to nine decimal places; documented one-decimal `Y` values were retained
at that resolution. Every result below is exact for those frozen rational
numbers. Positive row scaling used by common SCM implementations does not
change the union of lower solutions when all `V` in the closed simplex are
considered; it only reparameterizes `V`.

## Certified result

The predictor hull has four vertices and the continuous-`V` lower graph lifts
to five rational donor-weight polytopes. Exact support enumeration gives

```text
L_star                  = 4.289103410373433
exact optimum effect    = 0.783651573894741
full attainable effects = [-36.7941977121, 16.0797726704]
```

The full attainable range ignores validation fit and is included only as a
geometry check. Certified near-optimal envelopes are:

| outer tolerance | absolute epsilon | sharp effect envelope (box outward-rounded) |
|---|---:|---:|
| exact optimum | 0 | `[0.783651574, 0.783651574]` |
| 1% relative | 0.0428910341 | `[0.407735131, 1.132687113]` |
| sign breakdown | 0.1863934343 | lower endpoint within `[-0.000000115, 0.000000445]`; upper within `[1.530745829, 1.530746741]` |
| 5% relative | 0.2144551705 | `[-0.056922757, 1.590754319]` |

Each displayed nonexact endpoint is enclosed in a rational interval of width at
most `10^-6`. The sign-spanning and attained-zero thresholds coincide exactly:

```text
epsilon_span = epsilon_zero = 0.186393434265964
relative gap                         = 4.34574353733605%
```

Thus a 1% relative outer relaxation creates substantial magnitude ambiguity but
does not reverse the sign; the first sign reversal/attained zero needs about a
4.35% loss increase. This is evidence that the diagnostic can be nontrivial in
a public SCM-shaped dataset, but it is not evidence of empirical prevalence or
policy relevance because the panel is artificial and the predictor list was
restricted for certification.

## Replay

`certificates/synth_demo_k2.json` stores exact fractions and three endpoint
boxes. `scripts/verify_synth_demo_k2.py` independently replays each box using
four exact cutoff-feasibility QPs: infeasible/feasible cutoffs around the lower
endpoint and feasible/infeasible cutoffs around the upper endpoint.

Run:

```bash
python3 scripts/verify_synth_demo_k2.py
```

No floating-point optimizer or acceptance tolerance enters the certificate.
