# Compact audit of an additional public benchmark

Date checked: 2026-08-27.

## Search rule

The screen sought one fully public, documented classical-SCM design that could
add a qualitative result not already supplied by the Basque and California
analyses. Selection was based on provenance and behavior, not on whether a case
looked fragile in a local run. The search was intentionally compact.

## Candidates considered

1. **German reunification (Abadie, Diamond, and Hainmueller, 2015).** This is a
   canonical 16-donor application with public annual data and six published
   predictor categories. The journal article states that its original
   procedure selects predictor weights on a 1971--1980 training design by GDP
   fit in 1981--1990, then refits donor weights on 1981--1990 predictor data.
   That extra refit is outside the one-graph estimand implemented in this
   project. I therefore screened a declared, paper-aligned standard-fitting
   variant: the six published 1981--1990 predictor summaries, GDP in 1960--1989
   as the outer criterion, and mean GDP in 1991--2003 as the effect summary.
2. **The exact two-stage German cross-validation replication.** Not computed.
   Treating training donor weights, the selected metric, and refitted donor
   weights as a coupled graph would enlarge the estimand and solver. It is not
   justified merely to obtain another empirical column.
3. **Brexit and later public package examples.** Not computed after the
   canonical German screen was already sign-robust. Their design and licensing
   trails were less direct, so they offered lower value for a compact audit.

## Provenance and reproduction

The screened data copy is distributed in the Apache-2.0 repository
[`OscarEngelbrektson/SyntheticControlMethods`](https://github.com/OscarEngelbrektson/SyntheticControlMethods)
at commit `3f496b36ed46c4e5c1e08ce6e903013e6eeb29df`. The source CSV has SHA-256
`992153e4159b0af2c1ce730e61076704dc5b04c2a45f8d0dedcb9b8bae79c063`.
The repository notebook explicitly describes itself as a worked example rather
than an exact replication, so it is used only as a legally redistributable data
route. The specification comes from the published article.

Reproduce the local input and exact screen with:

```bash
curl -L https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/3f496b36ed46c4e5c1e08ce6e903013e6eeb29df/examples/datasets/german_reunification.csv \
  -o .work/public-benchmarks/germany/german_reunification.csv
.work/venv/bin/python scripts/audit_germany_candidate.py \
  .work/public-benchmarks/germany/german_reunification.csv \
  certificates/germany_candidate_audit.json
```

All source decimals are parsed as exact rationals. GDP validation rows are
divided by exactly 10,000 for numerical discovery; this common scaling does not
change the donor optimizer. SciPy proposes supports, and rational KKT checks
accept them.

## Result and decision

The exact donor-simplex/coordinate-branch sandwich gives:

```text
global validation MSE       [0.00003702035245, 0.00003854988242]
relative loss-box width     4.1316%
globally optimal effect     [-1717.0340, -1256.3704] source GDP units
relative zero-effect gap    [560.3297%, 616.2846%]
```

The loss sandwich is too wide to supply inner endpoint bounds at 1%, but the
donor-simplex outer relaxation alone certifies that every full-graph endpoint
remains negative:

| Relative gap | Lower endpoint outer box | Upper endpoint outer box |
| ---: | ---: | ---: |
| 1% | `[-1758.7272,-1758.7190]` | `[-1239.8386,-1239.8332]` |
| 5% | `[-1894.3618,-1894.3537]` | `[-1190.0705,-1190.0651]` |

This design is clearly sign-robust and does not add a qualitatively distinct
empirical result. It is therefore recorded but omitted from the manuscript.
The compact screen found no honest public fragility case; the exact synthetic
examples carry the possibility result, while the two retained public designs
show that the method can certify robustness.
