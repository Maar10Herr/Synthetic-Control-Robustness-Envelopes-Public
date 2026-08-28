# Basque Country: real-data two-predictor slice

**Status: globally certified for a declared altered design; not a replication
of Abadie--Gardeazabal or the official `Synth` example.**

## Scope and provenance

The panel is the real Basque Country dataset distributed in CRAN package
`Synth` `1.1-10`, originally associated with Abadie and Gardeazabal (2003).
Source URL, tarball hash, GPL license, and six-decimal transformation are frozen
in `instances/empirical/basque_k2_metadata.json`; the transformed panel is
`instances/empirical/basque_k2.csv`, and `scripts/extract_basque_k2.R` replays
the export.

The official package example uses 13 transformed predictors. To remain inside
the proved continuous-`V`, `K=2` class, this diagnostic fixes only:

- mean GDP per capita over 1960--1969;
- mean investment over 1964--1969;
- GDP per capita in 1960--1969 as the outer validation path;
- mean GDP per capita in 1970--1975 as the post-treatment scalar;
- treated region 17 and all 16 official donor regions.

This is a real-data stress test with the canonical treated unit/donor pool and
dates, but its altered predictor list means its effect must not be compared as
the published estimate.

## Certified result

The predictor hull has six vertices and the full continuous-`V` graph has seven
rational polytope branches. Exact values for the six-decimal fixture are:

```text
L_star               = 0.00412727504107716
exact optimum effect = -0.269280469152580
full graph effects    = [-0.871072000000000, 2.65571441808437]
```

The full graph can produce either sign, but near-optimal validation fit does
not:

| outer tolerance | absolute epsilon | certified sharp envelope |
|---|---:|---:|
| exact | 0 | `[-0.269280469, -0.269280469]` |
| 1% relative | 0.00004127275 | `[-0.295170264, -0.241343004]` |
| 5% relative | 0.00020636375 | `[-0.325592251, -0.189907552]` |

The first envelope sign span and the first attained zero coincide at

```text
epsilon_span = epsilon_zero = 0.00581536560021365
relative gap                         = 140.900849648632%
```

This case pushes against the adversarial examples: the magnitude envelope
widens visibly, but its sign is extremely robust relative to `L_star`. One
altered real-data design cannot establish prevalence in either direction.

The certificate also freezes sparse witnesses. One global optimum puts positive
weight only on regions 5, 14, and 18. The first zero-effect fit puts positive
weight only on regions 10, 14, and 18. Their exact rational weights are stored
in `certificates/basque_k2.json`; the replay checks branch membership, effect,
and validation loss rather than trusting rounded displays.

## Scalable discovery, exact acceptance

The optional backend uses SciPy/HiGHS/SLSQP only to propose LP/QP active
supports. Every proposed branch optimum is then reconstructed and checked in
standard-library exact rational arithmetic for primal equality/nonnegativity,
active stationarity, inactive reduced-cost/gradient feasibility, and exact
objective value. An incorrect numerical support is rejected.

On the current host, scalable construction including exact acceptance took
about `0.45s`. A one-time exhaustive rational support run took `161.6s` and
returned identical exact optimum, attainable range, and breakdown fractions.

Set up and replay the optional backend:

```bash
UV_CACHE_DIR=.work/uv-cache uv venv .work/venv
UV_CACHE_DIR=.work/uv-cache uv pip install --python .work/venv/bin/python -r requirements-scalable.txt
PYTHONPYCACHEPREFIX=.work/pycache .work/venv/bin/python scripts/verify_basque_k2_scalable.py
```

Dependencies are pinned to NumPy `2.5.2` and SciPy `1.16.1`; both use
BSD-family licenses. No numerical objective or tolerance is accepted as the
certificate.
