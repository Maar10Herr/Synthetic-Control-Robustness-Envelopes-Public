# Synthetic Control Robustness Envelopes

[![CI](https://github.com/Maar10Herr/Synthetic-Control-Robustness-Envelopes-Public/actions/workflows/ci.yml/badge.svg)](https://github.com/Maar10Herr/Synthetic-Control-Robustness-Envelopes-Public/actions/workflows/ci.yml)
[![License: GPL v3+](https://img.shields.io/badge/License-GPL_v3%2B-blue.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--8721--6588-A6CE39.svg)](https://orcid.org/0009-0005-8721-6588)

Exact and certified optimization-robustness envelopes for classical
Abadie–Diamond–Hainmueller synthetic control.

> [!IMPORTANT]
> **Research implementation.** Scope: variation across globally optimal or
> nearly optimal solutions of a declared fitting problem. Sampling uncertainty,
> causal identification, and production synthetic-control workflows require
> separate methods and established software.

**[Read the paper](paper/Herrmann_Sharp_SCM_Robustness_Envelopes.pdf)** ·
**[Inspect the certificates](certificates/)** ·
**[Download the latest release](https://github.com/Maar10Herr/Synthetic-Control-Robustness-Envelopes-Public/releases/latest)**

## Result

For predictor weights \(v\) and donor weights \(w\), let \(G\) be the complete
lower-level solution graph of the classical nested fit and let

```math
\mathcal A_\varepsilon=\{(v,w)\in G:L(w)\le L^\star+\varepsilon\}.
```

The paper proves that the donor-weight projection of \(G\) is a finite union of
rational polytopes. Consequently, for any linear post-treatment effect
\(\tau(w)\), the sharp optimization-robustness envelope

```math
\left[\min_{(v,w)\in\mathcal A_\varepsilon}\tau(w),
      \max_{(v,w)\in\mathcal A_\varepsilon}\tau(w)\right]
```

reduces to finitely many linear or convex quadratic programs. The theory covers
nonunique inner solutions, boundary predictor weights, disconnected attainable
effect sets, and exact sign- and zero-breakdown thresholds.

## Verify

Core verification uses only the Python standard library:

```sh
make verify-core
```

This runs 56 tests (50 pass with the standard library; six optional SciPy
checks skip when scalable dependencies are absent) and replays five rational
certificates, including the disconnected sign-flip construction and the
continuous-predictor-weight examples.

Scalable public-data checks use pinned NumPy and SciPy versions in a local
environment:

```sh
uv sync --frozen --extra scalable
make verify-scalable
```

The scalable path replays the Basque and Proposition 99 certificates and the
comparison with two prior convex feasible-set approaches. Numerical solvers
propose supports; exact rational KKT checks accept or reject them.

## Certificate architecture

```mermaid
flowchart LR
    A[Declared SCM design] --> B[Discovery optimizer]
    B --> C[Candidate supports and endpoints]
    C --> D[Exact rational KKT replay]
    D --> E[Saved certificate]
    E --> F[Independent verifier]
    F --> G[Sharp or outward-certified envelope]
```

| Evidence | Design | Certified conclusion |
|---|---|---|
| Exact synthetic fixture | 3 donors | Two global optima; attainable effects \(-1/3\) and \(1/3\) |
| Near-optimal fixture | 3 donors | Opposite sign enters at loss gap \(1/1000\) |
| Synth demonstration slice | 2 predictors, 6 controls | Exact envelope and sign-breakdown threshold |
| Basque design | 13 predictors, 16 controls | Global optimum and 1%/5% envelopes remain negative |
| Proposition 99 design | 7 predictors, 38 controls | Outward-certified global and 1%/5% envelopes remain negative |

“Sharp” is used only where lower and upper bounds coincide. The Proposition 99
result is reported as an outward-certified box.

## Trust boundary

- `src/` contains theorem-specific exact and scalable algorithms.
- `certificates/` contains accepted proof objects, not discovery logs.
- `scripts/verify_*.py` reconstruct acceptance conditions from frozen inputs.
- `instances/` records exact inputs and transformations.
- `experiments/` records scope, provenance, hashes, and interpretation limits.
- Optional R scripts reconstruct package-derived fixtures and conventional
  `Synth` comparisons without installing packages globally.

No API key, database, private dataset, manuscript source, or working research
note is included. The release verifier does not require network access.

## Prior work and attribution

The fitting problem is the classical synthetic-control method of Abadie and
Gardeazabal and Abadie, Diamond, and Hainmueller. Ferguson and Ross and Koo and
Guo motivate nearby robustness feasible sets. Malo et al. establish the
California corner solution and document local-optimizer failure on the
source-precision design. Full citations and the precise distinction between
those contributions and this project appear in the paper.

The public fixtures are reproducible transformations of data distributed with
CRAN `Synth` and `tidysynth`; upstream authors, versions, licences, and archive
hashes are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
corresponding experiment records.

## Repository map

```text
paper/          version-0.4 preprint PDF
src/            exact and scalable implementation
tests/          unit, adversarial, and tamper-oriented checks
certificates/   frozen accepted proof objects
instances/      exact synthetic and public-data fixtures
scripts/        independent replay and provenance tools
experiments/    public benchmark definitions and limitations
```

## Citation and licence

Use [CITATION.cff](CITATION.cff) and cite the specific release. Author:
[Maarten Linus Herrmann](https://orcid.org/0009-0005-8721-6588), Erasmus
School of Economics, Erasmus University Rotterdam.

Original software is licensed under [GPL-3.0-or-later](LICENSE). The paper is
copyright © 2026 Maarten Linus Herrmann; all rights reserved. Package-derived
fixtures retain their upstream terms.
