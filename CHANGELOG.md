# Changelog

## 0.4.3 — 2026-08-28

- Make the release manifest fail closed on unlisted, stale, duplicate, or
  symbolic-link files.
- Clarify the core-test count and optional-solver skips in the README.

## 0.4.2 — 2026-08-28

- Ignore runtime version metadata when replaying the exact Proposition 99
  certificate, while preserving it as provenance.
- Refresh GitHub Actions to current Node.js-compatible major releases.
- Pin CI actions to immutable upstream commits, with version comments retained
  for reviewability.
- Make manifest verification reject unlisted, stale, duplicate, or symbolic-link
  release files.

## 0.4.1 — 2026-08-28

- Correct manifest generation and verification to exclude interpreter-generated
  bytecode while retaining fail-closed checks for every release file.

## 0.4.0 — 2026-08-28

- Public research release of the paper, implementation, exact fixtures, and
  replayable certificates.
- Finite rational-polytope theorem for the complete classical SCM solution
  graph.
- Exact small-design solvers and scalable rational KKT support certification.
- Frozen Basque and Proposition 99 public-data checks.
