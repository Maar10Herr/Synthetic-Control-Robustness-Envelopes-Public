# Exact two-predictor solver scaling probe

**Status: descriptive timing only, not a complexity theorem.**

Command:

```bash
python3 scripts/benchmark_two_predictor.py
```

The script uses standard-library exact rational arithmetic, seed `20260826`,
integer data in `[-5,5]`, two validation periods, and the first full-dimensional
predictor hull found for each donor count. A 2026-08-26 run on the current host
gave:

| J | hull vertices | graph branches | seconds |
|---:|---:|---:|---:|
| 3 | 3 | 5 | 0.1332 |
| 4 | 3 | 4 | 0.1429 |
| 5 | 4 | 7 | 0.7152 |
| 6 | 5 | 7 | 5.1516 |
| 7 | 7 | 7 | 11.0146 |

Timings are hardware- and instance-dependent, but the growth confirms the
known implementation limitation: support enumeration is exponential. The
six-control public demo is still practical (about four seconds for graph,
optimum, and breakdown construction), while the 16-control Basque design is not
an honest target for this exact implementation. Positive-gap bisection repeats
cutoff QPs and is correspondingly slower; saved endpoint boxes should be
replayed at their four boundary cutoffs rather than recomputed from scratch.

