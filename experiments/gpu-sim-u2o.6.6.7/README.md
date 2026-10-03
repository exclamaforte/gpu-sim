# gpu-sim-u2o.6.6.7 — a comparison must state the baseline's spread before calling an effect

Two measured incidents forced this bead:

1. A **+3.9% single-replicate reading** that a three-replicate repeat showed was
   inside the noise (`gpu-sim-u2o.6.6.1` rework 3).
2. The **v6 anchor's own cross-run spread** — same code, same configs, different
   run: median 1.018, p90 1.235, max 1.559 over 106 comparable cells
   (`gpu-sim-u2o.6.6.2` `variance_probe.json`).

A single-run ratio against a baseline therefore resolves *nothing* below that
baseline's spread, and nothing in the workflow forced a comparison to say so.

## What was added, and where

`epic_u2o/measurement_stats.py` — the right home, since that module is already
"repeated-trial statistics" and this is the other half of it. Everything above
the new section describes **within** one run; this describes **between** runs.

| | |
|---|---|
| `SpreadRecord` | a baseline's measured spread, versioned **with** the baseline (`spread_version`), carrying `n_runs`, `n_cells` and the job ids that produced it |
| `spread_from_runs(runs, ...)` | measures it from ≥2 runs, joined on the config label the measurement itself emits |
| `verdict_for_ratio(ratio, spread)` | `effect_slower` / `effect_faster` / `inconclusive` / `no_spread_recorded` |

## Three refusals, each of which would have inverted the point

- **A single run is REFUSED, not defaulted to zero.** A one-run spread is zero by
  construction; returning `0.0` would give every comparison a zero-width band and
  make *any* difference read as an effect. This is the module's central refusal.
- **A config present in only some runs is excluded, not zeroed.** Counting it as
  unchanged would manufacture cells sitting exactly on 1.0 and pull the median
  toward "no movement" — flattering whatever the spread is used to justify.
- **No spread recorded yields `no_spread_recorded`**, deliberately *not*
  `inconclusive` and *not* an effect. "I cannot tell" and "it passed" must not
  share a value.

## The guard caught my own modelling error

The first implementation used `max/median` across runs. It is one-sided (always
≥ 1, silently discarding the fast side) and it compressed the tail — it reported
a p90 of **1.11** where the committed probe reports **1.235** for the same data.

`anchor_band_is_wide` failed, which is the only reason I found out. Switching to
the **`max/min` envelope** reproduces the probe exactly: overall p90 **1.235**,
reduce 1.1012, gemm 1.0133.

The same change then broke a test that hardcoded `1.05` as "inside the band".
That was the correct refusal and a bad test — a typed constant that happened to
be inside one particular band. Both probe ratios in that suite are now **derived
from the band**, so they state the rule rather than a number.

## What a comparison may now conclude, against the real anchor

```
ratio 1.02 -> inconclusive      ratio 1.25 -> effect_slower
ratio 1.04 -> inconclusive      ratio 1.50 -> effect_slower
ratio 1.10 -> inconclusive
```

A **4% single-run difference is inconclusive** against this anchor. That is the
bead's whole claim, asserted in code against the committed spread.

## The caution it must not become

A check that only ever reports disagreement is indistinguishable from one whose
accounting was simply wrong. The v6 anchor's p90 is 1.235, so a hard rule like
*"any ratio under 1.2x is inconclusive"* would retire most of this lab's own
A/B results. The band is **per-baseline and measured**, and the suite asserts
both directions — including `test_ratio_above_the_band_is_a_real_slower_effect`,
whose docstring notes that if it ever fails, the module is a "nothing matters"
machine.

## Artifacts

`run.sh` recomputes `metrics.json` and runs the suite, nonzero on failure.
CPU only, no GPU, no `ts` job.

```
experiments/gpu-sim-u2o.6.6.7/measure_spread.py   rebuilds the spread from the
                                                  family files and cross-checks it
                                                  against the committed probe
experiments/gpu-sim-u2o.6.6.7/{run.sh,metrics.json,output.log,job.json}
tests/test_replicate_discipline.py                13 tests, fixture + real anchor
```