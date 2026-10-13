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

## The statistic had to be able to say "faster"

Two modelling errors, both found by guards rather than by reading.

**First: `max/median`.** One-sided — always ≥ 1 — so it compressed the tail and
reported p90 1.11 where the probe reports 1.235. `anchor_band_is_wide` caught it.
Switching to `max/min` fixed the magnitude but created a worse defect:

**Second: `max/min`.** Also one-sided, and it cannot represent a config that got
**faster**. The anchor's fastest cell moved to **0.8523 of v6**, and `max/min`
reported that as **1.1745** — the wrong *direction*, on the one artifact whose
purpose is to stop exactly that. A field named `ratio_quantiles.min` that cannot
go below 1 while the data contains 0.8523 is a number that means something else.

The statistic is now **signed and baseline-relative**: each later run contributes
`v_i / v_baseline`. That is what the probe computes, so the two now agree
**exactly** — `min 0.8523, p10 0.9408, median 1.0181, p90 1.235, max 1.5587`,
identical on every quantile — and `p10 < 1`, so `effect_faster` is judged against
real fast-side noise instead of being unreachable.

The old synthetic fixtures all moved **one way**, which is precisely how this hid.
The suite now also runs against **real anchor data with cells that moved both
ways**: the fast side is measurable, a ratio below the measured `p10` is a real
`effect_faster`, and a mild speedup *inside* the band is still `inconclusive`.

`ratios_per_config` is carried too: with `n_runs == 2` each config contributes
exactly **one** ratio, so these quantiles are quantiles over single samples per
config, not over a distribution. That limit was implicit before.

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