# gpu-sim-u2o.6.6.1 — the noise floor of the real measurement path

One short `ts` job, read-only, fixed hand-written reference kernel,
`preflight_check` bypassed on purpose. Full data in `metrics.json`, transcript
in `output.log`, job id in `job.json`.

## Headline: the ~2.6 µs is launch latency, not kernel time

**At the stage-1 configurations, 98.5% of the reported time is not kernel
work.** Measured decomposition (us, median of 3 independent process
invocations, 200 trials each):

| n | reported (single launch) | irreducible launch | measurement overhead | **kernel work** | kernel share |
|---|---|---|---|---|---|
| 1 024 | 2.624 | 1.319 | 0.425 | **0.039** | 1.5% |
| 16 384 | 2.656 | 1.322 | 0.902 | **0.064** | 2.4% |
| 262 144 | 2.672 | 1.347 | 1.373 | **0.030** | 1.1% |
| 4 194 304 | 10.848 | 8.087 | 1.577 | **0.250** | 2.3% |
| 67 108 864 | 69.104 | 31.015 | 2.361 | **40.079** | 58.0% |

Method: an **empty** kernel and the reference `out[i]=in[i]*alpha` kernel are
both timed two ways — one launch per event region (`single`, which is what
`kernel_driver.synthesize()` does, and therefore what every stage-1
measurement so far has measured) and 100 launches per event region
(`batched`). Subtracting the empty kernel from the real kernel at the same
problem size cancels the launch cost, leaving the kernel's own contribution.
The empty-kernel medians themselves:

* single: **1.744 µs** (this run), 2.272 µs (previous run) — noisy, see below
* batched: **1.319 µs**, cv 0.03–0.09, stable to ±0.02 µs across invocations

So of the ~2.6 µs the dataset has been recording: ~1.32 µs is the
irreducible cost of dispatching a kernel on this part, ~0.4–1.4 µs is the cost
of *measuring* a single launch in isolation, and a few tens of nanoseconds is
the kernel. **The fix is measurement design, not better statistics** —
larger problem sizes, or amortised timing over repeated launches inside one
event region.

## Crossover

Kernel work first exceeds 10% of the launch cost at **n = 2²⁶**, where it is
58% of the reported time. Below that, measuring these kernels measures launch
overhead. Caveat recorded in `metrics.json`: `grid.x` is capped at 65535 (the
classic launch bound), so at n = 2²⁶ with block 256 the kernel only covers
16 777 216 of the 67 108 864 elements. That cell is a compute-bound datapoint,
not a full-size one.

## Minimum detectable difference (2σ, two-sample)

Worst case — the empty kernel, i.e. the most overhead-dominated signal on the
box:

| trials | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| MDD % | 59.3 | 42.0 | 26.5 | 18.8 | 13.3 |

**20 single-launch trials cannot resolve a 10% difference** (they resolve
~42%). Trials needed for a 10% difference, per problem size:

| n | single-launch | batched |
|---|---|---|
| 1 024 | 68 | 16 |
| 16 384 | 83 | 40 |
| 262 144 | 52 | 7 |
| 4 194 304 | 3 | 0 |
| 67 108 864 | 1 | 0 |

The batched column is the actionable one: switching to amortised timing drops
the requirement to **7–40 trials** in the launch-bound regime and makes 20
trials sufficient everywhere.

## Within-process vs across-invocation spread

Three independent *processes* per cell, because the variance of the reported
median across invocations is what actually matters. The single-launch
measurements disagree between processes by up to **29%** of their median
(e.g. empty n=16384: 2.096 / 2.016 / 1.696 µs), while the batched medians
agree to **0.06–3.6%**. Within-process cv for single launches is 0.20–0.53 —
which is why three of four stage-1 kernels failed `measurement_stats`' own
`max_cv=0.15` gate. Understatement check: within-process cv alone would have
made these look better than they are; the across-invocation spread is the
honest figure.

## Telemetry (read-only, baseline for gpu-sim-u2o.6.4)

| | before | after |
|---|---|---|
| clocks.sm | 232 MHz | 2917 MHz (max 3105) |
| temperature.gpu | 37 °C | 48 °C |
| clocks_throttle_reasons.active | 0x0 | 0x0 |
| power.draw | 13.5 W | 380.9 W (limit 575 W) |
| persistence_mode | Disabled | Disabled |

No throttling occurred, so these numbers are not clock-drift artefacts. Note
the GPU idles at ~230 MHz and takes a full second or so of work to reach boost
— a real risk for short kernels measured back to back, and one more reason the
20 warmup launches in the driver are not optional.

## Constraints honoured

`ts` queue only; `preflight_check` bypassed (it executes on the real GPU per
the 6.1.1 ruling, and an unaccounted second execution is exactly what this
experiment must not inherit) — stated at the top of `output.log`; fixed
reference kernel, nothing generated or untrusted; no sudo, no
`nvidia-smi -lgc`, no persistence-mode change, read-only queries only; whole
job 15.7 s of GPU time.
