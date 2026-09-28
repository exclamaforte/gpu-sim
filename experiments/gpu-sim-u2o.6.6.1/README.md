# gpu-sim-u2o.6.6.1 — the noise floor of the real measurement path

**`measurement_version: 1` — FROZEN.** Authoritative run: ts job
`20260927t162535-cb00ea78` (2026-09-27). Any later change to these numbers
must be a `noise_floor_v2` with a stated reason, never an in-place edit: the
6.1.1 closing run is diffed against this file, and an unstamped successor
makes that diff wrong invisibly.

One short `ts` job, read-only, fixed hand-written reference kernel,
`preflight_check` bypassed on purpose. Full data in `metrics.json` (including
every raw per-trial vector), transcript in `output.log`, job ids in `job.json`.

**All figures below are regenerated from the final artifact
(`metrics.json`, job `20260927t162535-cb00ea78`).** An earlier draft of this
file quoted a previous run's numbers; the reviewer caught that and it is fixed.

## Negative control (acceptance criterion 4)

The same binary with `CUDA_VISIBLE_DEVICES=` blanked:

* **returncode 2**, stderr `{"error": "...:88 no CUDA-capable device is detected"}`
* positive control — the same invocation with the device visible: **rc 0**

So the measurements came from a real device, and the binary would have said so
loudly if it had not.

## Headline: the ~2.6 µs is launch latency, not kernel time

Decomposition (µs, median of 3 independent processes × 200 trials):

| n | reported (single) | irreducible launch | measurement overhead | kernel work | **residual** | kernel share |
|---|---|---|---|---|---|---|
| 1 024 | 2.688 | 1.308 | 0.836 | 0.046 | +0.498 (+18.5%) | 1.7% |
| 16 384 | 2.656 | 1.330 | 0.878 | 0.028 | +0.420 (+15.8%) | 1.1% |
| 262 144 | 2.752 | 1.365 | 1.483 | −0.016 | −0.080 (−2.9%) | −0.6% |
| 4 194 304 | 10.784 | 8.104 | 1.080 | 0.235 | +1.365 (+12.7%) | 2.2% |
| 67 108 864 | 68.880 | 31.011 | 2.093 | 40.095 | −4.319 (−6.3%) | 58.2% |

**These four components are measured in different modes and are not a
partition** — that is why the residual column exists and why it is not small.
`reported` is single-mode scale; `launch` and `kernel` are batched-mode
differences; `overhead` is single-minus-batched. The `kernel work` figure
itself is sound, because it comes from the internally consistent batched
difference (scale − empty at the same size, same mode).

Method: an **empty** kernel and the reference `out[i]=in[i]*alpha` kernel, both
timed two ways — one launch per event region (`single`, what
`kernel_driver.synthesize()` does and therefore what every stage-1 measurement
so far has measured) and 100 launches per event region (`batched`).

Empty kernel alone: **2.144 µs** single, **1.308 µs** batched.

So of the ~2.6 µs the dataset records: ~1.3 µs is irreducible dispatch,
0.8–1.5 µs is the cost of measuring a single launch in isolation, and tens of
nanoseconds is the kernel. **The fix is measurement design, not better
statistics** — larger problem sizes, or amortised timing over repeated
launches inside one event region.

## Crossover

Kernel work first exceeds 10% of the launch cost at **n = 2²⁶**, where it is
58% of the reported time (and it is the first size where kernel work exceeds
1 µs in absolute terms). Caveat recorded in `metrics.json`: `grid.x` is capped
at 65535, so at n = 2²⁶ with block 256 the kernel only covers 16 777 216 of
the 67 108 864 elements — a compute-bound datapoint, not a full-size one.

## Minimum detectable difference (2σ, two-sample)

**The top-level MDD block is the curve for exactly one cell** and is **not a
bound on anything**: `empty/single/n=1024`, from ts job
`20260927t162535-cb00ea78`. The **worst case is `empty/single/n=16384` at
76.3% (N=10)** — not the smallest n. MDD is **non-monotone** in problem size
across those cells: 51.4 → 76.3 → 27.8 → 11.2 → 2.3 % at N=10 for
n = 1024, 16384, 262144, 4194304, 67108864.

MDD tracks within-process noise, as it must (it is built from within-process
std and the median): Pearson **r(MDD@N=10, within-process cv) = +0.929 over
all 20 cells**; +0.950 over the 5 empty single-mode cells *(subset — not a
headline)*. The across-invocation column tracks about equally well and may
also be used: **r(MDD@N=10, across-invocation CV) = +0.929 over all 20
cells** (from `cells[].across_invocations_spread_pct`).

**A cell's median is the median of its three per-invocation medians**, and
those span 27.6–37.7% in this regime (n=16384: 1.472 / 2.208 / 2.304 µs;
n=1024: 2.144 / 2.144 / 1.552 µs). Every one of those is a real measurement
of that cell. **No single-run figure from the sub-10 µs regime may be quoted
as a cell's runtime**; the three-invocation median is the *minimum* defensible
estimator, not a refinement.

r(median, within-process cv) = **−0.414 over all 20 cells**: low medians and
high cv co-occur. The sign survives; the magnitude is not established and no
median here is called systematically biased on this evidence.

*Rule applied throughout this file: every figure names the cell it came
from, and every correlation names the population it was computed over.*

| trials | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| MDD % | 51.4 | 36.4 | 23.0 | 16.3 | 11.5 |

**20 single-launch trials cannot resolve a 10% difference** (they resolve
~36%). Trials needed for 10%:

| n | single-launch | batched |
|---|---|---|
| 1 024 | 59 | 7 |
| 16 384 | 110 | 6 |
| 262 144 | 48 | 28 |
| 4 194 304 | 5 | 0 |
| 67 108 864 | 1 | 0 |

The batched column is the actionable one. Note the honest exception the
reviewer asked me to state: batched is not uniformly excellent — n = 262 144
needs 28 trials, not the ~6 of its neighbours, and the across-invocation
spread of batched medians spans **0.01%–4.34%** in this run (it was
0.02%–12.57% in the previous run, which is itself the point: the spread is
run-dependent, so a fixed claim about it is a fixed claim about luck). Single
mode reaches 37.7% across-invocation spread. So: batched timing is the right
design change *and* its benefit is smaller than one clean run suggests.

## Within-process vs across-invocation spread

Three independent *processes* per cell. The artifact's `cells[].cv` field
(the cell's median across invocations of the per-invocation cv) ranges
**0.002–0.441**; the *different* field `per_invocation[0].cv` ranges
0.002–0.547 over the same cells. Both are recorded in the artifact under
`noise_floor.within_process_cv_range` so they cannot be conflated again. Either
way, within-process cv badly understates the disagreement between processes:
single-mode medians
differ by up to **37.7%** between processes of the same config. That is why
three of four stage-1 kernels failed `measurement_stats`' own `max_cv=0.15`
gate — and why judging reproducibility from within-process cv alone would be
the wrong call.

## Telemetry (read-only, baseline for gpu-sim-u2o.6.4)

| | before | after |
|---|---|---|
| clocks.sm | 210 MHz | 2917 MHz (max 3105) |
| temperature.gpu | 38 °C | 48 °C |
| clocks_throttle_reasons.active | 0x0 | 0x0 |
| power.draw | ~11 W | ~380 W (limit 575 W) |
| persistence_mode | Disabled | Disabled |

No throttling, so these numbers are not clock-drift artefacts. The part idles
at ~210 MHz and needs sustained work to reach boost — a real risk for short
kernels measured back to back, and one more reason the driver's 20 warmup
launches are not optional.

## Constraints honoured

`ts` queue only; `preflight_check` bypassed (it executes on the real GPU per
the 6.1.1 ruling, and an unaccounted second execution is exactly what this
experiment must not inherit) — stated at the top of `output.log`; fixed
reference kernel, nothing generated or untrusted; no sudo, no
`nvidia-smi -lgc`, no persistence-mode change, read-only queries only; whole
job 15.6 s of GPU time.
