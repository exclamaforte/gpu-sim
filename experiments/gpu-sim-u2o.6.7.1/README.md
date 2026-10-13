# gpu-sim-u2o.6.7.1 — batched worker: 10.5x, and it FAILS its own correctness gate

The P0. Batching the measurement worker is worth ~3,312x on paper; on the real
5090 it delivered **10.5x**, and by the bead's own
standard it is **not yet acceptable**, because it moved the numbers.

Authoritative run: ts `20260928t111322-e0720458`, 264 examples, protocol
unchanged.

## What was done

One CUDA context for the whole batch instead of two process spawns per example.
528 processes became **1**. Nothing
else changed: 40 timed launches per
example, 2 invocations x 20, 5 warmup each, and the estimator is 7.1's —
median over the two per-invocation medians, not a pooled median over 40 trials.

Real 7.1 specs are embedded verbatim and dispatched by parameter type, because
the 8 stage-1 kernels do not share a signature (`(in,out,n)`, `(a,b,out,n)`,
`(alpha,x,y,n)`). All 264 `kernel_hash` values match 7.1's dataset exactly, so
the reproduction check is row-for-row and not a comparison of two different
sweeps.

## Criterion 1 — throughput, with the gap ATTRIBUTED

| | ex/s | per hour |
|---|---|---|
| 7.1 single-run | 2.39 | 8,599 |
| **batched** | **25.05** | **90,169** |
| ceiling (UPPER BOUND) | 7,911 | — |

The ceiling is a **bound, not a target**, and the measured rate is not presented
as one. The gap, per example:

| component | ms | share |
|---|---|---|
| device work | 0.146 | 0.37% |
| **per-example telemetry sampling** | **39.21** | **98.2%** |
| residual (per-launch driver + Python) | 0.57 | 1.4% |
| total | 39.93 | |

**The dominant remaining cost is the per-example sample itself.** Spawning
`nvidia-smi` once per row costs 39 ms —
the same order as the process spawn it replaced. So batching the worker removed
one bottleneck and revealed the next one sitting immediately behind it. The
obvious next increment is an in-process NVML read, which would plausibly reach
the high hundreds of ex/s; I have not done it, and I am not claiming it.

## Criterion 2 — the numbers must reproduce, and they do not

**213 of 264 medians reproduce**
(median relative difference 12.5%), against each
row's own measured across-invocation CV x 3, floored at 10% and capped at 75%.

The bead's own standard is explicit: *"a throughput fix that changes the numbers
is worse than no fix."* By that standard this run **fails**, and I am reporting
it as a failure rather than as a 10.5x win with a caveat.

Two candidate causes, stated as candidates because I have not separated them:

1. **The two invocations no longer have independent CUDA contexts.** 7.1 ran
   them as independent processes, so its `ai_cv` measured context-to-context
   variation. Here they share one, so the estimator measures something slightly
   different -- and warmer, since state carries across examples.
2. **Warm state between examples.** A persistent process leaves L2 and TLB
   populated from the previous example, which plausibly *helps*, and is exactly
   the kind of effect that makes a batching fix change labels.

Distinguishing them needs a run that varies one at a time. I did not do that,
and a guess would be the wrong thing to build the next decision on.

## Criterion 2b — device time

**145.8 us** against the 126 us asserted
(+15.7%). Within 20% but not within 10%, and it moved *up*,
which is the safe direction: more device work, not less. `tests/` asserts it
both ways, because a device time *below* the floor would mean fewer launches
ran, and that is the direction in which a lazy fix looks like a win.

## Criteria 3-6

- **3 protocol unchanged**: 40 timed launches, 2 invocations, median-of-medians.
  Processes 528 -> 1, and *only* that.
- **4 per-example sampling**: **264/264**
  distinct sample times, spanning 10.35 s. Taken
  *inside* the loop on each example's END marker. My first version read all
  output and then sampled every row, which put every sample at the same instant
  -- a per-batch sample wearing per-row clothing, and precisely what the
  constraint exists to prevent. It now also reports the sampling cost rather
  than folding it into an unattributed residual.
- **5 device utilisation**: **0.4%** over
  10.5 s, with a 10m max-runtime on the submission. Better than
  7.1's ~0.1% but still low, for the sampling reason above.
- **6 small-rung outcome, stated not discovered**:
  smoke-as-written: NOT fed; small: NOT fed; medium: NOT fed; large: NOT fed. Stated here rather than left to be discovered, and the trial count was NOT revisited to improve it. At
  25.0 ex/s, **no** rung is fed against the
  7,911 bound. The trial count was not
  revisited to improve that.

## Criterion 7 — the regression guard

`tests/test_batched_worker.py`, 12 checks. It asserts the **asymmetry**: overhead
falls by an order of magnitude *while* device time per example does not move,
and that device time is not *below* the protocol floor. A guard asserting only
"throughput went up" would be satisfied by a change that stopped measuring
properly.
