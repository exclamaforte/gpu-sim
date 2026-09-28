# gpu-sim-u2o.9.4 — the analytic roofline baseline, on real kernels

The audit note: the baseline "was only evaluated against
`compare_baselines.py`'s `_synthetic_kernel()` fixtures, never real measured
kernels." This evaluates it against 6.6.2's **124 real cells**
(`lab/gpu-sim-u2o.6.6.2-r2:experiments/gpu-sim-u2o.6.6.2/metrics.json`, ts `[{'job_id': '20260928t032717-6356dbf2', 'family': 'launch', 'state': 'finished', 'configs': '18/18'}, {'job_id': '20260928t032717-5c2e74a0', 'family': 'memory', 'state': 'finished', 'configs': '90/90'}, {'job_id': '20260928t032717-ff2f2da2', 'family': 'reduce', 'state': 'finished', 'configs': '8/8'}, {'job_id': '20260928t032717-3f5587a6', 'family': 'gemm', 'state': 'finished', 'configs': '8/8'}]`). No GPU, no ts job.

Zero learned parameters: `max(compute_time, memory_time)`. The **max** is not a
style choice — for a fixed amount of work the kernel is bounded *below* by both
resources, and `6.6.4` measured `min()` under-predicting ~100% at every size on
this same anchor.

## The headline is that my hypothesis was wrong

I expected to find the ceiling overstated **1.50×**, on the reading that
`6.6.2` used a flat `2*n*4` — which its `memory_ceiling_definition` prose still
says. **Checking the fields showed the opposite.** The ceiling cell
`saxpy n=67108864 auto/256` carries `bytes_moved = 805,306,368 = 3·n·4`, the
*correct* accounting for `y[j] = alpha·x[j] + y[j]`. `6.6.2`'s commit
`734980f` is literally *"one byte convention in the file, and the artifact
regenerated with it"*.

So I believed the **prose over the numbers** — the exact inversion of the
standard this lab applies everywhere else, and a good argument for why
`6.11.1`'s lint exists. It is recorded in `metrics.json` and pinned by a test
rather than deleted.

**The published ceiling stands:** 1,575.8 GB/s
published, 1,584.9 GB/s from my
argmax — and that 0.6% gap is a **launch-config** difference (block 1024 vs
256), not a byte correction.

## The two real defects, narrower than the one I expected

1. **Stale convention prose.** `memory_ceiling_definition` says `2*n*4` while
   the fields it describes use `3·n·4`. No numerical effect; it misleads the
   next reader into auditing correct numbers — and it misled *me*.
2. **`k_random_access` and `k_scatter_write` are credited `2·n·4` but move three
   arrays** (read seeds/indices, read the data, write the output). Their
   achieved GB/s is **1.50× optimistic**. Not the ceiling — but they are in the
   `access` family, **45 of 124 cells**, the anchor's largest.

The per-kernel check is deliberately **two-sided**: our accounting **agrees on
12 of 21** reconciled cells. The 9 disagreements split *both*
ways — we are more correct on scatter/random, and **cruder on `strided_read`**,
where the anchor deliberately accounts for the reduced stride footprint and we
ignore it. A check that only ever reported disagreement would be
indistinguishable from one that had the accounting wrong.

## All seven criteria, evidenced in the artifact

| # | criterion | result |
|---|---|---|
| 1 | per-kernel bytes, never flat `2·n·4` | 9 eligible cells; accounting table for all 10 kernels with reads/writes and the source line justifying each |
| 2 | ceiling = argmax over a **named** set | 1,584.9 GB/s from `saxpy n=67108864 auto/1024`; top 4 emitted with ranks; 115 rejected **with reasons** |
| 3 | reconcile against the **committed** path | 21 cells reconciled, **0 mismatches** |
| 4 | per-kernel counts beside per-family | 10 kernels; single-kernel families **['gemm', 'reduce']** |
| 5 | contention bound | ceiling cells all `uncontended`; a contended cell is **refused** by `is_ceiling_eligible()` |
| 6 | regime weighting | memory-bound ×2.33, launch-bound ×1.19 (both measured); `compute-bound` ×1.0 declared **NOT MEASURED** |
| 7 | no clock-based contention | `clocks.sm` (0.098) and `clocks.max.sm` **raise**, against `power.draw` 0.866 / `utilization.gpu` 0.939 |

Census: 124 cells — families access 45, gemm 8, launch 18, memory 45, reduce 8; kernels `k_block_sum` 8, `k_copy` 15, `k_empty` 9, `k_noop_store` 9, `k_random_access` 15, `k_relu` 15, `k_saxpy` 15, `k_scatter_write` 15, `k_strided_read` 15, `k_tiled_gemm` 8.

## One guard that exists because I tripped it

My first run selected a ceiling of **3,855 GB/s** on a part whose HBM peak is
**1,792 GB/s** — impossible, and the loudest possible signal that the candidate
set was wrong. It had admitted `n=4M` cells that fit in the 96 MB L2. Two
guards now refuse it independently: `is_largest_n` in the membership rule, and
`assert_not_above_spec_peak()`. A value above the physical maximum is not a
finding, it is a bug, and publishing it would put an impossible number in an
artifact a later reader has no reason to doubt.

26 tests, 0 failures, CPU only.
