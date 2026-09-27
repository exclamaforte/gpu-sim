# gpu-sim-u2o.6.6.2 — anchor + scaling set, measured BEFORE the rework

94 configs across four hand-written kernel families, on the current
measurement path (`kernel_driver.synthesize` → `compile_kernel` →
`run_kernel`), so that **6.1.1's post-rework closing run can be diffed
against this**. That comparison is the entire point of the bead, which is why
it is committed rather than left in a scratch worktree.

`preflight_check` is bypassed on purpose (it executes on the real GPU per the
6.1.1 ruling, and an unaccounted second execution is exactly what an anchor
must not inherit); this is stated at the top of every `output_<family>.log`.
Kernels are hand-written in `anchor_kernels.py`, never generator output —
gpu-sim-9ti showed stage2-5 `kernel_name` is untrustworthy, and an anchor that
moves is not an anchor. `anchor_kernels.self_check()` asserts at load time that
every declared `kernel_name` really is the `__global__` function in its
source, so this bead cannot be bitten by the same class of bug.

## Headline: the across-invocation CV of the median

| regime | across-invocation CV (median over configs) |
|---|---|
| all 94 configs | **0.025** |
| cells under 10 µs | **0.074** |
| cells over 10 µs | **0.0013** |

Within-process `cv` over the same sub-10 µs cells actually spans
**0.114–3.306** (my earlier comment said 0.3–0.7; that range came from an
earlier round, not from this artifact).

This is the number job `9c2420b3` never measured, and it is the one that
decides reproducibility. **The 20-trial single-launch protocol would not have
been adequate** for the sub-10 µs cells: across-invocation CV of 7.4% means
two runs of the same config can differ by ~7% before any statistics are
involved, so a 20-trial median is not a stable target there. Above 10 µs the
same protocol is fine (0.13%).

Note the contrast with the within-process `cv` field in the same records,
which spans 0.114–3.306 for the sub-10 µs cells: within-process spread badly
understates the disagreement between processes, exactly as gpu-sim-u2o.6.6.1
found. The bead's headline is that larger number, not the prettier one.

## Anchor block

* **Empty-kernel launch cost: 1.68 µs** (median over every `k_empty` config
  and all three invocations). Single-launch `cudaEvent` timing, so it still
  contains the ~0.4–1.4 µs of measurement overhead that 6.6.1 decomposed —
  it is the cost of *measuring a launch*, not of dispatching one.
* Best-known config per family is in `metrics.json` under
  `anchor.best_known_config_per_family`, for later use as a sanity check on
  the model and the roofline baseline.

## Regimes, from the data

* **Launch-bound** (achieved bandwidth below 10% of the measured ceiling):
  n = 1 024 and n = 16 384, all four elementwise kernels — 0.003 to
  0.058 GB/s, i.e. the time is not moving data at all.
* **Memory-bound**: n = 4 194 304 at **3.10–3.21 TB/s** and n = 67 108 864 at
  **1.54 TB/s**. Both are *achieved* rates for 2·n·4 bytes moved, not datasheet
  figures.
* **The ceiling is the HBM one: 1541 GB/s**, taken from the n = 2²⁶ cell. The
  reviewer was right that my previous revision took the ceiling from the best
  cell, which is the n = 2²² case — L2-resident on this part, sustaining
  3.1 TB/s. Publishing that as "the memory ceiling" would calibrate every
  later roofline against cache. Each cell now carries `working_set_MB` and an
  `l2_resident` flag so a consumer can see which regime a number came from.
  (The earlier 3.2 GB/s figure in this file was also a 1000× unit error:
  `median_ms` is milliseconds, so bytes/ms is 1000× short of bytes/s. Both
  defects are fixed in `merge_metrics.py` and recomputed from data already
  collected — no re-run.)
* **Crossover: n = 262 144** — the smallest measured size whose achieved
  bandwidth reaches 10% of the ceiling. Between 16 384 and 262 144 the
  elementwise families cross from "measuring the launch" to "measuring the
  memory system".
* The tiled GEMM is the compute-bound end of the ladder and is by far the most
  reproducible thing measured: 4096³ at 3.08 ms with across-invocation CV
  0.0004.
* Every cell also carries `covers` = elements actually processed
  (`min(n_problem_size, grid*block)`). The deliberately-poor `1block/1024`
  rows are labelled `n=2^26` but cover 1024 elements, which the label alone
  did not make obvious.

I deliberately did **not** use "within 10% of the empty-kernel median" as the
launch-bound test: single-launch timing adds ~1 µs on top of the 1.68 µs
launch cost, so every small-n cell lands near 2.7 µs and that test would
mislabel the whole sweep. The bandwidth-ratio test is in the code with its
definition recorded next to the number.

That test is scale-invariant — the threshold and the measurement carry the same
unit error — which is why the crossover value of **n = 262 144** survived both
corrections unchanged. The reviewer recomputed it three ways and got the same
answer, and I agree that the value being right did not excuse the reasoning
being wrong.

## Three rounds, and why the first two were thrown away

The superseded runs are kept in `ts_job_logs/` rather than deleted, so a
reader can see exactly which of my own defects moved the numbers:

1. **v1** — the elementwise kernels used one-element-per-thread indexing, so
   the deliberately-poor "1 block" config silently computed only 1024 elements
   and reported ~0.4 µs. A broken measurement wearing a terrible config's
   clothes; an anchor set containing one is worse than no anchor set. Fixed
   with grid-stride loops so every config covers all n.
2. **v2** — `k_block_sum`'s shared tile was `REDUCE_TILE 256` while the sweep
   also ran `block=1024` configs: an out-of-bounds shared-memory write. Fixed
   to 1024.
3. **v2 also exposed the worst one** — I passed the block size as `dim3.z`
   while the kernels index with `threadIdx.x`, so *every* 1-D config ran with
   one active thread per block. **v3** fixes the block tuple to `(block, 1, 1)`
   and re-measures all four families. v3 is the authoritative set; v1/v2 are
   retained only as a record.

## Files

* `anchor_kernels.py` — hand-written kernels, hand-written specs, `self_check()`
* `measure_family.py` — measures one family; `run_family.sh` — ts entry point
* `merge_metrics.py` — CPU-only merge into `metrics.json`
* `family_<name>.json`, `metrics.json`, `output_<family>.log`, `job.json`,
  `ts_job_logs/` (all 13 jobs)

Telemetry: no throttling was observed in any family
(`clocks_throttle_reasons.active = 0x0` before and after each), so no
cross-config comparison here is contaminated by clock behaviour. The GPU
still idles at ~230 MHz and boosts to ~2.9 GHz under load, per 6.6.1.
