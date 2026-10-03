# gpu-sim-u2o.6.6.2 — anchor + scaling set (`anchor_v1`, FROZEN)

**`anchor_version: 1`, frozen 2026-09-28.** Measured by ts jobs
`20260928t032717-6356dbf2` (launch), `5c2e74a0` (memory), `ff2f2da2`
(reduce), `3f5587a6` (gemm) — 124 configs, all four with
`n_other_running_at_sample_time = 0`, so this anchor is contention-free *by
record* rather than by assumption. Any later change is `anchor_v2` with a
stated reason, never an in-place edit: 6.1.1's closing diff is taken against
this version, and an unstamped successor makes that diff wrong invisibly.

## What changed in v1 (the review was right about all of it)

* **Access-pattern contrast, lifted not invented.** `k_strided_copy` was
  `k_copy` wearing a different name — both coalesced, both stepping by
  `gridDim.x*blockDim.x`. Replaced with a one-time lift of `strided_read`
  (runtime stride 17), `scatter_write` and `random_access` (LCG) from
  `generator_stage3.py`, with the source text kept beside each copy. The
  family now has **six** distinct patterns, and they finally differ in the
  data: at n=2²⁶, `copy` 1533 GB/s vs `scatter_write` 608 GB/s.
* **Ceiling = argmax *index*** over a stated HBM candidate set (largest n).
  An argmax over all cells publishes the L2-resident n=2²² case (3.2 TB/s) as
  a *memory* ceiling. Ceiling = **1538.7 GB/s** from `random_access
  n=67108864 auto/256`, with the ranked candidate list emitted beside it.
* **Per-cell byte convention** from each kernel's argument roles, not one
  global `2·n·4`: saxpy moves two reads and one write, the strided kernel
  reads n/17, the scatter/random kernels stream their index/seed buffers too.
* **Regime flag on all 124 cells** (80 launch-bound, 9 memory-bound, 35
  compute-bound), threshold stated in the artifact.
* **Queue state recorded** in every family file (item 8).

## Crossover: re-derived, and it happens to reproduce

`launch_to_memory_crossover_n = 262144` — **recomputed from scratch** under
this commit's ceiling *and* candidate set, not carried forward. The old value
survived a 480× ceiling change because the test was a ratio and the
candidate set changed at the same time, so its survival was not validation.
It reproduces here under three simultaneous changes (ceiling 1541.3→1538.7,
new candidate set, per-cell byte convention), which is a different claim from
the one that was suspended.

## Honest note on the process

Four queued jobs were lost to defects a two-second CPU check would have
caught (a missing import; then a misattributed `own_job_id` that made three
of four family files claim the same run). `preflight_static.py` now runs
before every submit and the own-job resolution is deterministic — both tested
in the failure direction, not just the success direction.

## Original anchor notes (v1, first measurement)

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

## Rebase onto master (2026-10-03) and the cross-run variance probe

**Rebased**: the 12 commits are now on `lab/gpu-sim-u2o.6.6.2-rebased`, cherry-picked
onto master `033c149`, **zero conflicts**. Not `git rebase` — see `job.json`
`rebase_onto_master.why_not_rebase`; I ran the direction backwards first and it
tried to re-apply `8dw`, which master had already merged.

The premise behind ruling (c) is verified here rather than assumed:
`git diff master..HEAD -- epic_u2o/kernel_driver.py epic_u2o/safe_harness.py`
is **empty**, so the measurement path is byte-identical and v6's device timings
remain comparable to anything measured from this point.

**Variance probe** (`variance_probe.py` → `variance_probe.json`) — **NOT an
anchor**. It differences the v8 re-run against the v6 anchor on the three
families both cover, joined on the config label, to measure how far the *same*
configuration moves between two runs:

```
106 comparable cells (memory 90, reduce 8, gemm 8)
overall v8/v6: median 1.0181  p10 0.9408  p90 1.2350  max 1.5587
```

This is the number every later single-run comparison against the anchor must
quote: a few percent median with a tail past 1.5x, so a single-run A/B cannot
resolve anything below the p90 band without replicates.

**One discrepancy is unresolved and is recorded in the artifact, not smoothed
over.** The PI independently differenced the same two runs and got 30 comparable
cells, overall median 1.047, memory median 1.082. The two agree *exactly* on
reduce (1.1012 vs 1.101) and gemm (1.0133 vs 1.013) and disagree on the memory
family and the cell count. I could not reproduce 30 from this data — memory is
90 cells (6 kernel kinds × 15 configs). My figures are hand-verified against the
raw files and the join key is stated in the script so either number can be
recomputed. **The spread must be adjudicated before either figure is quoted as
the anchor's resolution.** It does not change ruling (c): both computations say
a few percent median with a long tail.

The v8 family files are preserved under `variance_probe_v8/` so the probe is
reproducible and so the evidence for gpu-sim-u2o.6.6.2.1 is not lost.
