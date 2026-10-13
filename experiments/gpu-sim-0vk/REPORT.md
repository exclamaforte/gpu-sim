# gpu-sim-0vk REPORT: thicken the low-noise corpus

## Outcome

The extension ships **983 new rows, 933 measurable**, all through the shared
`classify()` with a pre-registered keep-list, gate, and census. Merged with the
frozen 2278, the corpus is **3261 rows, 1285 measurable = 39.4%** (was 15.6%),
and the **memory->memory class is 538** (was ~111). Both halves of the 9.1
revisit condition clear. Every gate below is PASS on committed artifacts;
`run.sh` replays merge + census + OOB-naming + verify byte-identical.

## 1. Prediction (before generation, through the verdict's own classify())

`predict_0vk.py` enumerated candidates from the 11.3-diagnosed flip condition,
ran each through `census.classify()`, and committed `keep_list.json` (990
cells) + `prediction.json` **before any binary was built**:

- keep 990, predicted mm **399**, predicted measurable@0.015ms 963,
  gate (280/900) PASS.
- The 399 is **one short of the ~400 bar, named not rounded**: the measured
  result is 395 (below), and the gap is explained in section 5, not tuned away
  -- the keep-list was never edited to cross a threshold (only 108 strided
  hashes moved for the overflow fix, zero class changes, gate re-PASS).

## 2. Build: 990 specs, 0 failures, 38/38 templates guard-pass

`build_0vk.py` derives every spec from the generators' own schemas, asserts
`kernel_hash == keep-list hash` per cell (hash moved = build fails), and runs
every template through f6y's OOB guard (`guard_verdicts.json`: 38 templates
pass, 0 refused). `manifest.json`: 990 specs, 0 failures, zero hash collisions
with the frozen 2278 (append-only enforced in code).

Compile ran **unqueued** (acceptance 8): 990/990 in 1596s (`compile.log`), plus
108 strided recompile in 209s and 93 refill in 124s after the overflow fix.

## 3. The strided-768M overflow (fixed, f6y precedent: fix, don't enshrine)

Three strided families faulted deterministically at n=768M: template guard
`i*stride` overflows 32-bit int for i >= 2^31/stride before the comparison.
Fixed to `i < n/stride` (identical active set since 8|n); guard extended with
the mirror rule; f6y `test_oob_guard` 6/6 still green. Slice-A trial v2
confirmed the fix on GPU (3/3 now pass).

## 4. Measure: queued pace-chunks, 983 rows

Biggest-first slices were killed by the queue's underuse gate (fill-bound
giants look idle to `nvidia-smi`), so measurement ran as 8 pace-chunks
(`chunks.json`, each <= 240s wall, plain short-smoke submits, same protocol
and binaries) plus mop-up MU2. Salvaged rows from the two killed slices
(A: 88, A2: 208) were verified complete and consumed by the merge, never
remeasured. Queued-measure wall: C0-C7 + MU2 = 1779s + trial legs; per-chunk
`measure_summary_*.json` carries wall_sec, queue_before/after, contention
disposition, and per-stage failures. Every row carries ai_cv (median 0.0018),
own_job_id, queue_state, protocol `single_run_stages_2_5`, and
guard_passed_template=true.

Shortfall (7, named in `shortfall.json`, never rows): six 768M-3-buffer cells
(random_access x2, scatter_write x2, gather_read x2 -- availability-marginal,
see section 6) + stencil_2d_5pt 16M f75dd6548da5eaac (twice-attempted, cause
not host-memory -- stated, not folded into another mechanism).

## 5. The mm "mystery": a stale artifact, not a selection failure

The bead record quoted `census.json` reading mm_new=0 against a 399 prediction.
That file was written by the census's **first version, which had the class keys
wrong** (`mm` vs `memory->memory`) and crashed before the rewrite ran
(`cp.corpus_specs` never existed -- `census_run.log` ends in AttributeError).
The current `census_0vk.py` reports, twice in a row:

- new-row census: latency->memory 586, **memory->memory 395**,
  latency->latency 2; require_census structure OK (sums to 983);
- forecast-vs-verdict mm agreement **983/983** -- prediction and verdict use
  the same classify() on the same features, and they agree on every row;
- bead bar (mm>=250, meas>=750): **PASS**; FULL diversity gate: **PASS**;
- negative control: 90%-one-regime mix FAILs the gate as required: **PASS**.

The 399 -> 395 gap is exactly the shortfall: 4 of the 7 shortfall cells were
predicted memory->memory. Nothing about the selection was wrong in class.

## 6. Availability pre-flight (PI part-5 fix, landed in measure_0vk.py)

`host_footprint_bytes()` mirrors the driver's own sizing
(`kernel_driver.py:480,508-514`: per-pointer-arg `buf_counts`, arity-correct
for GEMM via `_buffer_counts`); `preflight()` refuses cells whose footprint
exceeds current MemAvailable, recording a `preflight-skip` instead of a fault.
Demonstrated on CPU over all 990 manifest specs:

- at the PI-observed low availability (7.9 GiB): refuses **exactly the six
  768M-3-buffer shortfall cells**, nothing else;
- at 20 GiB: refuses nothing (no false positives);
- the stencil shortfall is NOT refused (small footprint) -- different cause.

**Correction to the bead record, with the driver lines as evidence:** the
"30 cells unrunnable at any position" (PI part-2, max 27.0 GiB) used
problem_size x nargs, which overcounts GEMM (problem_size M*N*K is not
per-buffer). The driver's true max staging cost is **9.0 GiB** (the 768M
3-buffer cells). So **0 cells are structurally unrunnable**; 6 are marginal
when availability dips below ~9 GiB (observed 7.9). The shortfall mechanism
stands; the count is corrected.

## 7. Frozen corpus untouched (hash comparison, not assertion)

- Counts: 264 + 192 + 1804 + 18 = 2278 unique hashes; new rows disjoint
  (asserted in census_0vk.py, re-checked in verify).
- `experiments/gpu-sim-u2o.11.3/measured.jsonl` sha256 =
  dcc4d920...af4baf == f6y's `u113_measured_sha256_before.txt` exactly.
- The 4 known-OOB rows are named in `oob_frozen.json`, each proven by
  hash-match to the pre-fix template plus guard refusal with exact maxima:

| hash | op | n | grid x block | guard maximum |
|---|---|---|---|---|
| 007e97e800e2a0b9 | segmented_read | 131072 | 128 x 512 | in[seg*n+off] to 409599 vs 131072 alloc |
| 53c5810e34048ee1 | segmented_read | 131072 | 64 x 1024 | to 409599 vs 131072 alloc |
| 8a3894b2ac4bb135 | segmented_read | 131072 | 64 x 512 | to 147455 vs 131072 alloc |
| 62b7e7ddf8fb1bb1 | strided_write | 131072 | 128 x 256, stride 8 | out[i*8] to 262136 vs 131072 alloc |

All four stay frozen. `name_oob_rows.py` re-derives this from git history
(pre-fix templates at `3dd024b^`, self-validating on the broken idioms).

## 8. Families (acceptance 4)

983 rows across 38 families; largest family 50/983 (5.1%); plain elementwise
(add/copy/div/mul/relu/saxpy/scale/triad) 47/983 (4.8%). The thin regime was
thickened with reductions, gather/scatter/random, bank-conflict, histograms,
stencil/transpose, GEMM and divergence families -- not the same ops enlarged.

## 9. Expired held jobs (this turn's pages)

20261011t162835-1a2012fb, 20261011t163005-a7869d7d,
20261011t165135-db78c386: all three were superseded slice pairs that sat held
and expired after 6h without approval. Their logs are empty; they never held
the GPU and nothing was learned or lost. No action beyond this record.

## Gates (all green, all recomputed by committed code)

`verify_0vk.py`: 27/27 PASS. `census.json`: bead_bar PASS, full_gate PASS,
neg_control.control PASS. `metrics.json`: n_rows 983, mm_new 395, mm_full 538,
measurable_share 0.394.
