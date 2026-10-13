# gpu-sim-u2o.11.3 REPORT: corpus scale-up (stages 2-5 + mm-regime extension)

1804 measured rows over 1822 unique cells, four stages, one protocol
(`single_run_stages_2_5`), regime census PASS. All counts below are UNIQUE
CELLS unless marked occurrences.

## The numbers

| item | value |
|---|---|
| measured rows (unique) | **1804** (need >= 1544 additional; margin 260) |
| manifest occurrences | 2322 (2211 base + 111 extension) |
| unique cells | 1822 (500 phantom duplicates, see below) |
| per-stage rows | stage2 727 / stage3 631 / stage4 96 / stage5 350 |
| regime census | latency->latency 1460, latency->memory 233, **memory->memory 111** |
| census verdict | **PASS** (3 classes, smallest share 6.15% >= 5%, mm 111 >= 100) |
| stage-4 population | 96/96 (>= 64) |
| ai_cv | present 1804/1804, median 0.0783, 457 unreliable marked IN PLACE |
| queue_state | 1804/1804 carry own_job_id + n_other_running; contention rule ran in 11 slice jobs, fired none |
| protocol | single `single_run_stages_2_5` over all rows |
| split | compile 3079s CPU unqueued (+203s s4 recompile, +276s extension) / measure 1736s queued |
| anchor_v1 | untouched (path filter, 0 hits) |
| join | 1804/1804 recompute + join; 0 mismatches, 0 duplicates |

## How the corpus was built (and what fought back)

**Base (2211 occurrences, size-stratified manifest for regime coverage, not
row count).** Per (stage,op): all of the two largest sizes, then smaller
sizes up to cap. 0 preflight failures (every spec synthesized on CPU).
Compiled unqueued (2211/2211, 3079s). Measured in 8 queued slices
(287/285/233/259/210/89/180/54s walls); a paired trial first paced the
full batch at ~37m (over the 30m slice) and proved queue identification.

**Trial.** `20261011t122933-e206d4a9` (30 rows, 27s) + held
`20261011t122933-65af9df8` (rejected: full-batch projection ~37m outruns
the slice; the held job was itself held at 0.2% GPU util for printing no
progress output). own_job_id identified, contention clean.

**Stage-4 kill and the arity fix.** The s4 slice (96 big-GEMM rows)
averaged 5.5% GPU util and was `killed_underused` at 301s (10% minimum
over the 5-minute grace). Cause: the generic driver sized every pointer at
N = M*N*K elements, so a (1024,1024,256) GEMM filled/copied 3GB per buffer
on the host while the timed kernel ran microseconds. Fix: GEMM buffers are
arity-sized (A: M*K, B: K*N, C: M*N); the timed launch is byte-identical,
host work falls ~500x on square cases. 96 binaries recompiled (203s CPU),
re-run slice `20261011t130848-fd150d4d` finished 96/96 in 89s. The 81 rows
measured before the kill are kept (build scratch) as a replication
reference, not shipped.

**Hash reality (finding for 9.1).** `kernel_hash_for` covers
(source, grid, block, problem_size) but NOT driver scalar values, so 2211
occurrences collapse to 1711 base cells: 500 phantom duplicates (same
binary built last-wins, consumed once by the first slice measuring it;
later slices logged "no prebuilt binary" for the phantom tails -- a
misleading message for an expected event, recorded here). Every shipped
row carries
`driver_scalars` (the winner's int metadata, recomputed deterministically)
and `manifest_occurrences`. Recommendation carried to 9.1: extend the hash
inputs with driver scalars before pooling, or treat same-hash rows as one
cell. The manifest is kept as-is (regeneration record); merge/verify count
unique cells throughout.

**OOB shortfall (18 unique cells, generator defect, not fixed here).**
`segmented_read` indexes `in[segment*n+offset]` and `strided_write` writes
`out[i*stride]`: out of bounds by construction for segment/stride > 0/1,
so those kernels crash (binaries present, 0 successful invocations).
55 + 25 occurrences = 80 manifest rows affected, 18 unique cells. Filed as
follow-up (`gpu-sim-f6y`); the bead clears 1544 without them.

**Census FAIL then extension.** The 1693-row base corpus censused to
latency->latency 1460 / latency->memory 233 / memory->memory **0** --
census FAIL. Probing the construction (same `classify()` the verdict
uses): the memory->memory flip needs ~100MB+ counted bytes at modest block
counts. 2-ref streaming never flips at any grid (tested to 131072 blocks);
the reference regime's 32 members are high-ref-count ops (e5q large-n
reductions/saxpy-class). The extension keep-list (`mm_keeplist.json`, 111
cells: reductions, gather/scatter/random, bank-conflict patterns,
histograms, stencil/transpose 2D, all full-coverage threads == n) was
predicted BEFORE generating through the identical code path; the gate
(<105 flips refuses to generate) fired once at 104 and passed at 111
after honest candidate additions. Measured: **111/111 cells classify
memory->memory -- prediction 111/111 correct, zero drift.** Extension
specs were appended (base 2211 byte-identical asserted), compiled
(111/111, 276s), measured in slice `20261011t135536-0fe42b23` (111/111,
140s). One extension attempt failed first (`20261011t134705-1172e7b7`,
exit 1 in 1s: missing `covers`/`source_path` fields, fixed to the
identical base rules) -- CPU-side, no GPU wasted.

**Shared-memory subtlety.** Occupancy feeds both pinned models, so the
keep-list predicts with construction shmem values (reduce/tiled block*4,
fixed tiles, histogram_shared 4096), not zero. Predicting with 0 then
measuring with block*4 could move the class after the fact.

## Shortfall ledger (all named, none hidden)

- 500 phantom duplicate occurrences (hash under-specification; rows unique).
- 18 unique OOB cells (segmented_read 13, strided_write 5): generator defect.
- 1 wasted ~20 CPU-min partial recompile (timed-out --only-missing rebuilt
  base binaries unnecessarily; killed by operator timeout, no artifact
  impact; stale base .out files remain in gitignored build/).
- Contention fired nowhere (11 slice jobs, all stages covered); queue states
  at both submits recorded in queue_at_submit.txt.

## Provenance

- Queue jobs (all `u2o.11.3`-labelled): trial pair
  `20261011t122933-e206d4a9` / held `20261011t122933-65af9df8`; slices
  s2a `20261011t123221-2521361e`, s2b, s3a, s3b, s3c, s4-killed
  `20261011t123222-54909c19`, s4r `20261011t130848-fd150d4d`, s5a, s5b;
  mmx-attempt `20261011t134705-1172e7b7` (failed, exit 1, missing fields);
  mmx `20261011t135536-0fe42b23`.
- Recompute: `run.sh` re-runs merge + verify + census controls + unit
  tests from committed files (CPU-only). GPU steps (compile, measure) are
  forward-only and recorded, not recomputed.
- Registry landing: the `single_run_stages_2_5` entry in
  `epic_u2o/corpus_protocol.py` is completed with measured values
  (rows 1804, wall 1735.67s, 1.039 ex/s, median ai_cv 0.0783); the
  6.7.1 rows test now expects per-protocol counts (a new protocol with
  no declared N fails loudly) instead of a single 264 constant.
- 9.1 is NOT started (this bead delivers its training corpus; modelling
  begins after review).
