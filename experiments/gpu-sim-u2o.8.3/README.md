# gpu-sim-u2o.8.3 — numeric feature vector, populated from real measurements

The audit note said the feature-vector code "has never been populated from any
real measured example (no such example exists)". **456 now exist**: 7.1's 264
(ts `20260928t074004-b429f469`) + e5q's 192 (ts `20261010t224024-465ad815`),
hash-disjoint, joined here. So this populates all 456 — and the more useful
results are what the vector could not see *before* it was populated.

No GPU, no ts job. The `.cu` sources are build artifacts and uncommitted, so
each row's source is **regenerated and then verified by hash**: `kernel_hash`
is the unified `kernel_hash_for` (`sha256(source + json([grid, block,
n]))[:16]`), so a regenerated spec hashing to a row's recorded value is
provably that row's source. **456/456 verified; zero rows matched by name or
position.** A row whose hash cannot be recovered aborts the run.

## Defects fixed (five, all in or around `epic_u2o/features.py`)

1. **Unimportable.** `from tokenizer import ...` instead of
   `epic_u2o.tokenizer` — the file could never run. Package-relative import
   + an import test so it cannot rot again.
2. **Duplicated struct.** `roofline.py` carried its own `KernelFeatures`
   ("matches features.py") plus a third copy of two constants. The copy is
   deleted; `roofline.py` imports from `features.py` (verified by deleting,
   watching the suite pass).
3. **No source in the corpus.** Solved by regenerate → verify-by-hash →
   extract, per corpus sweep. The hash check is load-bearing.
4. **Blind to problem size.** 192 e5q rows collapsed to 80 vectors (56
   groups, worst 59.3%). Fix: `covers = min(threads, problem_size)` — the
   work actually performed, already recorded on every row — appended at
   vector position 28, and the scale for all per-thread estimates
   (guarded-out threads do no memory/compute).
5. **Dead read/write fields (the 72.6%).** `estimated_global_reads/writes`
   were initialised to 0 with no writer anywhere, so `bytes_moved` was 0 for
   all 456 rows, the memory arm was 0.00, and the baseline predicted the
   launch floor for bandwidth-bound kernels (24.5x under on scale n=2^22).
   Now counted syntactically per-thread single-pass: subscripts (read vs
   write by what follows `]`), `*p` derefs, atomics as read+write, float
   scalar params as data loads (alpha, not n), shared-vs-global by
   `__shared__` declaration. Loop- and branch-resident accesses count once
   (trip counts are runtime values; inventing one would be a fitted
   constant by another name).

## Further separations the corpus demanded (all principled, none fitted)

- **Operator mix** (`estimated_add/mul/div_ops`): add/mul/div are identical
  except the operator, and div measures 137% slower at the same launch —
  different units, so one flop number cannot separate them.
- **Call sites** (`num_call_sites`): opaque work the counters cannot see
  (`fmaxf` in relu, `atomicAdd` in stage-5, `__syncthreads`); also the
  stage-5 signal for acceptance item 10.
- Drive-by repairs the work exposed: `uses_shared_memory` never fired
  (`__shared__` lexes as SHARED, never IDENTIFIER); `normalize` indexed
  grid_x at the problem_size slot (rewritten by NAME); `vector_dim` is
  derived (33), not typed.

| | distinct feature vectors, of 456 rows |
|---|---|
| as written (no launch config, no problem size) | **8** |
| corrected | **456** |

## Acceptance mapping (all asserted in `populate_feature_vector.py` or tests)

- Items 1–2: import + exercise tests (`test_feature_vector.py`).
- Item 3: D2 deletion above; suite green.
- Item 4: 456/456 joined AND `bytes_moved != 0` on every row (a 456/456 run
  with zero traffic fails the driver, not just a test).
- Item 5: zero same-vector/different-runtime groups over the real 456
  (driver gate + unit mirror).
- Item 6: `vector_dim == len(to_vector()) == 33` (+ normalized twin).
- Item 7: constants single-sourced (`hw_5090` via `features.RTX_5090_SPECS`;
  roofline's shadow literals deleted).
- Item 8: this file.
- Item 9: scale n=2^22 full-cover row reads=2n writes=1n bytes=3·4·n,
  checked against the hand computation in driver and test.
- Item 10: `TestStagesTwoThroughFive` — one kernel per stage 2–5 through
  the same path, all nonzero; no per-stage branches exist.

## Recompute

`bash run.sh`: unit gates + populate re-run, byte-compares `metrics.json`
and `features.jsonl`. CPU-only.
