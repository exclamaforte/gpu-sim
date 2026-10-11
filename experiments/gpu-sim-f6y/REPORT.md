# gpu-sim-f6y: OOB template fix + driver preflight guard

## What was broken

11.3's shortfall ledger named 18 unique hashes (55 + 25 = 80 occurrences)
whose kernels index out of bounds **by construction**:

- `segmented_read`: `out[i] = in[segment * n + offset]` -- the segment base
  multiplies by `n` (the whole array) instead of `segment_size`. Any
  `segment > 0` reads up to `7 * n` past a length-`n` buffer.
- `strided_write`: `if (i < n) out[i * stride] = in[i]` -- the guard bounds
  the thread id but the write strides past it. Any `i * stride >= n`
  writes out of bounds (up to 8x past the end at stride 8).

Both crashed in the GPU slot (binaries present, 0 successful invocations).

## Deliverable 1: fixed templates, 18 cells measured

Two one-line fixes in `epic_u2o/generator_stage3.py` (diff: 2 lines changed,
nothing else):

- `in[segment * n + offset]` -> `in[segment * segment_size + offset]`
- `if (i < n) out[i * stride]` -> `if (i * stride < n) out[i * stride]`
  (mirrors `strided_read`'s established guard shape for strided access).

New sources -> new hashes (all 18 moved; verified disjoint from the 1804 and
from the 18 shortfall hashes, which stay missing as the broken-template
record). Selection: every 11.3 manifest occurrence under the 18 missing
hashes, regenerated with the fixed templates -- 80 occurrences, no sampling,
no discretion. Compile unqueued (80/80, 129s), one queued measurement job
`20261011t145826-efc35eab` (18s wall, contention did not fire):
**18/18 groups measured exactly once** (11 segmented_read + 7 strided_write).
The 62 other occurrences are phantom-unlink events (their group's binary was
already consumed -- identical in kind to 11.3's 500, proven by count:
80 - 18 = 62 exactly, and every group has exactly one row). Zero fixed
binaries crashed. Every row carries `ai_cv`, `own_job_id`
(`20261011t145826-efc35eab`), `queue_state`, protocol stamp
`single_run_stages_2_5` (the identical procedure), `driver_scalars`, and
`manifest_occurrences`.

Regime of the 18 (item 4): census class 14 latency->latency / 4
latency->memory; measured_regime launch_bound on all 18 (medians < 10us --
small stage-3 kernels, reported, no target attached).

New total + census (item 8): **1822 rows**, 1474 / 237 / 111, verdict PASS
(smallest share 6.09% >= 5% floor). The 11.3 files are byte-identical
(sha256 `dcc4d920...` matches the pre-measurement snapshot; 11.3 directory
git-clean) -- item 7 holds: append-only, nothing regenerated.

Note on counts: the 11.3 REPORT prose says "segmented_read 13,
strided_write 5". The machine record (`missing_hashes` x manifest) gives
**11 + 7** -- computed, see `run.sh`. The prose hand-count was wrong; the
ledger it pointed at was right (18 hashes, 80 occurrences either way).

## Deliverable 2: OOB guard in the driver preflight

`synthesize()` now runs `check_subscripts_bounded()` on the kernel source
before emitting any driver. It evaluates subscripts built solely from the
linear thread id, `n`, known int scalars, and literals (+ - * / %) by
interval arithmetic, over the thread range guard contexts establish
(`if (i<n)` / early-return narrow; `if (<EXPR><n)` caps that expression;
monotone `i*K` / `i+K` narrow by inversion; `&&`-condition remainders covered
left-to-right). Refusal fires ONLY on proven overflow or proven-negative --
never on "cannot tell". Honest scope, stated in the module docstring:
data-dependent indices (gather/scatter/random), loop induction variables,
`__shared__` tiles, and driver host code pass through UNCHECKED with a note.
The guard does NOT catch arbitrary OOB; it catches exactly the idiom class
both f6y bugs belong to.

Three false-positive classes were found by sweeping and fixed before
committing (each fix is monotone-toward-pass, so it cannot introduce a false
refusal): declaring assignments seen as mutations (masked), `++k`
pre-increment missed by the mutation scan, `&&`-condition reads unguarded.
One real guard bug was caught the same way (the bare-`if (i<n)` branch
dropped during cleanup) -- the sweep earns its keep.

## Negative control (item 6)

The pinned pre-fix sources are refused on CPU (no GPU involved):

```
oob-guard: segmented_read: subscript in[segment * n + offset] can reach
  3948543 (thread range [0,262144)) but the driver allocates 262144 elements
oob-guard: strided_write: subscript out[i * stride] can reach 2097144
  (thread range [0,262144)) but the driver allocates 262144 elements
```

Commands: `tests/test_oob_guard.py::test_broken_templates_refused`
(pinned literals, exact maxima asserted) and the full-manifest sweep
(`test_manifest_all_pass`: 2322/2322 synthesize under current templates).
The urr warning is heeded: the negative control pins the broken behavior in
a test that survives the fix, not in the fixed code's shadow.

## Incidental finding: 4 OOB-but-completing rows in the frozen corpus

The guard refuses 84 occurrence-cells; 80 never produced rows (the shortfall)
but 4 share occurrence keys with shipped rows (e.g. strided_write n=131072
stride 8 grid 128, segmented seg=16384 under-covered cells). Those kernels
are PROVEN over-allocation (up to 8x past the end) yet completed --
allocator slack for reads, luck for writes. Their timings are real executions
of off-spec programs; the rows stay frozen (item 7), but 9.1 should know the
corpus contains 4 rows whose programs read/wrote out of bounds. This is the
guard's value stated concretely: without it, OOB executions ship as rows.

## Gates

- Narrow: `tests/test_oob_guard.py` 6/6 PASS.
- Repo-wide: 430 passed; 8 failures, all reproduced on the base commit with
  this work stashed (pre-existing, other beads' history-lint/recipe tests).
- Recompute: `run.sh` replays build-manifest gate + merge + census + verify
  CPU-side (byte-identical `measured_new.jsonl` expected; the GPU job is not
  replayed).
- Registry: `single_run_stages_2_5` rows 1804 -> 1822 (same procedure, 18
  appended rows), median_ai_cv recomputed over the 1822; EXPECTED_ROWS
  updated so the suite enforces the new N.
