# claims_inventory.md — gpu-sim-u2o.6.5.1

Every GPU-free / CPU-only / no-GPU isolation claim in `epic_u2o/*.py` and
`shim/*.h`, one row each: file:line, the claim, TRUE/FALSE today, and the
executable check that proves it (or the correction). Line numbers verified
2026-10-06 against branch lab/gpu-sim-u2o.6.5.1.

Rule of the game (bead design §4): a device-free claim must not be
manufactured by hiding the GPU. Checks use a HOSTILE device env
(`CUDA_VISIBLE_DEVICES=nonexistent-device-651`) and a PATH-shadowed
`nvidia-smi` tripwire, never a blanked env passed off as a sandbox.

## Rows

| # | file:line | claim | true? | executable check |
|---|-----------|-------|-------|------------------|
| 1 | epic_u2o/batch_measure.py:4-5 | BUILD phase is CPU-only, "No GPU contact" (compile only, never executes) | TRUE | tests/test_isolation_claims.py: `build.compile_succeeds_hostile_env`, `build.binary_produced`, `build.no_nvidia_smi_contact` (one-spec BUILD mechanism under hostile env + tripwire PATH); `build.only_spawns_nvcc` (AST pin: build_batch spawns only nvcc — produced exes are written, never executed) |
| 2 | epic_u2o/batch_measure.py:38 | `build_batch` docstring: "CPU-only: specs -> drivers -> binaries -> manifest" | TRUE | same as row 1 |
| 3 | epic_u2o/batch_measure.py:86 | `device_snapshot`: "Read-only nvidia-smi snapshot (no GPU workload)" — GPU-TOUCHING BY DESIGN (read-only query, RUN phase inside ts job) | TRUE (touching, justified) | tests/test_isolation_claims.py: `snapshot.calls_nvidia_smi`, `snapshot.query_only` (--query-gpu captured from argv), `snapshot.no_reset_flag` (no -r/--gpu-reset), `snapshot.returns_record` |
| 4 | epic_u2o/preflight_validator.py:2-5 | "GPU-free smoke test" (module docstring) | TRUE | tests/test_cudart_none.sh (ldd purity, blank-env smoke, fake-latency proof, no-fallthrough negative control) + tests/test_preflight_cannot_pass.py (checks 1-5: failure paths fail loudly) |
| 5 | epic_u2o/safe_harness.py:13-16 | module docstring: "-cudart=none + libfakegpu smoke run of the synthesized driver (GPU-free)" | TRUE (corrected this bead; was FALSE — see below) | tests/test_isolation_claims.py: `smoke.cudart_none_links`, `smoke.ldd_no_cudart`, `smoke.ldd_no_libcuda`, `smoke.runs_hostile_env`, `smoke.prints_trial_json` (mechanism) + `preflight.smoke_passes_real_path` (the fixed `SafeHarness._preflight_smoke_run` end to end) |
| 6 | epic_u2o/safe_harness.py:245-252 | `preflight_check` docstring: "Real structural validation, GPU-free ... ldd shows neither libcudart nor libcuda ... FAKE and never used as measurements" | TRUE (corrected this bead) | same as row 5 + `preflight.message_carries_no_ms_timing` (return carries a trial-line COUNT, no ms payload flows toward targets) |
| 7 | epic_u2o/safe_harness.py:300 | "nvcc -ptx syntax check (no link, no GPU)" | TRUE | tests/test_isolation_claims.py: `preflight.ptx_gate_rejects_broken`, `preflight.ptx_gate_names_cause` (broken kernel rejected with the -ptx error; nothing linked, nothing executed) |
| 8 | shim/cuda_runtime.h:13-14 | "Workloads compiled for CPU-only CI include this header instead of <cuda_runtime.h>" | TRUE | `make check`: tests/fake_client.c, tests/vector_add.c, tests/ring_allreduce.c all `#include "cuda_runtime.h"` (the shim, via -Ishim) and pass |
| 9 | shim/cuda_runtime.h:43-59 | mandated preflight route incl. :43-46 note ("running GPU-free against the direct link ... no LD_PRELOAD", corrected this bead — was stale LD_PRELOAD narrative) + :49-59 "structurally incapable of touching a real GPU" (no per-symbol fallthrough) | TRUE | tests/test_cudart_none.sh: ldd matches neither libcudart nor libcuda, libfakegpu is the provider, same compile without -lfakegpu fails to link |
| 10 | shim/cuda_runtime.h:277-301 | fake semantics notes ("always reports success", "no-ops returning success/0 are the correct fake behaviour") — descriptive of fake behaviour, not an isolation claim | TRUE (descriptive) | `make check` fake_client + vector_add + ring_allreduce under LD_PRELOAD all exit 0 on the fake |
| 11 | shim/nccl.h:10-11 | "Workloads for CPU-only CI include this header instead of <nccl.h>" | TRUE | `make check`: tests/ring_allreduce.c `#include "nccl.h"` (the shim) passes with 4 fake ranks under LD_PRELOAD |

## False claims found and corrected (this bead)

- **THIRD instance of the bead's failure mode** (the two known-false ones from
  6.1.1/6.1.3 are gone — verified: `grep GPU-free` returns only rows 4-6
  above, all checked). `SafeHarness._preflight_smoke_run` compiled the
  synthesized driver with plain `nvcc -O2` (default `-cudart=static`) and ran
  it under `LD_PRELOAD` — the exact pattern 6.1.1 proved false and
  shim/cuda_runtime.h:46-56 documents as abandoned: static cudart shadows the
  preload, so the "GPU-free" smoke could genuinely touch the real 5090.
- **Correction**: `_preflight_smoke_run` now compiles with `-cudart=none`
  (verified to link: `cudaMemGetInfo` et al. are exported by libfakegpu) and
  runs with no `LD_PRELOAD` (rpath direct link); docstrings at
  safe_harness.py:13-16, :250-252 and the §4 comment updated to name the
  mechanism. Fail-closed behaviour preserved (link/run failure → fail loudly).
- **Stale narrative (same bead)**: shim/cuda_runtime.h:43 described smoke
  binaries as "running GPU-free under LD_PRELOAD" as the current mechanism
  — false after the correction above. Rewritten to the -cudart=none direct
  link (row 9). Comment-only; `make check` re-verified green after the edit.

## Counts (for the bd comment)

- Claims inventoried: 11 (9 isolation claims + 2 descriptive notes).
- False on arrival: 1 (rows 5-6, one code defect behind two docstrings).
- Rows that gained a NEW executable check this bead: rows 1, 2, 3, 5, 6, 7
  (tests/test_isolation_claims.py, 17 checks, all passing).
- Rows backed by pre-existing checks: rows 4, 8, 9, 10, 11
  (test_cudart_none.sh, test_preflight_cannot_pass.py, make check).
- `make check`: green (guard + fake_client + vector_add + ring_allreduce +
  test_cudart_none.sh, 9/9 PASS).
- Precision note (not changed, out of scope): safe_harness.py:252 says smoke
  timings are "FAKE"; the synthesized driver's per-trial prints read 0.0 even
  with FAKEGPU_KERNEL_LATENCY_MS set (latency injection lives in the
  event-elapsed path, src/fakegpu_cuda.c:273-276). Fake-but-arbitrary, still
  never measurements — the load-bearing half holds; the "(configured
  latency)" parenthetical was dropped from the docstring as imprecise for
  this path.
