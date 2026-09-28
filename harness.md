# harness.md — resource limits on the shared RTX 5090

Every limit the harness applies, why it is safe for CUDA, and the ts job that
proves it. **Generated** by `experiments/gpu-sim-u2o.6.3.1/render_harness_md.py`
from `epic_u2o/rlimit_cuda_gate.LIMIT_TABLE`. Do not hand-edit the table.

## Why this file exists

Two real-GPU runs were killed by limits that had never been checked against
CUDA, and both times the symptom pointed at the wrong subsystem:

| what | what you saw | what it was |
|---|---|---|
| `RLIMIT_NPROC=32`/`512` | `cudaErrorOsOperationNotSupported` at the first `cudaMalloc` | the kernel compares NPROC against the real uid's **total** task count |
| `RLIMIT_AS=24 GiB` | every run died in ~2 ms | the 5090's BAR1 mapping is 32 GiB |

A misleading CUDA error is worse than a resource error, because it sends you
looking at the driver instead of at the limit you just set. So the limit set is
a **gate** (`epic_u2o/rlimit_cuda_gate.py`), not a comment.

## The table

| limit | value | applied | why CUDA-safe | evidence (ts job) |
|---|---|---|---|---|
| `RLIMIT_AS` | 262144 MiB | yes | must exceed the 5090's 32 GiB BAR1 mapping or CUDA context creation fails instantly; 256 GiB clears it. NOT a VRAM cap -- on ReBAR the BAR mapping dwarfs any useful value, so this is a leak backstop only. | `20260927t180239-90a79c87` |
| `RLIMIT_CPU` | 60 | yes | a wall-clock-equivalent CPU-seconds budget; it does not constrain the driver's thread creation, and a kernel that exceeds it is killed by the process-group SIGKILL path anyway. | `20260927t180239-90a79c87` |
| `RLIMIT_DATA` | 8192 MiB | yes | brk/heap only. CUDA's allocations are mmap-based and the BAR1 mapping is not charged to RLIMIT_DATA, so this bounds a runaway heap without touching device or pinned memory. | `20260927t180239-90a79c87` |
| `RLIMIT_NOFILE` | 256 | yes | measured fine on the 5090 (u2o.6.1.1 bracketed NOFILE=256 as acceptable while NPROC=32 and 512 were both fatal). The driver opens far fewer than 256 descriptors for a single-context run. | `20260927t180239-90a79c87` |
| `RLIMIT_CORE` | 0 | yes | disables core dumps; touches no CUDA state. Safe by inspection and by the fact that the 5090 runs above produced no core. | `20260927t180239-90a79c87` |
| `RLIMIT_FSIZE` | — | **no** | NOT APPLIED. The bead description lists FSIZE(min(256, quota)) as part of the set the harness applies today, but apply() never calls setrlimit(RLIMIT_FSIZE). Recorded as a discrepancy rather than written into the table as if it were enforced. | `untested` |
| `RLIMIT_NPROC` | None | yes | NEVER SET. The kernel checks RLIMIT_NPROC against the real uid's TOTAL process+thread count, not our subtree, on every clone. This uid has ~2934 tasks (ps -eL -o uid=) against 256 processes, and a kernel-side probe bracketed the live total between 2048 (aborts at first thread creation) and 3072 (works). Because the count drifts with the other tenants on this shared box, EVERY fixed value is a race -- 2048 looks generous and is already fatal. Containment goes through the cgroup v2 pids.max route in gpu-sim-u2o.6.1.2 instead. | `20260927t123630-05fd9bd0` |


## Two invariants, executable rather than documented

1. **`RLIMIT_NPROC` is never set.** `ResourceLimits.processes_limit` defaults to
   0 and a non-zero value now **raises** instead of being applied. It used to be
   `if self.processes_limit: setrlimit(RLIMIT_NPROC, ...)`, and a guard around a
   live call is a footgun with a comment on it: the field exists, the name
   looks like the containment knob it is not, and setting it kills CUDA at the
   first `cudaMalloc`. This uid has ~2934 tasks against 256 processes, and a
   kernel-side probe bracketed the live total between 2048 (aborts) and 3072
   (works) — so **2048 looks generous and is already fatal**, and because the
   count drifts with the other tenants on this box, every fixed value is a race.
   Containment goes through cgroup v2 `pids.max` (gpu-sim-u2o.6.1.2).
2. **`RLIMIT_AS` is independent of `vram_limit_mb`.** Asserted on the numbers,
   not by a comment saying so, because a comment is exactly what was there when
   the 24 GiB failure happened. `RLIMIT_AS` must also exceed 32768 MiB (BAR1).

Honest limitation, unchanged: on ReBAR GPUs `RLIMIT_AS` **cannot** serve as a
VRAM cap — the BAR mapping dwarfs any useful value — so it is a leak backstop
only, not a VRAM accounting mechanism.

## Recorded discrepancy

`RLIMIT_FSIZE` is in the table as **not applied**. gpu-sim-u2o.6.3.1's
description lists `FSIZE(min(256, quota))` as part of the set the harness
applies today, but `apply()` never calls `setrlimit(RLIMIT_FSIZE)`. Recorded
rather than written in as though it were enforced, because a safety table that
describes a different file than the one that runs is worse than no table.
`tests/test_rlimit_cuda_gate.py` fails if `RLIMIT_FSIZE` ever appears in
`safe_harness.py`, so the entry cannot go stale.

## The probe

`epic_u2o/rlimit_cuda_gate.PROBE_SOURCE` builds a probe that runs under the
harness's **exact** limit set: context creation (the BAR1 mapping),
`cudaMalloc`, one launch, `cudaDeviceSynchronize`, and a read-back. It records
the live uid task count, because `RLIMIT_NPROC` is per-uid and that number is
what makes a future reintroduction detectable rather than mysterious. Cheap
(<10 s) so it can run on every batch job; see
`experiments/gpu-sim-u2o.6.3.1/metrics.json` for a real 5090 run.

On failure, `diagnose_probe_failure()` names the offending limit by stage. When
no limit in the set explains a failure it says exactly that — "cannot diagnose"
must never degrade into "add a limit to fix it".
