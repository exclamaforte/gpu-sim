# Harness design notes.
#
# Sections that have NOT been written yet are listed as placeholders below and
# are left visible rather than deleted: a visibly missing section is
# information a silently ignored one is not. gpu-sim-u2o.6.11 classifies this
# file as VOLATILE rather than as a stub, because gpu-sim-u2o.6.1.2 rewrote it
# while it was still untracked -- see the note at the bottom.
#
# placeholder gpu-sim-u2o.6.3   (resource limits: VRAM caps, disk quota) -- written on
#   lab/gpu-sim-u2o.6.3, not yet merged.
# placeholder gpu-sim-u2o.6.4   (clock locking) -- PARKED, needs root and a human.
# placeholder gpu-sim-u2o.6.6   (repeated-trial statistics) -- written on
#   lab/gpu-sim-u2o.6.6.1.
# placeholder gpu-sim-u2o.6.7   (throughput instrumentation) -- not started.

## Per-run task containment (gpu-sim-u2o.6.1.2)

Mechanism: **cgroup v2 `pids.max`**, applied per run. `RLIMIT_NPROC` is never
set to a fixed value on this host.

Why not RLIMIT_NPROC: the kernel checks it against the real UID's **total**
process *and thread* count, not the child subtree. On this box that is ~3 000
tasks for uid 1000 (`ps -eL -o uid=`, not the ~256 processes `ps -u` reports),
and the CUDA driver then fails its first thread creation, so the first
`cudaMalloc` returns `cudaErrorOsOperationNotSupported`
("OS call failed or operation not supported on this OS"). Measured in
gpu-sim-u2o.6.1.1 (ts `20260927t123630-05fd9bd0`): NPROC=32 and 512 both
fatal, and 2048 is still below the live count. The live count drifts with the
other tenants, so any fixed value is a race. Unset is the only safe value.

Three mechanisms, probed at run time (`epic_u2o/task_limits.py`), in order:

1. **`systemd-run --user --wait --pipe --collect -p TasksMax=N -- <cmd>`** —
   the route actually selected on this host. `--wait` keeps it synchronous so
   the harness's own `communicate(timeout=)` still governs the wall clock;
   `--pipe` keeps stdout/stderr on our pipes; `--collect` tears the unit down
   with the run instead of leaving units to accumulate.
2. **A delegated cgroup** — mkdir under the user's own cgroup, write
   `pids.max` **before** exec, move the child in. Available here (the user's
   cgroup directory is writable and the `pids` controller is in
   `cgroup.controllers`), but mechanism 1 is preferred because it needs no
   cgroup lifetime management of our own.
3. **RLIMIT_NPROC relative to the live count** (`live_task_count() +
   headroom`) — fallback only, and never a constant. It is still coupled to
   other tenants' activity, which is an accepted residual risk rather than a
   fix.

Default `ResourceLimits.tasks_max = 256`; `SafeHarness(tasks_max=0)` disables
it, which is not the default.

Enforcement is **measured, not assumed**: `tests/test_task_limits.py` shows a
child failing to fork with `EAGAIN` after 63 children under `TasksMax=64`,
the same child forking 200 children when uncapped, and a realistic small
child being unaffected by the default cap.

**Limits of this mechanism, stated plainly:** `TasksMax` bounds threads and
processes in a subtree. It does **not** bound device memory — that is the VRAM
accounting in gpu-sim-u2o.6.3, and the two are complementary, not
substitutes. It also does not bound a single process's memory use, only the
number of tasks. And like every cgroup route, it needs the pids controller
and a writable delegated cgroup; where neither exists, mechanism 3 is a
weaker substitute and the process-group SIGKILL on timeout remains the
backstop that always applies.

---

## Why this file is tracked, and what that means for its classification

This file was a four-line stub in the main checkout and untracked in master
when gpu-sim-u2o.6.11 classified `epic_u2o/`. That classification -- "literal
placeholder, do not track" -- was TRUE of the file as it stood, and it became
false the moment gpu-sim-u2o.6.1.2 rewrote it with the real containment
ladder, while the inventory row kept asserting the stub justification.

Leaving it untracked was worse than wrong: it is the one excluded path that
an in-flight bead also modifies, so it kept 6.1.2's merge refusing with
"untracked working tree files would be overwritten" -- the exact collision
6.11 was raised to remove. And the section it carries is the recorded reason
`processes_limit` is 0, which is the class of documented-but-untracked loss
this epic has already paid for once.

So it is TRACKED, carrying 6.1.2's content rather than the stub: a tracked stub
would have closed the collision only by putting a known-false file in master.
The 6.1.2 merge is now a small modification instead of a blocked add.

---

<!-- BEGIN GENERATED: gpu-sim-u2o.6.3.1 rlimit table -->
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
| `RLIMIT_NPROC` | — | **no** | NEVER SET. The kernel checks RLIMIT_NPROC against the real uid's TOTAL process+thread count, not our subtree, on every clone. This uid has ~2934 tasks (ps -eL -o uid=) against 256 processes, and a kernel-side probe bracketed the live total between 2048 (aborts at first thread creation) and 3072 (works). Because the count drifts with the other tenants on this shared box, EVERY fixed value is a race -- 2048 looks generous and is already fatal. Containment goes through the cgroup v2 pids.max route in gpu-sim-u2o.6.1.2 instead. | `20260927t123630-05fd9bd0` |


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

<!-- END GENERATED: gpu-sim-u2o.6.3.1 rlimit table -->
