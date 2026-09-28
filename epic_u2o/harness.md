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
