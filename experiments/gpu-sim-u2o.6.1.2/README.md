# gpu-sim-u2o.6.1.2 — per-run task containment via cgroup v2 `pids.max`

Disabling `RLIMIT_NPROC` (gpu-sim-u2o.6.1.1) left 6.1 with no per-tree
process/thread cap at all. This bead puts one back, with the semantics that
were actually wanted: count the tasks in *one run's* subtree, not the real
UID's total.

One ts job, `20260927t164508-b0857879` (finished), covering both halves.

## Mechanism, chosen at run time

`epic_u2o/task_limits.py` probes the host and takes the first available of:

| # | mechanism | available here | why |
|---|---|---|---|
| 1 | `systemd-run --user --wait --pipe --collect -p TasksMax=N -- <cmd>` | **yes — selected** | `--wait` keeps our `communicate(timeout=)` authoritative, `--pipe` keeps stdout/stderr on our pipes, `--collect` leaves no units behind |
| 2 | delegated cgroup: mkdir under our own cgroup, write `pids.max` before exec, move the child in | yes | needs cgroup lifetime management of our own; mechanism 1 is simpler |
| 3 | `RLIMIT_NPROC` = **live task count at apply time + headroom** | available but never chosen | still coupled to other tenants; a residual risk, not a fix |
| 4 | none | — | only the process-group SIGKILL |

Measured on this host: cgroup2fs, `pids` in `cgroup.controllers`,
`/usr/bin/systemd-run`, our own cgroup directory writable, **live task count
~3 000** for uid 1000.

## Enforcement is observed, not configured

`tests/test_task_limits.py`, CPU-only, 8 checks, 0 failures — the point being
that creating a unit with `TasksMax=64` proves nothing until something
actually fails to fork under it:

* uncapped child forks **200 children, succeeds**;
* the same child under `TasksMax=64` **fails with `EAGAIN` after 63 children**;
* a small child is unaffected by the default cap (`TasksMax=256`);
* the fallback computes `nproc` from the live count (measured 3058 → 3146),
  never a constant;
* the chosen mechanism is a cgroup route, not `RLIMIT_NPROC`.

## A real kernel still runs, at the same speed

`verify_capped_run.py`, hand-written saxpy, `n=8192`, grid `32x1x1`, block 256,
3 invocations, capped and uncapped:

| | median |
|---|---|
| capped (`TasksMax=256`) | **0.00272 ms** |
| uncapped (`tasks_max=0`) | 0.002704 ms |
| 6.1.1 reference (ts `20260927t145454-9c2420b3`) | 0.002848 ms |

Within the 10% tolerance, and the cap costs nothing measurable: the capped
and uncapped medians differ by 0.6%, which is inside the run-to-run spread
6.6.1 measured for this regime. Telemetry before and after: 2467 MHz, 41 °C,
throttle reasons `0x0` — no throttling, so the comparison is clean.

## Limits, stated rather than implied

`TasksMax` bounds threads and processes in a subtree. It does **not** bound
device memory (that is 6.3's VRAM accounting — complementary, not a
substitute), and it does not bound a single process's memory, only the task
count. And every cgroup route needs the `pids` controller plus a writable
delegated cgroup; where neither exists the relative-`RLIMIT_NPROC` fallback is
a weaker substitute and the process-group SIGKILL on timeout remains the
backstop that always applies. All of this is in `epic_u2o/harness.md`, which
previously held only `# placeholder` lines.

## Files

* `epic_u2o/task_limits.py` (new) — probing, selection, command wrapping
* `epic_u2o/safe_harness.py` — `ResourceLimits.tasks_max`, `SafeHarness(tasks_max=)`,
  the run is wrapped and the mechanism recorded in `metadata['task_limits']`
* `epic_u2o/harness.md` — the mechanism and its limits
* `tests/test_task_limits.py` (new) — the CPU enforcement test
* `experiments/gpu-sim-u2o.6.1.2/` — `verify_capped_run.py`, `run.sh`,
  `metrics.json`, `job.json`, logs
