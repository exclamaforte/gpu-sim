# gpu-sim-u2o.6.3 — resource limits: VRAM caps, disk quota, runaway kill

Reopened by the fabrication audit (u2o.15) with the note "backing doc
`harness.md` is the literal line `# placeholder gpu-sim-u2o.6.3` — no
implementation content at all". This directory is the implementation and its
evidence.

## Result: 19/19 checks pass on the real RTX 5090

ts job `20260927t153415-d811e1d5` (finished, exit 0), device
`NVIDIA GeForce RTX 5090, 610.57.04, 32607 MiB`. Machine-readable results in
`metrics.json`; full log in `output.log` / `ts_job_logs/`. The CPU-only suite
(20 checks, no GPU) is `test_resource_limits.py` → `cpu_tests.log`.

### VRAM — two layers, because one layer is not enough

`RLIMIT_AS` **cannot** cap VRAM on this box: the 5090 is ReBAR, so the 32 GiB
BAR1 mapping dwarfs any address-space cap (measured: 24 GiB kills every run
at context creation, 256 GiB works — that is already in `ResourceLimits`).
So the cap is:

1. **Static, pre-launch.** `resource_limits.check_vram_budget()` sizes every
   pointer argument from `problem_size` and refuses the spec before the GPU is
   touched. Measured: an 8 GiB spec against a 2048 MB cap is rejected in
   **0.15 s** with `VRAM budget exceeded: need 9216.0 MB (allocations
   8192.0 MB + 1024 MB context reserve), limit 2048 MB`. Argument types it
   cannot size are **refused, not guessed** — a silent under-count is exactly
   how a cap gets bypassed.
2. **Device-side audit.** The synthesized driver now prints its own
   `cudaMemGetInfo` delta (free-before minus free-after) and the harness
   applies the cap to *that* via `enforce_device_vram_cap()`. Measured for
   3×4 MiB buffers: `delta_mb = 12.0` — exactly the static estimate — while
   `device_used_mb = 1041.9` (context + other tenants).

The `delta` vs `device-wide` distinction is the point: this GPU is shared, so
capping the device-wide figure would flag innocent runs whenever a neighbour
is busy. Both numbers are reported, only the delta is enforced.

### Disk — accounted quota plus a hard per-file backstop

`resource_limits.ScratchDir` gives each run its own directory, refuses to
start when the base tree is already over quota, accounts bytes written, and
removes the directory afterwards. `SafeHarness(run_kernel=..., scratch_base=...)`
wires it in and records `metadata['disk']` on every run. Measured: a child
writing 1 MB stays within a 2 MB quota; a child that asks for 6 MB is killed
by the `RLIMIT_FSIZE` backstop at exactly 2 MB (`rc=-25`, SIGXFSZ) — the
backstop fires *before* the Python-side accounting can even see an overage,
which is the intended layering. The quota is per-directory and enforced in
Python; the rlimit is per-file and is honestly labelled as only a backstop.

### Runaway kill — the ladder, with the NPROC lesson folded in

`resource_limits.KILL_RULES` is the explicit, testable table:

| tier | trigger | action |
|---|---|---|
| 0 | wall-clock timeout | SIGKILL the whole process group |
| 1 | after any timeout | one GPU health check; `SOFT_RESET` only if hung/xid/off-bus |
| 2 | host memory / disk exhaustion | `RLIMIT_DATA` + per-run disk quota + `RLIMIT_FSIZE` |
| 3 | GPU unusable after SOFT_RESET | STOP, report to the owner — never auto-recover further |

`processes_limit` defaults to **0 = never touch `RLIMIT_NPROC`**, carrying
forward the u2o.6.1.1 finding (NPROC=32 *and* 512 both made the CUDA driver
fail its first `cudaMalloc` on this shared box, ts job
`20260927t123630-05fd9bd0`). A too-tight limit is not a safety feature, it is
an outage; a unit test now pins that default.

Measured runaway behaviour without touching the GPU: a host-side infinite
loop is killed by the timeout path at 3.00 s, `timeout=True`, nothing from
the process group survives (`pgrep` finds nothing), and the harness records
the post-timeout GPU health check (`"healthy"`).

## Files

* `epic_u2o/resource_limits.py` — `ResourcePolicy`, `estimate_device_bytes()`,
  `check_vram_budget()`, `ScratchDir`, `KILL_RULES` (the deliverable)
* `epic_u2o/safe_harness.py` — VRAM budget in `preflight_check`, device-side
  cap in `measure_spec`, disk quota in `run_kernel`, `NPROC` default fixed
* `epic_u2o/kernel_driver.py` — driver reports its own `cudaMemGetInfo` delta
* `test_resource_limits.py` — 20 CPU-only checks (safe to run outside `ts`)
* `verify_budgets.py` + `run.sh` — the GPU-side validation (19 checks)

## Not done here (deliberate)

The GPU-spinning `--test-hang` kernel is still **not** exercised: that is
gpu-sim-u2o.6.10 (supervised validation, display GPU). The runaway check here
is host-side, which proves the kill mechanics without wedging the display.
