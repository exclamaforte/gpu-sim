# gpu-sim-u2o.6.1.1 — end-to-end harness verification on the real RTX 5090

The reviewer (2026-09-22) verified the three code fixes by reading only, and
left this bead open pending **one real confirmed run** of
`SafeHarness.measure_spec()` against the actual GPU. This directory is that
run.

## Result (job `20260927t145454-9c2420b3`, finished, exit 0)

Full pipeline `generator_stage1.generate_kernels()` → `kernel_driver.synthesize()`
→ `preflight_check()` (bounds + construct scan + `nvcc -ptx` + LD_PRELOAD
fakegpu smoke) → `compile_kernel()` (real nvcc, `-arch=sm_120`) →
`run_kernel()` (real cudaEvent timing on the RTX 5090):

| op | grid | block | n | trials | median | mean | cv |
|----|------|-------|---|--------|--------|------|-----|
| scale | 4×1×1 | 256 | 1024 | 20 | 0.002624 ms | 0.002528 ms | 0.248 |
| add | 8×1×1 | 256 | 2048 | 20 | 0.002592 ms | 0.002563 ms | 0.234 |
| relu | 16×1×1 | 256 | 4096 | 20 | 0.002688 ms | 0.002378 ms | 0.268 |
| saxpy | 32×1×1 | 256 | 8192 | 20 | 0.002848 ms | 0.002820 ms | 0.097 |

Device (from `nvidia-smi` inside the queued job):
`NVIDIA GeForce RTX 5090, 610.57.04, 32607 MiB`. Raw per-trial values are in
`metrics.json`; the full log is `output.log` / `ts_job_logs/20260927t145454-9c2420b3.log`.

These are launch-overhead-bound kernels (~2.6 µs, below the ~2–4 µs minimum
CUDA launch latency on this part), which is why 3 of the 4 are flagged
`is_reliable=False` by `measurement_stats` (cv > 0.15). That is the
statistics module working as designed, not a harness defect — real sized
workloads belong to the dataset beads.

## New bug found and fixed: `RLIMIT_NPROC` kills the CUDA driver

The first end-to-end attempt (`20260927t052025-1f7067db`) failed 4/4 kernels
with `OS call failed or operation not supported on this OS` at the **first**
`cudaMalloc`. Bisect (`20260927t123630-05fd9bd0`, table in `bisect.log`):

| variant | result |
|---------|--------|
| bare (no rlimits, no injected env) | OK — real trials |
| `FAKEGPU_DEVICE_MEM_MB=24576` injected | OK |
| `RLIMIT_NOFILE=256` | OK |
| `RLIMIT_NPROC=32` | **cudaErrorOsOperationNotSupported at first cudaMalloc** |
| full harness limit set (incl. NPROC=32, and 512) | **same failure** |

Root cause: the kernel enforces `RLIMIT_NPROC` against the **real UID's total
process+thread count**, not the child's own subtree, on every `clone()`. On
this shared box (cclm/chessvar/conway jobs resident) the user count is already
far above 32 or 512, so the CUDA driver cannot create its first helper thread
and the first allocation fails with a misleading "OS call" error.

Fix: `ResourceLimits.processes_limit` now defaults to `0`, meaning "leave
`RLIMIT_NPROC` untouched", and `apply()` skips the `setrlimit` in that case.
Runaway containment is unchanged — it is provided by the process-group
`SIGKILL` in `run_kernel()`'s timeout path (the mechanism the plan already
endorsed). `RLIMIT_NOFILE=256` is verified safe and left as-is.

## Other change

`kernel_driver.spec_from_stage1()` had `sys.path.insert(0,
'/home/gabe/Documents/gpu-sim')` hardcoded, so it always imported the main
checkout even when run from a lab worktree. It now derives the root from
`__file__`.

## Files

* `run.sh` — the successful end-to-end run (what `job.json` calls SUCCESS)
* `verify_harness.py` — the 4-case driver used by `run.sh`
* `diagnose_malloc.sh` — rlimit/env isolation probe (`ts` job
  `20260927t094146-52fd3783`)
* `bisect_run_kernel.py`, `run_bisect.sh` — decisive variant table +
  in-process re-run (`ts` job `20260927t123630-05fd9bd0`), log in `bisect.log`
* `metrics.json`, `output.log`, `job.json`, `ts_job_logs/`

## Not done here (deliberately)

The `--test-hang` kill path is **not** exercised: that is
gpu-sim-u2o.6.10 (supervised validation), which needs a watched session
because the 5090 also drives the display.
