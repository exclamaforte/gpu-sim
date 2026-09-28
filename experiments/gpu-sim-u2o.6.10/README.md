# gpu-sim-u2o.6.10 — supervised single-dose validation of the kill path

**One dose. ts job `20260928t072424-9816875f`.** Queue verified empty
immediately before submitting; no tenant's job was jumped. The hang ran
**once**, with a 4 s internal timeout (the bead's 2–5 s band), watched.

**Revision exercised: `run_kernel` as it is in master @ `cd5e7da`.**
gpu-sim-u2o.6.3 restructures that same function (wrapper + `_run_kernel_inner`,
timeout handling moved across that boundary) and is still needs-rework, so
**6.3's restructure is not covered by this result**. If 6.3 lands first, re-run
this against the restructured path — same job, one more dose.

## Result: the kill path recovers this GPU

| step | outcome |
|---|---|
| 1 — normal stage-1 kernel | 20/20 trials, median **0.002768 ms**, process wall 150.2 ms → **2713× ratio** |
| 2 — `--test-hang`, once, 4 s | `timeout=True`, rc −1, wall **4.13 s**, gpu_health `healthy`, **0 survivors** |
| 2b — recovery probe | 20/20 trials, median **0.002864 ms** = **1.03×** the pre-hang median |

Telemetry: 345 MHz / 38 °C / 25.4 W / 0% before → 1357 MHz / 40 °C / 65.0 W / 0%
after, throttle reasons `0x0` throughout, 425 MiB resident before and after (no
leaked context from the killed process).

Three things this establishes, in increasing order of how much they matter:

1. **Step 1's timing is device-side.** The process spent 150 ms of wall time to
   run 55 µs of kernel work — 2713×. The number being recorded is a
   `cudaEvent` measurement, not process startup. Had these two agreed, "per-trial
   timing" would have been measuring `fork`/`exec`/context-creation.
2. **The runaway actually died.** A reported timeout is a claim; `pgrep` on the
   hang binary's path returns nothing 1 s later. The process group kill reached
   a GPU-resident infinite loop.
3. **The GPU came back.** A normal kernel immediately afterwards returns 20
   trials at 1.03× the pre-hang median, with no residual throttle flag. That is
   the claim this bead exists to test — not "the harness printed a timeout" but
   "this machine and this desktop session are usable again".

## The one check that failed, and why that is a note and not a result

The job exited 1. `overall_pass` first read `False` because
`wall_over_trials_ratio` **divided seconds by milliseconds** — my own summary
arithmetic, printing `3x` where the real ratio is 2713×, and failing the
device-side check on nothing but my unit error. The hang, the kill and the
recovery were never in doubt.

I corrected it **from the same stored run** and did not spend a second dose:
re-hanging a GPU that opguard, cclm and chessvar all share, in order to fix a
division, is the wrong trade. `report.json` carries `unit_bug_correction`
stating exactly this. The distinction matters for reading the artifact: the
measurement is sound and the summary was not, and a reader who saw only
`exit=1` would be misled about which.

This is the third time in two days that a unit error has surfaced in my own
reporting after a correct measurement — after `bytes_moved`/`achieved_GBps` on
6.6.2 and the hex `clocks_throttle_reasons` parse on 6.6.3. The pattern is
consistent enough to be worth naming: the *experiment* is right and the
*summary of the experiment* is where the error lives.

## Process

* One ts job for the whole dose; the queue was verified empty immediately
  before submitting.
* `preflight_static.py` gates the dose: it runs the module body, resolves
  every name, checks the hang timeout is inside the 2–5 s band, and **compiles
  both kernels** — nvcc is a compiler, not a device, so a failed compile costs
  a CPU second instead of a wasted hang.
* Not done, deliberately: no clock locking, no sudo, no persistence-mode change
  (6.4 stays held), and no second hang.
