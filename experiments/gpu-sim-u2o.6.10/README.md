# gpu-sim-u2o.6.10 — supervised single-dose validation of the kill path

**One dose. ts job `20260928t072424-9816875f`.** Queue verified empty
immediately before submitting; no tenant's job was jumped. The hang ran
**once**, with a 4 s internal timeout (the bead's 2–5 s band), watched.

**Revision exercised: `run_kernel` as it is in master @ `cd5e7da`.**
gpu-sim-u2o.6.3 restructures that same function (wrapper + `_run_kernel_inner`,
timeout handling moved across that boundary) and is still needs-rework, so
**6.3's restructure is not covered by this result**. If 6.3 lands first, re-run
this against the restructured path — same job, one more dose.

## Re-run dose, 2026-10-04 (criterion 7: 6.3 has landed)

**Revision exercised: `run_kernel` as restructured by 6.3 in master @
`c4072f3`** (wrapper + `_run_kernel_inner`). Staged one stage per ts job
with code gates between them, under a ts-plus-active-poll waiver for the
watched-foreground requirement (single dose only):

| stage | ts job | outcome |
|---|---|---|
| step 1 — normal kernel | `20261004t175237-eedb9478` | 20/20 trials, median **0.002832 ms** = **1.02x** the pre-hang baseline 0.002768 ms, device-side confirmed |
| step 2 — `--test-hang`, once, 4 s | `20261004t175333-0ae284fd` | `timeout=True`, rc −1, wall **4.16 s**, gpu_health `healthy`, **0 survivors** (in-job check plus independent outside recheck) |
| step 2b — recovery probe | `20261004t175527-b038b1a4` | 20/20 trials, median **0.002592 ms** = **0.92x** pre-hang, recovered |

Verdict `overall_pass=True` in `report.json` (assembled CPU-side from the
three stage files). Gate evidence per stage in `gate_evidence_step*.json`
(empty queue listing + idle device snapshot immediately before each submit:
1%/15 W and 0%/26 W). Per-step `queue_state` sampled in the stage that ran
it, each identifying its own job id in-record with `n_other_active=0`.

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

## The one item: queue state attached to nothing

The reviewer's finding, which I agree with and which is now fixed in code: the
queue record was **correct but attached to nothing**. It was sampled once at
the start of the run and then implicitly applied to all three steps, so no
individual measurement carried its own. A record that is not attached to the
measurement it governs cannot be checked against it.

Fixed: `step_state()` now samples queue + telemetry **per step**, and each of
`step1` / `step2` / `step2b` carries its own. It also reports
`own_job_entry_present` explicitly rather than assuming it.
`tests/test_step_queue_state.py` asserts the attachment — 12 checks, CPU-only,
no dose — and it fails on the pre-fix script (5 failures) and passes on this
one.

**This artifact is not re-run, and should not be.** It predates
gpu-sim-u2o.6.6.6's `queue_state` reaching master, so `own_job_entry` does not
exist in it. `report.json` now carries `queue_artifact_limitation` saying so,
recording what remains valid (own_job_id `20260928t072424-9816875f` non-null,
queue observed empty, and the verdict does not depend on per-step attachment)
and what is not verifiable (the `n_other_active=0` reading cannot be
cross-checked in-record, so it is an ordering artefact rather than a verified
fact). Re-hanging a shared GPU to attach a field the next run carries anyway
spends shared-box time to produce a record that is superseded by construction.

Merge-comment line the PI asked to be carried, verbatim in substance: *this
artifact's queue_state predates `own_job_entry`, so its emptiness reading is an
ordering artefact and not independently checkable; the verdict stands, and the
check becomes possible for runs after gpu-sim-u2o.6.6 lands.*

## Process

* One ts job for the whole dose; the queue was verified empty immediately
  before submitting.
* `preflight_static.py` gates the dose: it runs the module body, resolves
  every name, checks the hang timeout is inside the 2–5 s band, and **compiles
  both kernels** — nvcc is a compiler, not a device, so a failed compile costs
  a CPU second instead of a wasted hang.
* Not done, deliberately: no clock locking, no sudo, no persistence-mode change
  (6.4 stays held), and no second hang.
