# gpu-sim-u2o.6.3.1 — rlimit CUDA-compatibility gate

Two real-GPU runs have been killed by limits nobody checked against CUDA, and
both times the symptom pointed at the wrong subsystem:

| what was set | what you saw | what it actually was |
|---|---|---|
| `RLIMIT_NPROC=32`/`512` | `cudaErrorOsOperationNotSupported` at the first `cudaMalloc` | the kernel compares NPROC against the real uid's **total** task count, not our subtree (ts `20260927t123630-05fd9bd0`) |
| `RLIMIT_AS=24 GiB` | every run died in ~2 ms | the 5090's BAR1 mapping is 32 GiB |

A misleading CUDA error is worse than a resource error: it sends you to read the
driver instead of the limit you just set. So this is a **gate**, not a comment.

## What is enforced, and how

**`RLIMIT_NPROC` is never set.** `ResourceLimits.processes_limit` still exists
and defaults to 0 — but a non-zero value now **raises** instead of being
applied. It used to be:

```python
if self.processes_limit:
    resource.setrlimit(resource.RLIMIT_NPROC, (self.processes_limit, ...))
```

and a guard around a live call is a footgun with a comment on it: the field
exists, the name looks like the containment knob it is not, and setting it kills
CUDA at the first `cudaMalloc`. "Cannot contain" must never degrade into "apply
it anyway", so the refusal names the fix (cgroup v2 `pids.max`, gpu-sim-u2o.6.1.2).

Note that **2048 is refused too**. This uid has ~2934 tasks against 256
processes, and a kernel-side probe bracketed the live total between 2048 (aborts
at first thread creation) and 3072 (works). 2048 looks generous and is already
fatal; because the count drifts with the other tenants on this box, every fixed
value is a race. There is a test for 2048 specifically, because "looks safe" is
the exact reasoning that produced the incident.

**`RLIMIT_AS` is independent of `vram_limit_mb`.** Asserted on the numbers, not
by a comment saying so — a comment is precisely what was there when the 24 GiB
failure happened. The check fires both when AS is too small and when someone
couples it to the VRAM limit, which is how the failure would return.

## The gate is reachable, which took a correction

`check_rlimit_cuda_compatibility()` takes an optional `limits` object, and the
first version of my injected-failure demonstration did **not** fail. I had
injected `vram_limit_mb=24576` and expected the AS gate to trip; it correctly
did not, because lowering the VRAM limit must leave `RLIMIT_AS` alone — that is
the independence invariant *working*, not the gate being unreachable. So the
demonstration now injects a genuinely bad AS value, and there is a separate
negative control asserting that lowering `vram_limit_mb` alone leaves AS at
256 GiB and the gate green.

## Recorded discrepancy: `RLIMIT_FSIZE`

This bead's description lists `FSIZE(min(256, quota))` as part of the set the
harness applies today. `apply()` never calls `setrlimit(RLIMIT_FSIZE)`. Rather
than write a safety table describing a different file than the one that runs,
the table records FSIZE as **not applied**, and a test fails if
`RLIMIT_FSIZE` ever appears in `safe_harness.py`, so the entry cannot go stale
in the other direction either.

## Verification

* `python3 tests/test_rlimit_cuda_gate.py` — **25 checks, 0 failures**, CPU
  only, no GPU. Covers both invariants, the table/evidence consistency, the
  FSIZE discrepancy, the diagnostic path (including "no limit explains this" →
  *do not add a limit*), and four `preflight_check` wiring cases.
* One check exercises the real `apply()` in a child process with
  `processes_limit=1` and asserts it is **refused with the gate's message**
  rather than dying from an NPROC abort — the old code's two failure modes were
  indistinguishable, which is why the message is the thing being asserted.
* The table-vs-code check derives the applied set from the real
  `ResourceLimits` dataclass, so the table cannot describe a different file
  than the one that runs.

## The probe

`PROBE_SOURCE` runs under the harness's **exact** limit set, applied in the
child's `preexec_fn` — the same mechanism and values `run_kernel` uses, so a
pass means the real limit set is CUDA-safe rather than a hand-written
approximation of it. It does the three steps that actually broke (context /
`cudaMalloc` / launch + `cudaDeviceSynchronize`), reads the result back, and
records the **live uid task count**, because NPROC is per-uid and that number is
what makes a future reintroduction detectable rather than mysterious.

`rc == 0` is not accepted on its own: the read-back is asserted, so a probe that
launches and computes the wrong answer still fails. On failure
`diagnose_probe_failure()` names the offending limit by stage — and when no limit
in the set explains the failure it says exactly that, because "cannot diagnose"
must not become "add a limit to fix it".

See `metrics.json` for the real 5090 run and `job.json` for the ts job id.
