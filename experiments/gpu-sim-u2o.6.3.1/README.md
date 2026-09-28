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

### Real 5090 result (authoritative: ts `20260928t091903-9d586536`)

```
applied_limits: {RLIMIT_AS: 274877906944, RLIMIT_CPU: 60, RLIMIT_DATA: 8589934592,
                 RLIMIT_NOFILE: 256, RLIMIT_CORE: 0}
cuda_malloc: ok   launch: ok   result_correct: true   rc: 0
device: NVIDIA GeForce RTX 5090      uid_tasks: 2006      0.70s
```

`cudaFree(0) → cudaMalloc → launch → cudaDeviceSynchronize → read-back`, all
under the harness's exact limit set applied in the child's `preexec_fn`. 0.70 s
against the design's <10 s budget, so it can gate every batch job.

## The probe caught its own author — which is the part worth keeping

The **first** probe run (ts `20260928t091846-a635d8f7`) reported `rc=0`,
`cuda_malloc: ok`, `launch: ok`, `device: NVIDIA GeForce RTX 5090` — and
`result_correct: false`.

That was my bug: the kernel doubled **uninitialised** device memory because I
never copied the host buffer in. Every signal a naive probe checks said pass.

The only reason it surfaced is that the probe asserts its **read-back** instead
of trusting `rc`. A probe that had trusted the exit code would have gone green
forever while verifying nothing — the same shape as the preflight stub this lab
closed out in `gpu-sim-8dw`, one layer down: a check that cannot fail. The failed
job is kept in `job.json` deliberately rather than deleted.

## The live task count, and why it is the strongest evidence here

The probe records the uid's task count because `RLIMIT_NPROC` is per-uid. Across
my two runs, two minutes apart: **1979 → 2006**, a drift of **27 tasks**.

gpu-sim-u2o.6.1.1 bracketed this uid's live total between **2048** (aborts the
process at its first thread creation) and **3072** (works). The count is now
**2006** — *42 tasks below the abort threshold* — and it moves by tens in
minutes as the other tenants come and go.

So the gate's central claim is not a style preference: **every fixed
`RLIMIT_NPROC` value is a race on this host**, and 2048, which reads as a
generous limit, is currently within a rounding error of killing CUDA. A limit
that is fatal today may work tomorrow and fail again next week, which is exactly
the failure mode that costs a whole measurement run and gets misdiagnosed as a
driver problem.

## Queue hygiene incident, recorded rather than cleaned up quietly

I ran `ts queue`, `ts status`, `ts logs ID` and `ts cancel ID` believing they
were queries and commands. This front end accepts only bare `ts` for status,
`ts -c` for logs and `ts -k` for cancel — **any other leading word is taken as
the COMMAND to run**. I therefore submitted **six junk jobs** into a queue shared
with cclm, opguard and chessvar, three of them ahead of my real probe, and the
first real attempt failed with `no such executable: submit` for the same reason.

All six were cancelled and the queue verified clean, but the cost was other
people's queue time and the lesson belongs in the record: on a shared queue, an
unrecognised flag is not an error message, it is a job.
