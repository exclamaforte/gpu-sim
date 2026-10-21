# gpu-sim-ep3 results — per-core quiet window

**Verdict: `NO-QUIET-CORE`.** No window was run, by the pre-registered rule.

Executed by the PI (commit `lab/gpu-sim-ep3`); the implementer session wedged
mid-turn twice and stood down, and the reviewer reviewed this measurement
independently. Pre-registration: `pre_registration.md`, written before any
measurement. Protocol executed exactly as written, including §5's
fabrication-time clarification of the utilisation instrument.

## What was measured

One pass, `systemd-run --user --scope --property=CPUQuota=800%`, sampling all
**32** CPUs over **5 s** (`select_cpu.py`, raw vector in `selection.json`).

| quantity | value |
|---|---|
| sample timestamp | 2026-10-21T13:56:00Z |
| CPUs sampled | 32 |
| least-busy CPU | **cpu 0** |
| its utilisation | **0.3584** |
| min / median / max utilisation | 0.3584 / 0.5288 / 0.8984 |
| CPUs **below** the 0.25 threshold | **0** |
| loadavg (1/5/15) | 19.68 / 25.55 / 29.34 |

The selection rule (§3) is: take the least-busy CPU; if its utilisation exceeds
0.25, **run no windows** and report `NO-QUIET-CORE`. It exceeded it — the *least*
busy of 32 cores was 35.8 % occupied, i.e. **every core on this machine was more
than a third busy at sample time**. Per §3 the pass stopped; per §8 this is a
statement about the machine, not about the gate.

## Why this is the right answer rather than a non-answer

`s9j` failed for a reason that was mine: it gated quiescence on the whole *user
slice*, which contains every research lab on this machine (chessvar, cclm,
megakern, torch-oxide, pi-research-lab all run here), so the gate could only ever
fail. `ep3` fixed the framing — measure the resource actually used, i.e. the
**core** the process is pinned to.

The answer that framing returns is unambiguous: **there is no quiet core to pin
to.** With 32 cores at 36–90 % occupancy and a system load of ~20–29, this box
has no idle capacity at any granularity: not the box (`s9j`), not the core
(`ep3`). A timing measurement at a 36.2 µs requirement cannot be made here.

## What this does and does not change

- **Does not change** `m3d.10`'s `NO-CLOSE-ADMISSIBILITY`. The gate still does
  not run; the verdict stands. The tightest spread measured anywhere in this
  direction remains `vs2`'s S2_2 at **47.9 µs**, 1.32× over requirement.
- **Does change** the *attribution* of the timing limit. It is no longer "we
  happened to measure while busy" (ambient) and no longer "our own cgroup's
  parallelism" (`vs2`'s mechanism) — it is that **this machine has no
  uncontended execution resource at core granularity**, which is a measured
  property of a shared 32-core box running five research labs.
- **Does not license** any workaround. Sampling repeatedly until a quiet core
  appears would be noise-mining; the pre-registration fixes one sample per pass,
  and the pass was taken.

## Limitation of this pass — stated so the finding is not over-claimed

This is **one** 5 s sample (n=1 pass), taken at the low end of the load range
this direction has observed (`tfq` saw load 19–59 across its windows; this
sample's loadavg was 19.68/25.55/29.34). A quiet core could in principle exist at
another moment. What the sample supports is exactly the claim made — **at sample
time the least busy of 32 cores was 35.8 % occupied** — plus the inference that a
core below 25 % is unlikely on a box whose cores run 36–90 % busy.

What it does **not** support is "no quiet core ever exists here". Testing that
needs a pre-registered **multi-sample** pass (N samples over T minutes, take the
best core observed, same 0.25 threshold and decision rule) — a different
protocol, and therefore a different bead. Re-sampling now, after seeing this
result, would be sampling until it looks good, which is what this direction
pre-registered as forbidden; it was not done.

## What would answer the timing question

An execution resource with no neighbours: an exclusive core (cpu pinning plus an
isolated cpuset that no other job shares), a dedicated machine, or a
hardware-isolated measurement path. None exists here, and this bead's honest
output is that the question is **not answerable on this box** — not that the
criterion fails.

## Reproduce

```bash
cd <repo root>
systemd-run --user --scope --property=CPUQuota=800% -- \
  bash experiments/gpu-sim-ep3/run.sh
# Expect: selection.json with the 32-core vector, then
# "NO-QUIET-CORE: least-busy cpu=<n> has util=<u> > 0.25 -- NO WINDOWS RUN"
# or, if a core is ever below 0.25, three window files and a decision.
python3 experiments/gpu-sim-ep3/check_statistic_agreement.py   # AGREEMENT OK
```
