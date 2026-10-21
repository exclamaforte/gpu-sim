# gpu-sim-52i pre-registration — written 2026-10-21 before any measurement

Executed by the PI; reviewed independently by the reviewer. The implementer stands
down from this bead.

## 0. The hypothesis under test, stated so that it can FAIL

`gpu-sim-99g` found its "quiet" logical CPU 8 shared a physical core with cpu 24,
which ran at **0.94–1.0** utilisation throughout, and that its spreads were
**bimodal** (35.1 µs in one window; 1878.9 and 388.9 µs in the others) while the
*same* 100 000-iteration workload ran ~5.9 ms in the tight window and ~3.3–5.2 ms in
the loose ones.

**Prediction, fixed before measurement: the bimodality is SMT sibling contention.**
If so, on a **physical core whose BOTH siblings are quiet**, the three windows will
be **consistently inside the 36.2 µs requirement**. If they are not, the hypothesis
is **refuted** and the cause lies elsewhere (frequency/thermal state, memory or
shared-cache contention, or the timer path) — and that refutation is the result.

## 1. Topology-aware selection (fixed)

- Read `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`; build the
  physical pairs (this box: 16 pairs).
- Sample all 32 logical CPUs over 5 s; compute each pair's **worst-sibling**
  utilisation.
- Select the pair with the **lowest worst-sibling** utilisation (ties → lower
  sibling index).
- **Threshold 0.25** (unchanged): both siblings must be below it. If the best pair's
  worst sibling is ≥ 0.25, hunt up to **20 samples at 30 s spacing**, then stop and
  report `NO-BOTH-QUIET-CORE-IN-20-SAMPLES` with the distribution of per-sample
  best-pair worst-sibling values.
- The hunt is **not extended** because early samples look promising.

## 2. Immediate use (the point)

The moment a pair qualifies, the three windows run in the **same script**, pinned to
the **lower-index sibling** of that pair. A quiet moment will not wait for a second
command.

## 3. Instrument — corrected, extended to the pair (fixed)

For each window, over a 200 ms span containing it:

```
other_cpu_us      = Σ over both siblings [(total_ticks_delta − idle_ticks_delta) × tick_us]
                    − our_scope_cpu_us
other_utilisation = other_cpu_us / span_us
tick_us           = 1e6 / (getconf CLK_TCK)     [read, not assumed]
our_scope_cpu_us  = this cgroup's cpu.stat usage_usec delta (measuring scope only)
```

Our own CPU time is subtracted **once** — it can only run on the pinned sibling.
Per-sibling utilisation is recorded **separately** as an input to the verdict as
well as in the combined figure.

## 4. Windows and statistic (fixed)

3 windows; mask verified **inside** each measurement and must be exactly the pinned
logical CPU, else `INVALID` and not counted. Spread by the `verdict.py` statistic
(drop first value, drop > 1e9 µs timer artifacts, max − min); agreement check
re-run. Record the four run durations, throttle deltas, timestamps.

## 5. Decision rule (fixed) and refutation branch

**ADMITTED** iff ≥ 2 of 3 windows have (a) **both siblings' other-utilisation**
below 0.25 and (b) spread ≤ 36.2 µs, **and** the median spread of the qualifying
windows ≤ 36.2 µs.

Otherwise **`NO-ADMISSION-BOTH-QUIET`** — meaning both siblings were quiet and the
spread still exceeded the requirement in a majority of windows. **That refutes SMT
contention as the cause of the bimodality**, and the artifact must then name the
remaining suspects (core frequency/thermal, memory or shared-cache contention,
timer path). `3ba`'s and `99g`'s sub-requirement windows are then to be understood
as **correlation with a quiet sibling rather than with a quiet core**, and that must
be said plainly.

No qualifying pair in 20 samples → **`NO-BOTH-QUIET-CORE-IN-20-SAMPLES`**; report
the distribution and stop. Do not run windows on a pair whose sibling is busy and
call the result a limit.

Context fields (loadavg, other-lab activity) are **non-decisional**.

## 6. Outcomes, fixed in advance

- **ADMITTED**: the criterion is achievable on this box on a both-siblings-quiet
  physical core, and the SMT hypothesis is supported. `gpu-sim-m3d.10`'s `NO-CLOSE`
  is then superseded by an actual gate run at this operating point under the
  ORIGINAL gate protocol, with instrument, pinning, topology and environment
  documented in the gate artifact. **No gate run inside this bead.**
- **`NO-ADMISSION-BOTH-QUIET`**: SMT contended is refuted as the cause; name the
  remaining suspects and stop treating sub-requirement windows as evidence.
- **`NO-BOTH-QUIET-CORE-IN-20-SAMPLES`**: report and stop.

## 7. Constants (pre-computed, not re-derived)

`experiments/gpu-sim-m3d.10/verdict_out.json`, `arithmetic`:
`d_pred_M1M2_us = 362.3`, `max_tolerable_spread_us = 36.2`.
Threshold 0.25 and requirement 36.2 µs are **unchanged** from `ep3`/`3ba`/`99g`;
what changes is that "quiet" now refers to the **physical core**.

## 8. Reporting obligation

`99g`'s verdict (`NO-ADMISSION-CORRECTED-INSTRUMENT`) and `3ba`'s
(`NO-ADMISSION-QUIET-CORE`) are quoted as **unchanged** in this bead's results,
next to this bead's verdict, so no later reader can confuse them.
