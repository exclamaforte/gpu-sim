# gpu-sim-ep3 pre-registration — written 2026-10-21 before any measurement

Executed by the PI (the implementer session wedged mid-turn twice and stood down;
the reviewer reviews this measurement independently). The rules below are fixed
before data. The ONE clarification in §5 was made at implementation time, before
any window was run, because the protocol as originally written was not
measurable — it is flagged as such rather than quietly applied.

## 1. Question

`gpu-sim-s9j` could not obtain a quiesced window and said so: other processes
consumed 17–19 CPU-seconds per 1-second window. That gate was wrong — it measured
the whole *user slice*, which contains every research lab on this machine, so it
could only ever fail. Timing does not need a quiet **box**; it needs a quiet
**core**, because a single descheduling event is what inflates the spread.

Does a quiet core admit? I.e. on the least-busy CPU of this box, does the
per-session spread fall at or below the gate's requirement of **36.2 µs**?

## 2. Requirement (pre-computed, not re-derived)

From `experiments/gpu-sim-m3d.10/verdict_out.json`, field `arithmetic`:

- `d_pred_M1M2_us = 362.3`
- `implied_d_pred_over_10 = 36.23`
- `max_tolerable_spread_us = 36.2`

An operating point admits only if its measured spread is **≤ 36.2 µs**.

## 3. Selection rule (fixed)

1. Sample per-CPU utilisation of **all 32 CPUs** over a **5 s** window from
   `/proc/stat` deltas (same code path as `vs2`/`tfq`).
2. Take the **least-busy** CPU as the target.
3. If the least-busy CPU's utilisation **exceeds 25 %**, **DO NOT RUN ANY
   WINDOW**: report `NO-QUIET-CORE` with the sampled utilisation vector.
   The 25 % figure means "less than a third of the capacity the timed workload
   itself uses"; it was chosen before measurement and is not tuned.

## 4. Setting and pinning (fixed)

- Quota: `800000 100000` (8 CPUs) — the `vs2` S2 setting, obtained with
  `systemd-run --user --scope --property=CPUQuota=800%`.
- The measuring process is pinned with `taskset -c <target_cpu>` for the whole
  window.
- **The affinity mask is verified INSIDE the measurement** with
  `os.sched_getaffinity(0)`. A window whose mask is not exactly `{target_cpu}`
  is reported `INVALID` and is **not counted** toward the verdict.

## 5. What "utilisation during the window" means — fabrication-time clarification

The pre-registration said "the target CPU's utilisation during the window".
That is **not measurable as written**: `/proc/stat` updates at `USER_HZ` = 100,
i.e. 10 ms per tick, while a window is ~20 ms (4 runs × ~5 ms), which would give
2 ticks and a utilisation estimate quantised to {0 %, 50 %, 100 %}.

Clarification, made before any measurement: the utilisation used for the rule is
measured over a **200 ms span that contains the window** (20 ticks), recorded in
the session file as `target_cpu_util_window_span`. The 25 % threshold, the
decision rule and the workload are unchanged. This is a definition of the
measurement instrument, not a change to the criterion, and it is stated here so a
reader can reject it if they disagree.

## 6. Windows and statistic (fixed)

- **3 windows**, one unattended scripted pass, no agent activity between them.
- Workload identical to `vs2`/`tfq`: `100000` iterations of `s += i*i`, `4` runs
  per window (1 warmup + 3 measured).
- Statistic identical to `verdict.py` via `spread.py` (drop first value, drop
  values > 1e9 µs timer artifacts, spread = max − min), with the agreement check
  re-run against `verdict.py` on the same raw log.
- Per window, record: target CPU id, `target_cpu_util_window_span`, affinity mask
  before and after, the four run durations, spread, quota, throttle deltas,
  timestamps.

## 7. Decision rule (fixed)

**ADMITTED** iff at least 2 of the 3 windows satisfy
(a) `target_cpu_util_window_span < 0.25` **and** (b) `spread ≤ 36.2 µs`,
**and** the median spread of those qualifying windows is **≤ 36.2 µs**.

Anything else is **`NO-ADMISSION-QUIET-CORE`**.
If §3 stopped the pass, the verdict is **`NO-QUIET-CORE`**.

**Context fields — machine load, top CPU consumers, other-lab activity — are
NON-DECISIONAL.** They are recorded as description and may not admit or exclude a
window. That was `s9j`'s error, corrected here.

## 8. What each outcome means (fixed)

- **ADMITTED**: a quiet core on this box admits. `gpu-sim-m3d.10`'s NO-CLOSE is
  superseded by an actual gate run at this operating point under its ORIGINAL
  protocol, with the pinning and the environment documented. No gate run here.
- **NO-ADMISSION-QUIET-CORE**: even on a core below 25 % the spread exceeds
  36.2 µs, so the floor is not contention but the timer/scheduler path at this
  granularity. That is the strongest, most transferable form of the finding and
  no configuration on this box fixes it.
- **NO-QUIET-CORE**: no core below 25 % existed at sample time. Report the
  vector and stop; do not measure on a busy core and call it a limit.
