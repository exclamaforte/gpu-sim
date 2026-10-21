# gpu-sim-99g pre-registration — written 2026-10-21 before any measurement

Executed by the PI; the reviewer reviews it independently. The implementer stands
down from this bead (it wedged three times and once overwrote a running bead's
pre-registration mid-run).

## 0. What this bead is, and what it is not

`gpu-sim-3ba` found a quiet core (cpu 8, 0.2194) and ran three windows on it:
spreads **15.2 µs, 915.4 µs, 25.3 µs** — two of them **inside** the 36.2 µs
requirement — but the pre-registered rule refused them because the core's
utilisation read 0.35 / 0.35 / 0.2632 over a 200 ms span **containing** the window.
That span included the measurement's own ~25 ms of CPU (~12 %), i.e. **the gate
charged the workload for its own execution**. `3ba`'s verdict
(`NO-ADMISSION-QUIET-CORE`) stands and is **not** reinterpreted here.

This bead corrects the instrument and re-tests, with the correction stated **as a
correction**, and with all rules fixed before data. **It is explicitly forbidden to
reinterpret `3ba`'s numbers**; only new measurements count here.

## 1. Corrected instrument (fixed before data)

For the pinned core, over a span containing the window:

```
core_busy_us      = (total_ticks_delta - idle_ticks_delta) * tick_us
our_scope_cpu_us  = this cgroup's cpu.stat usage_usec delta   (measuring scope only)
other_cpu_us      = core_busy_us - our_scope_cpu_us
other_utilisation = other_cpu_us / span_us
tick_us           = 1e6 / (getconf CLK_TCK)                   (read, not assumed)
span              = 200 ms
```

The earlier instrument measured "core busy", which necessarily includes the thing
being measured; this one measures "core busy **with something other than us**",
which is what the criterion was always trying to express.

**Threshold 0.25 and requirement 36.2 µs are unchanged.** Only the definition of
"quiet" is corrected.

## 2. Selection and immediate use (unchanged from 3ba)

- Sample all 32 cores over 5 s; take the least-busy. At selection time no
  measurement is running, so the selection figure carries no self-charge.
- If it is below 0.25, run the three windows on it **immediately**, in the same
  script — a quiet moment will not wait for a second command.
- If not, hunt up to 20 samples at 30 s spacing, then stop. The pass is not
  extended because early samples look promising.

## 3. Windows (fixed)

3 windows; mask verified **inside** each measurement and must be exactly the
pinned core, else `INVALID` and not counted. Per window record: core ticks total
and idle before/after, scope `usage_usec` before/after, span, derived
`other_utilisation`, spread (verdict.py statistic: drop first value, drop >1e9 µs
timer artifacts, max − min), the four run durations, throttle deltas, timestamps.

## 4. Decision rule (fixed)

**ADMITTED** iff at least 2 of 3 windows have (a) `other_utilisation < 0.25` and
(b) `spread ≤ 36.2 µs`, **and** the median spread of those qualifying windows is
≤ 36.2 µs. Otherwise **`NO-ADMISSION-CORRECTED-INSTRUMENT`**.
If no quiet core appears in 20 samples: **`NO-QUIET-CORE-IN-20-SAMPLES`**.

Context fields (loadavg, other-lab activity) are **non-decisional**.

## 5. Outcomes, fixed in advance

- **`ADMITTED`**: the criterion is achievable on this box on a quiet core,
  demonstrated by a pre-registered instrument. `gpu-sim-m3d.10`'s `NO-CLOSE` is then
  superseded by an actual gate run at this operating point under the ORIGINAL gate
  protocol, with instrument, pinning and environment documented in the gate
  artifact. **No gate run inside this bead.**
- **`NO-ADMISSION-CORRECTED-INSTRUMENT`**: with the self-charge removed the windows
  still fail on spread, or the qualifying median exceeds 36.2 µs. Then the floor is
  real, the criterion is **not** achievable here, and `3ba`'s two windows are to be
  understood as two lucky samples rather than evidence — **and that must be said
  plainly.**
- **`NO-QUIET-CORE-IN-20-SAMPLES`**: report the distribution and stop.

## 6. Requirement constant (pre-computed)

`experiments/gpu-sim-m3d.10/verdict_out.json`, `arithmetic`:
`d_pred_M1M2_us = 362.3`, `max_tolerable_spread_us = 36.2`.

## 7. Reporting obligation

`3ba`'s verdict is quoted as **unchanged** in this bead's results, next to this
bead's verdict, so the two cannot be confused by a later reader.
