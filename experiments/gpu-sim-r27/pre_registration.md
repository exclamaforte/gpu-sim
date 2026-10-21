# gpu-sim-r27 pre-registration — written 2026-10-21 before any window

Executed by the PI; reviewed independently. **This is the last bead of the timing
thread** — if it does not land a mechanism, the thread is consolidated into
`REPORT.md` and closed.

## 0. What y1h proved about the instrument

Twelve windows pinned to cpu 7 (sibling cpu 23), sibling utilisation averaged over
200 ms per window. Four windows at sibling_util **0.2495 / 0.2496 / 0.2497 / 0.2497**
produced spreads of **423.8 / 986.4 / 79.2 / 757.7 µs** — a **12× range at identical
measured load**. A 200 ms average therefore cannot predict which windows blow up:
the inflating events live inside single runs (~6 ms), while `/proc/stat` ticks at
10 ms and the instrument averaged 200 ms.

The SMT hypothesis is still **untested**. This bead tests it at the only scale that
can, and names the occupier.

## 1. Setup (fixed)

- **Target cpu 7, sibling cpu 23** — same pair as `y1h`, so the two are comparable.
- **12** windows, back-to-back, `taskset`-pinned; affinity mask verified **inside**
  every window, and a window whose mask is not exactly `{7}` is `INVALID` and
  excluded.
- Workload and statistic unchanged: 100 000 iterations, 4 runs per window, spread =
  drop first value, drop > 1e9 µs timer artifacts, max − min.

## 2. Per-run instrumentation (fixed)

For **each individual run**, read the sibling's `/proc/stat` counters immediately
before and after that run:

```
sibling_busy_ticks_during_run = (total_delta − idle_delta)
sibling_util_during_run      = sibling_busy_ticks_during_run × tick_us / duration_us
tick_us                      = 1e6 / (getconf CLK_TCK)
```

Record per run: index, `duration_us`, `sibling_busy_ticks`, derived
`sibling_util_during_run`. The **first run of each window is warmup and is dropped**
from the per-run analysis (12 × 3 = up to 36 analysed runs).

## 3. Who occupies the sibling (fixed)

Per window, snapshot `ps -eo pid,pcpu,comm` sorted by CPU and keep the top 3 processes
whose `Cpus_allowed_list` **includes cpu 23**. Classify each by cgroup path:
**`OUR-LAB`** if the path contains `gpu-sim`, else **`EXTERNAL`**.

**This is the decisive datum of the bead.** If the occupier is consistently our own
lab, the interference is **self-inflicted and operational**; if external, the
environmental branch **closes**.

## 4. Decision rule (fixed before data)

Per-run analysis (up to 36 pairs):

- `ASSOCIATION-SUPPORTED` iff **ρ ≥ 0.5** with **≥ 4 runs below 0.25** and **≥ 4
  runs above 0.45** sibling_util.
- `ASSOCIATION-NOT-SUPPORTED` iff ρ < 0.5 **with those counts met**.
- `RANGE-INSUFFICIENT` if the counts are not met — **no ρ claim is made**.
- `PASS-INVALID` if any window's mask is wrong (all windows must be valid).

Also reported (secondary, descriptive): mean run duration in the low (<0.25) and high
(>0.45) groups, and the distribution of run durations. **Run duration is not the
admission quantity** — the requirement applies to the window spread — so no run
meeting 36.2 µs may be reported as an admission signal.

## 5. Outcomes, fixed in advance

- **`ASSOCIATION-SUPPORTED`**: individual runs are slowed by same-run sibling
  activity. The mechanism is SMT contention at run scale, and the direction's timing
  finding becomes **mechanistic**: the 36.2 µs criterion needs an **exclusive
  physical core**, which 16 cores shared with five labs does not provide.
  `m3d.10`'s `NO-CLOSE` then stands with a named mechanism rather than an ambient
  excuse.
- **`ASSOCIATION-NOT-SUPPORTED`**: run durations are not attributable to measured
  sibling activity at 10 ms resolution. Then either the events are sub-10 ms — in
  which case **this instrument cannot test the hypothesis** and that is the honest
  conclusion — or the cause is not the sibling (frequency/thermal/cache). State which
  the evidence supports.
- **`RANGE-INSUFFICIENT`**: report the counts and stop; no ρ claim.
- **If the top occupants are consistently `OUR-LAB`**: add a section stating that the
  interference is **self-inflicted** and what quiescing would require. That changes
  the interpretation from environmental to operational and must be said plainly.

## 6. Constants (not re-derived)

Requirement `36.2 µs` from `experiments/gpu-sim-m3d.10/verdict_out.json`,
`arithmetic.max_tolerable_spread_us`. Thresholds 0.25 / 0.45 and ρ ≥ 0.5 fixed
before data.

## 7. Reporting obligation

`y1h` (`RANGE-INSUFFICIENT`), `52i` (`NO-BOTH-QUIET-CORE-IN-20-SAMPLES`), `99g`
(`NO-ADMISSION-CORRECTED-INSTRUMENT`) and `3ba` (`NO-ADMISSION-QUIET-CORE`) are
quoted as **unchanged** in this bead's results, beside this bead's verdict. No
admission is claimed here: this bead is correlational.
