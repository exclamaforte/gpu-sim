# gpu-sim-y1h pre-registration — written 2026-10-21 before any window

Executed by the PI; reviewed independently. The implementer stands down.

## 0. Why this bead exists

`gpu-sim-52i` could not obtain a physical core with **both** SMT siblings below 0.25
in 20 samples (best-pair worst-sibling min 0.2500 — exactly at the threshold), so the
conditional prediction from `99g` — *if the bimodality is SMT sibling contention,
then windows on a both-quiet core will be consistently inside 36.2 µs* — was **never
tested**. The antecedent is unobtainable here.

The hypothesis can still be decided **correlationally**: if sibling load drives the
spread, then across windows taken at a **range** of sibling utilisations the spread
must increase with sibling utilisation. That needs no quiet pair — only variation,
which this box provides in abundance.

**This bead is correlational and cannot produce an admission.** Admission requires
the conjunctive quiet-pair rule from `99g`/`52i`, which this bead does not attempt.

## 1. Fixed target (not selected during the pass)

Pin every window to **cpu 7**, whose sibling **23** was the quieter half of the best
pair observed in `52i`. CPU identity is therefore not a confounder. If cpu 7 is
unusable, use its sibling 23 and say so once; **do not switch targets between
windows**.

## 2. Windows (fixed)

**12** windows, back-to-back in **one unattended scripted pass**. There is
deliberately **no selection step**: the point is to sample a *range* of sibling-load
conditions, not to wait for a favourable one.

Per window record: pinned CPU and sibling; the **sibling's own** busy fraction over
the 200 ms span containing the window (`sibling_util` from `/proc/stat` deltas, which
by construction excludes our own CPU); our own scope's `usage_usec` delta; spread by
the `verdict.py` statistic; the four run durations; mask validity (`INVALID` if not
pinned to the fixed target); timestamp; loadavg as context only.

## 3. Statistic (fixed)

**Spearman rank correlation** between `sibling_util` and `spread_us` across the valid
windows, with average ranks for ties, computed in-process (no external dependency).
Report ρ and the range check.

## 4. Decision rule (fixed before data)

- `ASSOCIATION-SUPPORTED` iff **ρ ≥ 0.5** AND the pass includes **≥ 3 windows with
  sibling_util < 0.25** and **≥ 3 windows with sibling_util > 0.45** (a range wide
  enough to see the effect).
- `ASSOCIATION-NOT-SUPPORTED` iff ρ < 0.5 **with** that range achieved.
- `RANGE-INSUFFICIENT` if the range is not achieved — **no ρ-based claim is made at
  all**.

Thresholds and the range bounds are fixed now and are **not** adjusted after seeing ρ.

Secondary description only: the spread distribution (min/median/max) and how many
windows met 36.2 µs. **A single window meeting the requirement is not an admission**
and must not be reported as one.

## 5. Outcomes, fixed in advance

- **`ASSOCIATION-SUPPORTED`**: the bimodality is SMT sibling contention, and the
  direction's timing finding becomes **mechanistic and transferable** — the 36.2 µs
  criterion requires an exclusively-owned **physical core**, a resource 16 cores
  shared with five research labs does not supply. `m3d.10`'s `NO-CLOSE` then stands
  with a named mechanism rather than an ambient excuse, and "is it achievable"
  becomes "yes, given a core nobody else shares — untested here because
  unobtainable".
- **`ASSOCIATION-NOT-SUPPORTED`**: sibling load does not drive the spread. The cause
  is then **intrinsic to the measurement path** (timer resolution, frequency/thermal
  state, memory/cache contention), and `3ba`'s / `99g`'s sub-requirement windows must
  be restated as **unexplained** rather than sibling-correlated. That is the harder
  finding and it must be stated plainly.
- **`RANGE-INSUFFICIENT`**: the pass did not span enough sibling-load variation;
  report the observed range and stop.

## 6. Constants (not re-derived)

Requirement: `experiments/gpu-sim-m3d.10/verdict_out.json`,
`arithmetic.max_tolerable_spread_us = 36.2`. Workload and statistic identical to
`ep3`/`3ba`/`99g`/`52i` (100 000 iterations, 4 runs; spread = drop first value, drop
> 1e9 µs timer artifacts, max − min).

## 7. Reporting obligation

`52i`'s (`NO-BOTH-QUIET-CORE-IN-20-SAMPLES`), `99g`'s
(`NO-ADMISSION-CORRECTED-INSTRUMENT`) and `3ba`'s (`NO-ADMISSION-QUIET-CORE`)
verdicts are quoted as **unchanged** in this bead's results, beside this bead's
verdict.
