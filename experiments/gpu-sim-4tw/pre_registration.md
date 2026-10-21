# gpu-sim-4tw pre-registration — written 2026-10-21 before any measurement

Executed by the PI. Reviewed independently.

## 0. The claim under test

`gpu-sim-5zv` asserts, as a closed form in the model route, that under per-piece drop
probability *p* the expected number of sends per delivered piece is **1/(1 − p)**,
i.e. the wire term is multiplied by that factor.

`gpu-sim-m3d.9`'s fabric implements loss: `fg_fab_transfer` draws `r = rng % 1e6` and
returns `FG_FAB_ELOSS` when `r < loss_ppm`, incrementing `attempt_count` on **every**
call (dropped or not) and `success_count` on every delivered transfer.

**Hypothesis:** the fabric's measured `attempts / deliveries` equals **1/(1 − p)**
within the pre-registered tolerance, for each test point.

**Falsification:** if the measured ratio departs beyond tolerance, then either the
model's counterfactual is wrong or the fabric's drop implementation does not match
the model's assumption (drop per transfer rather than per piece; a caller that does
not retry; or a retry that costs nothing). Either way the **mismatch is the finding**
and it must be reported with its mechanism, never averaged away.

## 1. Test points (fixed)

`loss_ppm` ∈ {0, 50000, 100000, 250000, 500000} → p ∈ {0, 0.05, 0.10, 0.25, 0.50}.

## 2. Workload (fixed)

**Deliveries per test point = 1000.** Chosen so the expected retransmit count at the
largest p (0.5) is ≈ 1000 ≫ 30, so the estimate is not noise-dominated. Airtight
termination: keep calling `fg_fab_transfer` until `fg_fab_success_count` reaches 1000;
the loop is bounded at 200 000 attempts, and hitting the bound is a **failure**
reported as such.

## 3. Seeds and repeats (fixed)

**3 seeds per test point: 1, 2, 3**, set via `GPUSIM_SEED` before `fg_fab_attach`
(the fabric derives `rng_state = seed ^ rank*golden`, so each seed is a distinct
stream). A fresh fabric per (p, seed). Report every seed individually plus the median
ratio per test point. **A test point fails if its median ratio is outside tolerance.**

## 4. Tolerance (fixed)

`|measured_ratio − 1/(1−p)| / (1/(1−p)) ≤ 0.05` for every test point.
**Not widened after results.**

## 5. Decision rule (fixed)

- **`MODEL-SUPPORTED`** iff all five test points' median ratios are within tolerance.
- **`MODEL-REFUTED`** otherwise, with the deviating points named and their ratios
  reported.
- **`INSTRUMENT-FAILED`** if the attempt bound is hit or the counters are unusable.

## 6. Outcomes, fixed in advance

- **`MODEL-SUPPORTED`**: the model route's loss counterfactual is validated against
  the simulator's own fault behaviour. That closes the loop `5zv` opened and makes the
  model's fault predictions usable for reasoning about specs that cannot be measured
  on this box.
- **`MODEL-REFUTED`**: name the deviating points and diagnose the mechanism (drop per
  transfer versus per piece; no retry in the caller; retry without cost). **Do not
  adjust the model to fit** — record the mismatch as the finding and file the model
  correction as a separate bead.

## 7. Instrument (named, with its own check)

`fg_fab_attempt_count()` and `fg_fab_success_count()` — the counters m3d.9 added for
its invariants. Before the sweep, a **counter sanity check** runs at `loss_ppm = 0`:
100 transfers must give `attempts = 100`, `successes = 100`; and a second check at
`loss_ppm = 1e6` must give `successes` unchanged while attempts advance by the number
of calls (every transfer dropped). If either check fails, the verdict is
`INSTRUMENT-FAILED` and no ratio is reported.

## 8. Reporting obligation

`5zv`'s verdict (`AC4 PARTIAL`, model route validated statistically, plugin side
refused on the one-device fact) is quoted as unchanged in this bead's results.
No admission claim of any kind is made here; this bead is not a timing measurement.
