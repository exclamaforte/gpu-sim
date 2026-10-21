# gpu-sim-3ba pre-registration — consolidated 2026-10-21, before run 2

Executed by the PI (the implementer session wedged mid-turn three times).
The reviewer reviews the result independently.

## 0. Provenance of this document — read this first

Two passes were attempted. The record is:

- **Run 1** (started 2026-10-21T14:18:40Z, PI). Its pre-registration and its
  executing script were **replaced mid-run at 14:19Z by another session** (the
  restarting implementer, which had a queued wake and took the same bead). Run 1
  was **killed at sample 3** and is **excluded**: a pre-registration's only value
  is that the document is the one that was executed, and here it was not. Run 1's
  three samples are preserved verbatim at
  `/home/gabe/aborted-3ba-run1/samples_run1.jsonl` and are reported, clearly
  marked excluded, in `results.md`.
- **Run 2** (this document). The protocol below is the same protocol in every rule
  as the one run 1 executed (20 samples × 5 s, 30 s apart, threshold 0.25,
  immediate window use). This document adds: the requirement constant, the
  ep3-window rules used for the immediate-use branch, the outcome semantics, and
  this provenance note. **No threshold, sample count, spacing or decision rule
  changed between write-time and execution of run 2.**

## 1. Why

`gpu-sim-ep3` returned `NO-QUIET-CORE` from a **single** 5 s sample (least-busy of
32 cores at 0.3584; 0/32 below 0.25). `ep3`'s own results.md states the limit
explicitly: one sample supports "no quiet core *at sample time*", not "no quiet
core *ever*". This bead closes that gap with a **multi-sample** pass — not with
re-sampling until something looks good.

## 2. Sampling (fixed)

- **20** samples, each of **5 s**, spaced **30 s** apart (~11 minutes).
- Each sample records: timestamp, loadavg, the full 32-core utilisation vector,
  the least-busy core and its utilisation.
- Run under the same 8-CPU quota (`CPUQuota=800%`) as `ep3`.

## 3. Threshold (fixed, unchanged from ep3)

`0.25`. Not adjusted after seeing any sample. The pass is **not extended** if
early samples look promising: 20 samples and 30 s spacing are fixed now.

## 4. Decision rule (fixed)

- Any sample with least-busy core **below 0.25** → `QUIET-CORE-OBSERVED` (sample
  index, core id, vector).
- No such sample → `NO-QUIET-CORE-IN-20-SAMPLES`, with the distribution of the 20
  per-sample minima (min / median / max) and the count below threshold.

## 5. Immediate use of a quiet core (the point of the bead)

The pass is **one script**: sample → if the least-busy core is below 0.25,
**immediately** run `ep3`'s three windows pinned to that core in the same process,
because a quiet moment will not wait for a second command. `ep3`'s rules apply
unchanged:

- mask verified **inside** the measurement (must be exactly the pinned core, else
  `INVALID`, not counted);
- the core's utilisation over a **200 ms span containing the window** (`ep3` §5 —
  `/proc/stat` ticks at 10 ms and a ~20 ms window cannot resolve finer);
- spread by the `verdict.py` statistic (drop first value, drop > 1e9 µs timer
  artifacts, max − min);
- **ADMITTED** iff ≥ 2 of 3 windows have core-utilisation < 0.25 **and**
  spread ≤ 36.2 µs, and the median spread of those clean windows is ≤ 36.2 µs;
  otherwise `NO-ADMISSION-QUIET-CORE`.

Context fields (loadavg, other-lab activity) are **non-decisional** (`ep3` §7).

## 6. What each outcome means (fixed)

- **`QUIET-CORE-OBSERVED` + `ADMITTED`**: a quiet core appears and admits.
  `gpu-sim-m3d.10`'s `NO-CLOSE` is superseded by an actual gate run at that
  operating point under its ORIGINAL protocol, with pinning and environment
  documented. No gate run inside this bead.
- **`QUIET-CORE-OBSERVED` + `NO-ADMISSION-QUIET-CORE`**: a quiet core exists, but
  even on it the spread exceeds 36.2 µs. The floor is then the timer/scheduler
  path at this granularity, not contention — the strongest form of the finding.
- **`NO-QUIET-CORE-IN-20-SAMPLES`**: `ep3` hardens from one moment to a measured
  distribution over ~11 minutes, and the timing question is unanswerable on this
  box within any protocol we can run. The question retires.

## 7. Requirement (pre-computed, not re-derived)

From `experiments/gpu-sim-m3d.10/verdict_out.json`, `arithmetic`:
`d_pred_M1M2_us = 362.3`, `max_tolerable_spread_us = 36.2`.
