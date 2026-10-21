# gpu-sim-99g results — corrected instrument

**Pre-registered verdict: `NO-ADMISSION-CORRECTED-INSTRUMENT`** (1 of 3 windows
qualified) — and the reason is a **third** instance of the same mistake, now
identified exactly: the unit of "quiet" was a **logical CPU**, and the core it
belongs to was half-saturated by its **SMT sibling**.

Executed by the PI; the reviewer reviews it independently. Pre-registration fixed
before the pass. `gpu-sim-3ba`'s verdict (`NO-ADMISSION-QUIET-CORE`) is quoted here
as **unchanged** and is not reinterpreted.

## 1. The measurement

First sample found cpu 8 at 0.2143 (below 0.25), so the three windows ran
immediately on it, as pre-registered.

| window | mask pinned | other_util (corrected) | our own CPU | spread | ≤ 36.2 µs? | run durations (µs) |
|---|---|---|---|---|---|---|
| 1 | ✅ | 0.1817 | 23.6 ms | **35.1 µs** | **YES** | 5858.8, 5896.2, 5892.8, 5861.1 |
| 2 | ✅ | 0.1739 | 15.2 ms | 1878.9 µs | no | 3304.3, 3296.1, 3308.2, 5175.1 |
| 3 | ✅ | 0.1027 | 19.4 ms | 388.9 µs | no | 4241.2, 4800.0, 5147.6, 5188.9 |

All three windows were **genuinely quiet with respect to other work on that
logical CPU** (0.10–0.18, i.e. below the 0.25 threshold — the corrected instrument
worked as designed). One window qualified on spread; the rule needs two.

## 2. Why one window qualified and two did not — the SMT sibling

`/sys/devices/system/cpu/cpu8/topology/thread_siblings_list` = **8,24**. The pinned
logical CPU 8 shares a physical core with logical CPU **24**, and during this pass:

- cpu 8 (ours) measured **0.2143** — "quiet";
- cpu 24 (its sibling) measured **0.9421** — essentially saturated, and still
  **1.0** when re-checked minutes later.

So the "quiet core" was **half of a busy physical core**. That explains both
anomalies in the table without any further hypothesis:

1. **Bimodal spread**: when the sibling leaves execution units and cache free, the
   measurement is tight (window 1: 35.1 µs); when it does not, runs stretch by
   hundreds of microseconds (windows 2–3). SMT contention is bursty at exactly this
   timescale.
2. **The duration shift**: the same 100 000-iteration workload ran at ~5.9 ms in
   window 1 but ~3.3–5.2 ms in windows 2–3. Identical code taking half the time is
   not noise — it is the sibling stealing (or not stealing) issue slots.

The topology was there to read all along; the instrument measured a logical CPU
because that is what `taskset -c` pins. **That is the third time in this direction
that the instrument measured the wrong resource** — `ep3` measured the box, `s9j`
the user slice, `3ba`/`99g` a logical CPU. Each time the verdict was honest and the
framing was wrong.

## 3. What this establishes

- **`3ba`'s two sub-requirement windows (15.2, 25.3 µs) are not yet evidence of an
  achievable operating point.** Combined with `99g`'s window 1 they show the
  criterion is **reachable ~3 times in 6 windows**, i.e. bimodally: sometimes well
  inside the requirement, sometimes far outside.
- Per `99g`'s pre-registration §5, that is what must be said plainly: the corrected
  instrument **did not admit**, and the honest reading is *bimodal spread with an
  identified mechanism*, not *a proven admissible operating point*.
- **`m3d.10`'s `NO-CLOSE` stands untouched.** No gate run has occurred.

## 4. What the successor must do (filed as the next bead)

Measure the resource that actually determines the timing: the **physical core**,
i.e. both SMT siblings. Selection requires **both** siblings quiet; the corrected
instrument is extended to sum other-work over both siblings and subtract only our
own scope's CPU. Prediction, stated in advance: **if the bimodality is SMT
contention, then windows on a both-siblings-quiet physical core will be
consistently inside 36.2 µs.**

Feasibility, measured while writing this: this box has **16 physical cores**; in a
3 s sample **0 of 16** had both siblings below 0.25, but the best was **cpu(7,23) at
0.252 / 0.215** — at the boundary. So a both-siblings-quiet core may be rare but is
plausibly findable, and the multi-sample hunt protocol is the way to find it.

## 5. Reproduce

```bash
cd <repo root>
systemd-run --user --scope --property=CPUQuota=800% -- \
  python3 experiments/gpu-sim-99g/run.py
cat /sys/devices/system/cpu/cpu8/topology/thread_siblings_list   # -> 8,24
python3 experiments/gpu-sim-99g/check_statistic_agreement.py     # AGREEMENT OK
```
