# gpu-sim-y1h results — SMT correlational test

**Pre-registered verdict: `RANGE-INSUFFICIENT`.** No ρ-based claim is made, per
pre-registration §4. The descriptive data, however, is unusually clear about what
the *instrument* can and cannot see — and that is the finding.

Executed by the PI; reviewed independently. `52i`'s, `99g`'s and `3ba`'s verdicts are
unchanged and are not reinterpreted here.

## 1. Passes run

| pass | pinning | windows valid | outcome |
|---|---|---|---|
| 1 | **none** (my bug: `run.py` measured in-process without `taskset`) | 0 of 12 | `PASS-INVALID` — discarded, preserved in `invalid_pass_unpinned/` |
| 2 | `taskset -c 7` | **12 of 12** | `RANGE-INSUFFICIENT` |

Pass 1 is recorded rather than deleted: the mask check caught it (every window
reported `mask_valid_pinned: false`), which is exactly what that check exists for. The
run was re-done with the target pinned to the fixed CPU 7.

## 2. Pass 2 — the data

12 back-to-back windows, cpu 7 pinned (affinity verified `{7}` in all 12), sibling
**cpu 23**; sibling utilisation measured over the 200 ms span containing each window.

| sibling_util | spread (µs) | run durations (µs) |
|---|---|---|
| 0.1498 | 2063.8 | 3322, 3282, 3605, 5346 |
| 0.1996 | 39.8 | 3260, 3249, 3280, 3289 |
| 0.2495 | 423.8 | 3291, 3304, 3728, 3417 |
| 0.2496 | 986.4 | 4403, 3444, 3212, 4198 |
| 0.2497 | 79.2 | 3428, 3265, 3277, 3344 |
| 0.2497 | 757.7 | 3372, 4357, 3599, 3712 |
| 0.2995 | 67.5 | 3327, 3352, 3346, 3284 |
| 0.2996 | 32.3 | 3996, 3363, 3386, 3395 |
| 0.3495 | 1989.1 | 5245, 5526, 3537, 4176 |
| 0.3994 | 701.4 | 3368, 4579, 5280, 5278 |
| 0.4493 | 167.7 | 3141, 3294, 3272, 3439 |
| 0.4493 | 1316.0 | 3367, 3239, 3392, 4555 |

Range achieved: sibling_util **0.1498 – 0.4493**; **6** windows below 0.25, **0**
above 0.45. Spearman ρ = **−0.0035** (computed, **not** verdict-bearing: the range
leg of the rule failed).

## 3. What the data does say — descriptive, not a verdict

The most telling feature is **within-band**: four windows measured essentially the
same sibling utilisation (0.2495, 0.2496, 0.2497, 0.2497) and produced spreads of
**423.8, 986.4, 79.2 and 757.7 µs** — a **12×** range at identical measured sibling
load. Whatever makes a window blow up is therefore **not visible in a 200 ms average
of the sibling's utilisation**.

That points at resolution, not at the hypothesis: the events that inflate a window
are visible in the *durations of single runs* (~6 ms each) while the instrument
averages over 200 ms and `/proc/stat` ticks at 10 ms. A 200 ms average cannot resolve
a ~6 ms interference event, so this bead could not test the SMT hypothesis even if
the range had been achieved.

## 4. What this bead settles

- **Not** the SMT hypothesis: it remains **untested** (`52i` could not obtain a quiet
  pair; `y1h` could not achieve the sibling-load range, and its instrument is too
  coarse besides).
- **Yes**, something concrete: **sibling utilisation averaged over 200 ms does not
  predict which windows are bad**, and the variation is 12× *within* a single
  200 ms band. Any test of the SMT hypothesis must be at the **per-run** scale.
- `m3d.10`'s `NO-CLOSE` stands. No gate run has occurred. No admission is claimed
  anywhere in this bead, and no single window meeting 36.2 µs would justify one.

## 5. Successor (filed)

Per-run correlational test: instrument the sibling at the granularity of the thing
being measured — sample `/proc/stat` at ~10 ms during each window, attribute the
sibling's busy time to **each of the four runs**, and correlate run duration with
same-run sibling activity (12 windows × 4 runs ≈ 48 pairs). Also identify **who**
occupies the sibling (`ps` with affinity), because if it is our own lab's sessions,
the interference is self-inflicted and testable; if it is external, that closes the
environmental branch.

## 6. Reproduce

```bash
cd <repo root>
systemd-run --user --scope --property=CPUQuota=800% -- \
  taskset -c 7 python3 experiments/gpu-sim-y1h/run.py
python3 experiments/gpu-sim-y1h/check_statistic_agreement.py   # AGREEMENT OK
```
