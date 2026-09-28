# gpu-sim-u2o.6.6.4 — does the analytic roofline predict our measurements?

Pure CPU post-processing over the committed 6.6.2 anchor (124 cells). No GPU,
no ts job, no harness changes, and **neither disputed value used as an input** —
not `launch_to_memory_crossover_n`, not `measured_memory_ceiling_GBps`. The
ceiling is swept over a stated band instead.

## First: the bead's own DESIGN is wrong, and running it is how I know

The design specifies `t_roofline = min(flops/peak, bytes/ceiling)`. For a
fixed amount of work a kernel is bounded **below** by *both* resources, so the
time floor is `max(...)`. `min` always selects the FLOP branch — which for a
memory-bound kernel is orders of magnitude smaller — and under-predicts by
~100% at **every** size, including the largest.

Both forms are computed per cell, so the difference is auditable rather than
argued. Everything below uses the `max` form.

## The canonical population, disclosed and closed

Every number below is computed over **one** named population, so this file,
the band table and the generated verdict cannot disagree:

| population | cells | why |
|---|---|---|
| **canonical** | **74** | config covers the problem (threads ≥ n/4) **and** a roofline has work to bound |
| `k_empty` | 9 | the launch *measurement*; no bytes, no flops, so `t_roofline` is 0 by construction — these constrain the launch fit rather than being tested against a roofline |
| `k_noop_store` | 9 | O(1), no modelled bytes → `t_roofline` is **exactly 0**, which forces a 100% residual by arithmetic, not physics |
| under-covered | 32 | 1 block for a 2²⁶ problem: 22.9 ms against a 523 µs roofline. No roofline has a term for "your grid is 65536× too small" |
| **total** | **124** | closes against the anchor |

The 9 `k_noop_store` cells are the subtle ones: they survived my first
"well-covered" filter and each contributed a manufactured 100% residual to the
small-size bands. With them excluded the ≤4K band is 99.8% instead of 100.0% —
the *conclusion* is unchanged, but the earlier number was partly an artefact.

## The answer: the rule holds as a heuristic, and is not validated

Median residual against measured, on the canonical population:

| band | cells | `min` (as specified) | `max` (correct) | max + launch |
|---|---|---|---|---|
| <=4K | 18 | 100.0% | 99.8% | 31.5% |
| 4K-256K | 16 | 100.0% | 96.1% | 28.5% |
| 256K-16M | 26 | 99.8% | 53.6% | 42.4% |
| >16M | 14 | 99.8% | 17.5% | 26.7% |

**No band reaches 10%.** The best is **17.5%** (>16M). So the ≥10 µs rule is **not** backed by any analytic accuracy claim.

What backs it is the **sign** of the error: below 16M the model *over*-predicts
time, so reading "roofline says >10 µs" as "this will be slow" errs on the safe
side. The rule is therefore a **conservative heuristic, not a validated
predictor** — and it is not tight: the binding boundary is ~16M, not 10 µs.

## The launch term helps below 16M and HURTS above it

Adding dispatch unconditionally takes the residual from ~100% to ~28–40% below
16M, and from 17.5% to 26.7% above it. On a cell that is already work-bound
the launch term is double-counted. The right model is a regime switch, or
`min(t_launch + t_work, t_work_with_amortised_launch)` — not unconditional
addition. The launch term is fitted from the anchor's **own** empty-kernel
cells: `t_us = 1.891 + 4.622e-4 × blocks`, against block count rather than
threads (the anchor's 1-block and 16384-block empty cells differ 6×, while two
cells at equal threads differ by 3%).

## Ceiling sensitivity: the conclusion DEPENDS on it where the model works

Median residual on covered cells above 16M:

| ceiling | residual |
|---|---|
| 1538.7 GB/s (anchor argmax) | 6.5% |
| 1541.3 GB/s (disputed value) | 6.6% |
| 1790 GB/s (spec) | 19.6% |
| 1200 GB/s | 29.7% |

So **the ceiling must be settled before any parity claim at large n** — that
raises the priority of 6.6.2's ceiling fix rather than lowering it. Below 16M
the ceiling moves the aggregate by under one percentage point, because there the
model is dominated by launch cost and no bandwidth value rescues it.

## Under-covered cells are where it fails hardest, and they are not the model's fault

1 block for a 2²⁶ problem: **22.9 ms measured against a 523 µs roofline**.
No roofline has a term for "your grid is 65536× too small". The anchor keeps
those cells deliberately (6.6.2 needed that dynamic range), so they are
reported separately — averaging them into a band median makes the model look
broken for the wrong reason.

## What a learned predictor should use

* **Above ~16M, the plain `max` roofline is a usable sanity check** (6–17%
  depending on the ceiling) and is a good *baseline* to beat.
* **Below ~16M it is worthless** — 55–100% error — and any predictor must model
  launch/dispatch cost as a first-class term, not as noise.
* Never fit against the 1-block/under-covered cells expecting the roofline to
  explain them; they are the launch-cost axis, not the memory axis.
* The ≥10 µs generation rule stands, with the boundary note above.

## The script now refuses to emit a self-contradicting artifact

This bead failed review because the comment, the band table and the yes/no
field gave **three different answers**. So the writer asserts before writing:
populations sum to 124, the verdict agrees with its own band table, and the
headline quotes a residual the bands actually support. Tampering with the
verdict field makes the script exit 1 and emit nothing — verified, not assumed.

## Two errors of mine, both in the summary arithmetic

Worth recording because both produced a plausible-looking number: the GEMM's
`problem_size` is `D·D`, so `2·n³` gave a **137-second** prediction against a
measured 1.8 ms (the same two-conventions trap 6.6.2 found in bytes, here in
FLOPs); and the least-squares intercept was written `mx − slope·mx` instead of
`my − slope·mx`, which left the **slope exactly right** and the intercept
wrong by three orders of magnitude — it reported a "launch-corrected
under-predicts by 118,287%" verdict. Third time in two days that a correct
measurement carried a wrong summary, and the second time the bug hid inside a
term that was otherwise fine.

And a third, this time nearly invisible: while fixing the populations I moved
`CANONICAL` *after* the band loop that consumes it, so the script raised
`UnboundLocalError` — and I then read the **stale** `metrics.json` left by the
previous successful run, saw plausible numbers, and briefly believed the old
values still held. Nothing in the output said "stale". A crashed generator and
a stale artifact look identical from the outside, which is the whole reason the
self-check above now runs *before* the write.
