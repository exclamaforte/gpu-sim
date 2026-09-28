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

## The answer: the ≥10 µs decision survives, but it is not tight

Median residual against measured, on cells whose launch config actually
**covers** the problem (`threads ≥ n/4`):

| band | `min` (as specified) | `max` (correct) | max + launch term |
|---|---|---|---|
| ≤ 4K | 100.0% | 99.8% | 31.3% |
| 4K – 256K | 100.0% | 96.8% | 27.8% |
| 256K – 16M | 99.9% | 54.5% | 39.7% |
| **> 16M** | 99.8% | **17.5%** | 26.7% |

**The PI's ≥10 µs rule is safe but loose.** The plain analytic model needs
something nearer **16M**, not 10 µs: it is ~100% wrong at 4K–256K and only
reaches 17.5% above 16M. The rule errs conservative, which is the right
direction, but anyone reading it as "the model is fine above 10 µs" would be
wrong by an order of magnitude in n.

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
