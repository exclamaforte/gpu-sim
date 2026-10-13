# The confound test: partially confirmed, and the speed survives the fix

The PI named one measurement this lab had grounds to want. This is it, and the
answer is neither the confirmation nor the refutation I expected.

| arm | ex/s | vs 7.1 | reproduces | median rel diff | % faster | ai_cv |
|---|---|---|---|---|---|---|
| single-run 7.1 (reference) | 2.39 | — | — | — | — | 0.1120 |
| batched warm (**my** driver, 1 ctx) | 167.30 | 70.0× | 206/264 | 12.6% | 70.5% | 0.0344 |
| per-invocation process (**my** driver) | 3.78 | 1.6× | 210/264 | 12.5% | 84.5% | 0.0724 |
| **PER-KERNEL binaries, 1 ctx (7.1's own)** | **510.80** | **213.8×** | **214/264** | **9.4%** | 72.7% | 0.0307 |

## Partially confirmed: the executable is a contributor, not the cause

Swapping my single dispatch binary for **7.1's own per-kernel binaries** — one
translation unit per distinct kernel, each its own `nvcc -c`, each with its own
`cudaMalloc` calls sized to that kernel's arity — moves the median relative
difference from **12.6% to 9.4%** and reproduction from 206 to 214 of 264.

So roughly **a quarter of the offset was the executable and its memory
environment.** The PI's mechanism was specific and sufficient for that part.

But **9.4% does not collapse to zero** and 50 rows still fail, so something the
executable does not explain remains. Five hypotheses are now eliminated or
reduced, and there is a fifth I cannot name:

| hypothesis | verdict |
|---|---|
| shared CUDA context | reduced (part of the residual) |
| warm state between examples | refuted by the `cold` arm |
| the tolerance rule | refuted (12.6% under all three rules) |
| process isolation | refuted by the decisive arm |
| **the executable / memory environment** | **a quarter of it** |
| **unexplained residual** | **~9.4%** |

## And the surviving speed is NOT 1.6×, which corrects the PI's framing

The ruling was that the 70× and the 12.6% are one fact — the speed comes from
exactly the change that breaks comparability. The per-invocation-process arm
supports the shape of that: isolation alone is worth only **1.6×**.

But **7.1's own binaries in ONE shared context run at 510.80 ex/s = 214×**, not
1.6×. The speed does *not* require per-invocation processes; it requires a
shared context — and it **triples** when the executable confound is fixed.

So "the speed IS the offset" is **too strong**. The offset falls by a quarter
*and* the speed rises threefold when the executable is corrected. They are
related, not identical.

## A result I am recording precisely because I nearly reported it

The first working run of this arm reported **2/264 reproducing, 87.3% median, and
100% of rows "faster."**

That was uniform to the point of being a measurement of **nothing**. The hoisting
rewrote `const int N = 4096;` as `N = 0;` — and that line sits *inside* the
kernel's own `main`, which runs **after** the driver has set `N`. So every spec
reset `N` to 0, allocated zero bytes, and launched a kernel that computed
nothing.

**The tell was that it was too consistent.** A real measurement of 264 different
configurations does not come out 100% one-sided. The fix was to **delete** the
declaration rather than rewrite it.

This is the `0.00 ex/s` failure mode again — a run that measures nothing and
prints a well-formed number — except that this one printed a *plausible* number
with real-looking structure, which is harder to catch than a zero.

## Status, unchanged where it matters

Criterion 2 still fails, at 9.4%. The dataset disposition stays **HELD**. Under
R1 the single-run path remains authoritative, and the residual is a fifth cause
that I cannot name from the evidence I have.
