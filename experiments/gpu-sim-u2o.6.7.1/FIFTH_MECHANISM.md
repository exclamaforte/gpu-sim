# The fifth mechanism is unidentified, and that is the finding
## The dataset was never expensive to regenerate, and the decision


## The decision, in one place

**A fifth mechanism is UNIDENTIFIED.** Recorded as a named unknown, with the
discriminating measurement named alongside it, and a test that fails if a sixth
candidate is ever invented to fill the gap.

| hypothesis | verdict | arm |
|---|---|---|
| warm state between examples | REFUTED | `cold` — 205/264 at 11.9% |
| the tolerance rule | REFUTED | three like-for-like rules — 12.6% under all |
| process isolation | REFUTED | per-invocation process — 210/264 at 12.5% |
| the executable / memory environment | **A QUARTER OF IT** | 7.1's own per-kernel binaries — 12.6% → 9.4% |
| a shared CUDA context | reduced, not eliminated | same correction tripled the speed |
| **fifth mechanism** | **UNIDENTIFIED** | — |

### The dataset was never expensive to regenerate

**7.1's slowness was a BUILD cost — a compile and a process per kernel — not a
DEVICE cost.** 264 rows took **110.52 s** at 2.39 ex/s and take **0.52 s** at
510.80 ex/s: **214×**, with the kernel itself at ~2.9 µs per example, about
**3%** of 7.1's wall clock. The constraint that shaped hours of work was never
the GPU and never the corpus.

That changes the economics of everything still open: a discriminating
re-measurement now costs about **half a second**, so the price of waiting for
the fifth mechanism to be named is no longer a dataset rebuild. The HELD
decision was right on its own terms; the stakes were never as high as the
arithmetic made them look.

### Is 9.4% small enough to proceed? Split by claim type

Against the dataset's **own measured dispersion** — 7.1's per-row
across-invocation CV of **0.0854** on its 264 sub-10 µs rows, and 6.6.1's
priced MDD at N=20 of **0.074** — the residual sits at **1.10×** and **1.27×**
respectively. **Just outside, not comfortably inside.** So:

- **Proceed** for relative and correlational claims, where a systematic offset
  largely cancels or is absorbed: 8.3's 0.462–0.53 Spearman, 9.4's roofline
  ratios, 7.7's split and cardinality findings, anything about which families
  exist or what a representation collapses.
- **Do not proceed** for absolute timings and latencies.

I stored one ratio against the wrong denominator on the first pass — 1.26 is the
6.6.1 quotient, filed under a key derived from 7.1's figure — and the test that
recomputes the ratio caught it. Both are now stated.

### The discriminating measurement, named and costed

The design is a 2×2 of {executable} × {process model}, and the confound test
filled three corners. The missing one:

> **7.1's own per-kernel binaries, under 7.1's own per-invocation process
> model.**

Same code as the best arm, 7.1's process model. If it reproduces 7.1's medians,
the fifth mechanism is the shared context and it **gets a name**. If it does not,
the difference lives in `run_kernel()` or the preexec limits and the context is
exonerated. Either way the fifth hypothesis stops being a guess.

Cost: one GPU job of about 70 seconds, plus a few seconds of build. Not run —
twenty jobs tonight went to this harness and the session is very long. A named
next measurement is worth more at this hour than a twenty-first scaffolding
failure.
