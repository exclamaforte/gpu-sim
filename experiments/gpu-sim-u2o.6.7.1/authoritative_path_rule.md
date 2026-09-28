# Which path is authoritative? A rule, written before the answer is convenient.

`gpu-sim-u2o.6.7.1`'s criterion 2 fails: the batched worker's per-example medians
do not reproduce `gpu-sim-u2o.7.1`'s. This file states the rule for deciding
which path is authoritative **now**, while the answer is still unknown, so the
rule cannot be reverse-engineered from whichever number turns out to be
preferable.

## Why this needs a rule rather than a measurement

**The 264-row dataset came from the single-run path.** `ts
20260928t074004-b429f469` is the single-run measurement, and it is the input to
`8.3`'s feature vector, `9.4`'s roofline baseline, `7.7`'s split analysis, and
`9.5`'s acceptance. So the consequence is not confined to the new run:

- if **single-run is authoritative**, the batching work is wrong and the
  dataset is fine;
- if **the batched path is authoritative**, then **everything derived from those
  264 rows is wrong** and must be re-measured, because ~15-25% is a different
  measurement, not a rounding;
- and a third possibility that should not be dismissed: **both paths could be
  wrong in the same direction**, in which case the 264 rows misstate runtime
  *systematically* and every absolute timing in the project inherits it.

## The rule

**R1 — The path that preserves the per-invocation structure owes the argument.**
An estimator is defined by the structure that produces it, not by the number it
emits. A path that keeps the estimator's structure intact (same invocations,
same trial count, same warmup, same estimator) is the default authority. A path
that *alters* the structure must demonstrate equivalence before it may displace
the original.

**R2 — Process boundaries are part of the structure, not an implementation
detail.** The batched worker preserves the *shape* — 2 invocations × 20 timed
trials, 40 launches, median-of-invocation-medians — and changes the *isolation*:
7.1 ran each invocation in an independent process, and the batched worker runs
both in one. Whether that is a change to the estimator or to the environment is
exactly the open question, so under R1 the batched path **owes the argument**.

**R3 — No path is authoritative on throughput.** A 70x speedup is not evidence
about correctness. The two claims are separate, and the faster one does not
confer anything on the slower one.

**R4 — The comparison must carry a noise floor measured under the same
protocol.** The current tolerance uses 7.1's per-row `ai_cv`, which was measured
under the *single-run* protocol. Using a noise floor from one protocol to judge
another is the error this whole episode is about. Until a floor is measured per
protocol, the reproduction fraction is indicative, not decisive.

**R5 — If the cause cannot be identified, nothing is re-measured and nothing is
declared wrong.** A third hypothesis beats a confident wrong attribution, and
so does no attribution. The discriminator already refused to attribute, and that
refusal is the correct state, not a gap to be closed by picking a winner.

**R6 — A change of authority is a versioned event, never a silent swap.** The
dataset carries a version; if the authority changes, the version changes, every
artifact derived from the old one says so, and the 168.81 ex/s and the 0.112 ms
type of result — a throughput or a cost claim — are not affected either way.

## A correction to the inclination I offered, because it matters for the rule

My inclination was that "the three-invocation median from 6.6.1 is the minimum
defensible estimator, so a path that alters the per-invocation structure owes
the argument." The **shape** of that is right, and the rule above is derived
from it — but the detail does not transfer, and the distinction changes what has
to be re-measured:

| artifact | invocations × trials | isolation |
|---|---|---|
| `6.6.2` anchor | **3 × 100** = 300 timed | independent **processes** |
| `7.1` dataset (the 264 rows) | **2 × 20** = 40 timed | independent **processes** |
| `6.7.1` batched worker | 2 × 20 = 40 timed | one shared process |

So the "three-invocation median" applies to **`6.6.2`'s 124-cell anchor**, not
to the 264-row dataset. The 264 rows are a 2 × 20 estimator. Both the anchor and
the dataset used **independent processes**, so the shared-context change is the
same structural change to both — but the *estimator* they depend on is not the
same one, and a rule that quietly carried "3 invocations" across both would
have protected the wrong artifact.

## What would settle it

1. Measure a noise floor per protocol (R4) — currently the largest source of
   ambiguity, and the metric's own run-to-run spread is 161–218 of 264.
2. Run the batched path with **independent processes per invocation** while
   keeping everything else identical. That is the single experiment that
   separates "the estimator changed" from "the environment changed", and it is
   cheap.
3. Only then apply R1 to decide authority, and only then re-measure anything.

Until (1) and (2) are done, the honest status is: **criterion 2 fails, the
cause is unidentified, and the dataset's authority is undetermined.** Nothing
downstream should be re-measured or defended on the assumption that it will
survive.
