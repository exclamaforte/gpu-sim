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


---

## The dataset's disposition: **HELD**

Stated here, in my words, because a decision that lives only in a comment is a
decision nobody made. It is also mirrored machine-readably in
`dataset_disposition.json` and **asserted by
`tests/test_dataset_disposition.py`**, so promoting HELD to RETAINED later
without satisfying a condition fails a test rather than requiring someone to
re-read a paragraph. Naming a failure mode does not immunise you from it; a
check that lives in a careful sentence is not a check.

**HELD** means: keep the 264 rows exactly as committed, build on them only the
claims their precision supports, and do not re-measure them until a named
condition is met. It is not "retained as authoritative" and it is not
"discarded".

**Why not RETAINED.** Retaining them as authoritative means accepting that each
label was measured in a process that **paid a cold start**. The batched path pays
none and a real caller pays none. A dataset whose per-example measurement
includes paying for something a caller never pays is measuring the measurement
apparatus as much as the kernel. That is a **fidelity** problem, and it does not
go away by declaring the numbers reproducible.

**Why not REGENERATED.** Regenerating now is premature in both directions: with
the batched path I bake in the confound I introduced (my driver is *not* 7.1's
per-kernel binary), and with the single-run path I reproduce the rows I already
have. Neither buys anything until the binary question is settled, and
regenerating twice is worse than regenerating once.

### The number that decided it

The offset between the two paths is **−0.30 µs** (median signed difference;
70.5% of rows faster under batching). The 264 rows' **own** per-invocation
spread is a median of **0.408 µs**.

**The difference between the protocols is smaller than the dataset's own
measured dispersion.** So the dataset cannot arbitrate its own authority — it is
not precise enough to distinguish the two paths, and no amount of re-reading
these rows will settle which is right. That is why the answer is HELD rather
than either extreme.

### What may not be built on it meanwhile

**No absolute timing claim.** The rows' medians are sound for *relative*
structure — which families exist, what the split looks like, what the feature
vector collapses — and unusable for "this kernel takes X microseconds". 8.3's
0.462–0.53 Spearman and 9.4's roofline ratios are relative and survive; an
absolute throughput or latency claim does not.

### Two sentences that belong on this rule rather than in a comment

**A REDUCED VARIANCE IS NOT EVIDENCE OF INCREASED FIDELITY.** Those are different
properties, and confusing them is how "tighter" became evidence *for* my
hypothesis instead of against it. Faster and tighter together are the signature
of a *different quantity* — a warm process does not pay the cold-start path —
and a cleaner number is not the same number.

**THE SIGN CARRIES THE SIGNAL AND THE COUNT CARRIES THE NOISE.** The reproduction
count swung 161→218 of 264 on identical configurations and 50%→83% across
tolerance rules: fragile, and incapable of settling anything. The signed median
difference is −0.30 µs with 70.5% of rows faster, and is 12.6% under all three
rules: robust to the rule while the count is not. Noise gives symmetry; a
systematic offset gives a direction.
