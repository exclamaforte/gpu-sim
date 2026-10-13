# Which cause moved the medians? **The experiment cannot say.**

This is the run I said was needed: vary one thing at a time, protocol identical,
and see which isolation restores 7.1's medians. It ran 6 GPU jobs (2 repeats ×
3 modes). **Its answer is that the question is not answerable at this sample
size**, and it says so rather than picking a winner.

| mode | rows reproducing (each repeat) | mean | within-mode spread | median rel diff | what it tests |
|---|---|---|---|---|---|
| `warm` | [210, 161] | 70.3% | 18.6 pts | 16.1% | baseline: state carries between examples |
| `cold` | [202, 205] | 77.1% | 1.1 pts | 12.2% | removing carryover between EXAMPLES (tests H2) |
| `reset` | [218, 186] | 76.5% | 12.1 pts | 13.8% | removing the shared context between INVOCATIONS (tests H1) |

**within-mode spread 0.186 vs best-minus-warm gap
+0.068 — the effect is smaller than the noise.**

Across every run of the *same* configuration the count moved over
**161–218 of 264**. So `warm` scoring 210 once and 161 another
time is the metric's own variance, not a property of batching.

## The most important finding is about my own first version of this script

Its original verdict logic took the **argmax** reproduction fraction across
modes. With one run per mode that selected `reset` and printed:

> VERDICT: H1 -- SHARED CUDA CONTEXT BETWEEN INVOCATIONS

That verdict was **noise with a mechanism attached**. `reset` had scored 178 and
`warm` 173 in that particular pair of runs — a 5-row difference, when the same
configurations swing by 40+ rows on their own. I would have reported a confident
cause, with a plausible mechanism, and it would have been wrong.

The script now computes the within-mode spread and **refuses to attribute**
unless the effect exceeds it. That is the same lesson as 6.6.1's per-row noise
floors, applied to my own experiment instead of to the dataset: a comparison
without a noise floor is a coin with extra steps.

## What IS robust

**Criterion 2 still fails, and the failure is stable even though the count is
not.** Every one of the six runs — warm, cold and reset — lands between
161 and 218 of 264, a mean around
75%, against a 90% threshold. So whatever the
cause, the batched worker's medians do **not** reproduce 7.1's, and that
conclusion does not depend on which run you look at. Only the *magnitude* is
noisy.

## Two crashes and a bug, both instructive

- `reset` first died with *"an illegal memory access was encountered"*. After
  `cudaDeviceReset()` the buffers are freed and never reallocated, so invocation
  1 launched through dangling pointers. I had set `cap = 0` and expected the
  per-example allocation path to notice — but that path is **outside** the
  invocation loop. Fixed by reallocating inside the reset branch.
- The discriminator read `summary_{mode}.json`, a file the worker never
  writes; it writes a fixed `batch_summary.json`. Harmless (the throughput
  column was empty) but it is a derived value silently missing rather than
  loudly wrong.

## What a powered version needs

Not more mechanisms — **more repeats, or a lower-variance metric**. Counting
rows inside a tolerance is a binomial with huge variance at n=264. Better:

1. compare the **paired per-row difference** distribution rather than a
   per-row pass/fail count;
2. run each mode enough times that the between-mode gap clears the
   within-mode spread — at 40+ rows of swing, two repeats is nowhere near
   enough;
3. or fix the metric at its source: the per-row tolerance uses 7.1's own
   `ai_cv`, which was measured in a *different process model* and so may not
   bound this run's variance at all.

The honest status: **criterion 2 fails, the cause is unidentified, and the
discriminator that was supposed to find it is underpowered.** A third
hypothesis beats a confident wrong attribution, and so does no attribution.
