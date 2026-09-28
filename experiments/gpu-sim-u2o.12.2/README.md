# gpu-sim-u2o.12.2 — the bottleneck comparison, and the decision

The bead asks for one comparison — measured data-generation throughput against
measured training throughput — and then a **decision**: open further
training-optimisation work *only if* training actually dominates wall clock.

Both sides are **read from committed artifacts on other branches, never
retyped**, so the comparison cannot drift from its sources. No GPU, no ts job.

| side | source | ts job |
|---|---|---|
| generation | `lab/gpu-sim-u2o.7.1` measured.jsonl, 264 rows | `20260928t074004-b429f469` |
| generation (logged) | `lab/gpu-sim-u2o.6.7` metrics.json | — |
| training | `lab/gpu-sim-u2o.12.1` metrics.json, NVIDIA GeForce RTX 5090 | `20260928t094902-9a8684a2` |

My independent re-derivation of the generation rate agrees with 6.7's logged
figure to **0.00%**.

## A correction to 6.7 that this comparison depends on

6.7 computed `kernel_us_per_launch_median` as `sum(invocation_medians_ms)` —
**two** medians per row — divided by `total_launches` — **forty** per row. That
understates the per-launch kernel by exactly `n_invocations`, so:

| | 6.7 reported | correct |
|---|---|---|
| kernel per launch | 0.17 µs | **3.16 µs** |
| overhead fraction | 99.998% | **99.970%** |

The **conclusion is unchanged** — overhead still dominates by three orders of
magnitude — but the number I reported last turn was wrong by 20×, and the
ceiling derived from it is the number that actually decides things. Recorded in
`metrics.json` under `correction_to_6_7` rather than quietly amended.

## The naive comparison, and why it is not the answer

*"Generation produces 2.39 ex/s; training consumes
233–14,583 ex/s, so generation is the bottleneck."* True, and **useless for
deciding what to do** — it describes the current protocol, not the constraint.

The decision-grade question is what happens if generation is **fixed**, and that
needs the overhead-free ceiling, which is derivable from the same measurement:
40 launches per example at 3.16 µs
each is 126 µs of GPU work per example, so a
persistent worker with *zero* process overhead tops out at
**7,911 ex/s**. Labelled an **upper bound**
in the artifact, because it assumes the launches pipeline perfectly and they
will not.

## The comparison

| rung | params | training consumes ex/s | fed today? | fed after the overhead fix? | shortfall today |
|---|---|---|---|---|---|
| `small` | 581,761 | 14,583 | **STARVED** | **STARVED** | 6,105× |
| `medium` | 13,926,785 | 1,662 | **STARVED** | fed | 696× |
| `large` | 47,220,609 | 233 | **STARVED** | fed | 97× |

**Every rung starves today.** After removing process overhead, medium and large
are fed — and **`small` still starves**, because 14,583 ex/s of consumption
exceeds even the zero-overhead ceiling.

That is the non-obvious part, and it changes what to work on. The obvious fix
(6.7's: batch or persist the worker) is worth
~3,312×, and it is
necessary — but it is **not sufficient for the small model**. The residual
constraint is the **measurement protocol**: 40 launches per
example. Halving the trial count would double the ceiling — but that trades
against the trial count 6.6.1 justified, so it is a *statistical rigour*
decision, not a free win.

## The decision

**`open_training_optimization_work: false`.** The bottleneck is **data
generation, at every model size measured**. No training-side optimisation can
change the ratio, because the constraint is how fast examples are *produced*.

What would actually help:

1. batch or persist the measurement worker — worth ~3,312×
2. reduce launches per example — raises the ceiling proportionally, but costs
   statistical rigour, so it is a decision rather than a free win
3. **not** training optimisation, and **not** a faster GPU

16 tests, 0 failures, CPU only.

## Caveats, because a decision without them is the thing this lab keeps making

- **Throughput, not end-to-end wall clock** for a whole run; the two sides were
  separate jobs on a shared GPU.
- The training side consumed **synthetic** batches (`torch.randint`), not real
  dataset rows, because the model still cannot accept the real feature width
  (12.1 finding 3). Its rate is a property of the model and batch shape, not of
  the data.
- The ceiling is an **upper bound**.
- *"The v1 model size" is undefined* — 12.1 profiled a ladder. A decision about
  "the v1 model" is therefore a decision about a ladder, and should be restated
  once a size is chosen.
