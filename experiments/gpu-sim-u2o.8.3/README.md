# gpu-sim-u2o.8.3 — numeric feature vector, populated from real measurements

The audit note said the feature-vector code "has never been populated from any
real measured example (no such example exists)". **7.1 has since produced 264**
(ts `20260928t074004-b429f469`). So this populates all 264 — and the more useful
result is what the vector could not see *before* it was populated.

No GPU, no ts job. The `.cu` sources are build artifacts and uncommitted, so
each row's source is **regenerated and then verified by hash**: 7.1's
`kernel_hash` is `sha256(source + json([grid, block, n]))[:16]`, so a
regenerated spec hashing to a row's recorded value is provably that row's
source. **264/264 verified; zero rows matched by name or position.** A row whose
hash cannot be recovered aborts the run rather than being skipped quietly.

## First finding: the module did not import

`epic_u2o/features.py` had `from tokenizer import tokenize, TokenType` instead
of `epic_u2o.tokenizer`. So the file could never have been run by anything, and
the audit note — "code exists but has never been populated" — *understated* how
dead the code was. It was not unpopulated; it was unrunnable.

## Second: the vector could not see either determinant of runtime

| | distinct feature vectors, of 264 rows |
|---|---|
| as written (no launch config, no problem size) | **5** |
| with the measured launch and problem size | **165** |

Two independent causes:

1. **`extract_launch_config()` returns a 1-block default.** A `__global__`
   kernel *definition* contains no `<<< >>>` launch site at all — the launch is
   a host-side decision — so this path is taken for **every** real kernel, and
   it returned a plausible-looking `grid=1, block=1` with no signal. 13 of the
   264 rows used distinct grids; the vector described all of them as 1 block.
   `LaunchConfig` now carries `is_default_heuristic` so a caller can tell a
   measured launch from a guessed one.

2. **`KernelFeatures` had no `problem_size` field at all.** 27 fields, none of
   them the size of the problem. The vector was structurally incapable of
   distinguishing two launches of the same kernel at different `n`.

Also fixed: `vector_dim` returned **26** while `to_vector()` produced **27**
floats, so anything sized from `vector_dim` silently dropped a feature. It is
now derived from the vector.

## The honest part: populated ≠ predictive

Asked whether the vector actually tracks the label, rather than assuming it:

| feature | Spearman vs measured median |
|---|---|
| `estimated_flops` | 0.530 |
| `total_threads` | 0.520 |
| `problem_size` | 0.462 |
| `estimated_bytes_moved` | 0.462 |

Those correlations are weak, and the reason is the **dataset**, not the vector:
every one of the 264 rows is `launch_bound`, and the median runtime moves only
**2.97 µs → 3.77 µs across a 64× size range** — a total label range of 0.808 µs.
A vector that sees `n` cannot be shown to be predictive when the label it is
supposed to explain barely moves. That is 7.1's own finding, restated.

So: the vector is now correctly populated and demonstrably sees the right
things, and **it cannot be judged on this data**. Judging it needs 7.1's dataset
extended past 2^18 into memory-bound territory — which is also what unblocks
9.4, 9.5, 9.2 and 8.4, all of which consume this vector.

13 tests, 0 failures, CPU only, plus 13 more for the cardinality gate below.

## The cardinality gate: asserting what the number IMPLIES

I reported `distinct_vectors: 5` for 264 rows — correctly named, correctly
computed, correctly reported — and **nothing asked what 5-over-264 means.**
That is the same shape as 6.6.3's `discriminates` boolean, 6.6.4's band table
disagreeing with its own headline, and 7.7's 264-to-8 tokenizer collapse: a
derived value that is true, named, reported, and meaningless at the magnitude it
is reported.

`epic_u2o/representation_cardinality.py` closes it:

| representation | distinct / rows | ratio | verdict |
|---|---|---|---|
| feature vector, as written | 5 / 264 | 1.9% | **COLLAPSED** |
| token digest (7.7) | 8 / 264 | 3.0% | COLLAPSED |
| feature vector, corrected | 165 / 264 | 62.5% | **PARTIAL** (passes) |

The **ceiling is measured, not assumed**: the 264
rows are all distinct `(op_type, n, grid, block)` configurations, so a
representation that separated every one of them would reach 264. The floor
(25%) is a *decision*, recorded with its reasoning, sitting in a wide empty gap
between the 1.9% collapse and the 62.5% working value — not a constant fitted to
make the code pass. It is a floor, not a target: separating more is never
penalised, and lowering it is allowed but must be deliberate.

The as-written figure is **reported** rather than asserted, deliberately:
refusing to emit the artifact that documents the defect would hide the defect.
The corrected one is **asserted**, and a future collapse is a non-zero exit.

### The residual collision, and why it is the family axis

Even corrected, 165 of 264 — and the cause is not noise:

> 66 colliding groups covering 165 rows; 66 of them span MORE THAN ONE op_type, 0 are within a single family. The residual collision is therefore the FAMILY axis: a purely static feature vector has no op-identity field, so add, mul and div at the same size and launch are the same vector. That is a property of the channel, not a bug, and it is why the numeric vector cannot substitute for the token channel's family signal -- 7.7 needs family holdout precisely because of it.

A purely static feature vector has no op-identity field, so `add`, `mul` and
`div` at the same size and launch are the *same vector*. That is a property of
the channel rather than a bug, and it is the reason 7.7 needs family holdout:
**the numeric vector cannot substitute for the token channel's family signal.**
The two channels are complementary, and a ratio without a cause would have been
the same omission one level up.

## Relationship to 7.7

7.7 found the **tokenizer** collapses 264 rows to 8 distinct inputs, invariant to
size and launch config. This is the same blindness in the **numeric** channel,
and worse: 5, not 8. Two independent representations of the same dataset, both
unable to see the two things that determine runtime, and neither could have been
noticed by reading the code — only by running them against real measured rows.
