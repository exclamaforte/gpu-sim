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

13 tests, 0 failures, CPU only.

## Relationship to 7.7

7.7 found the **tokenizer** collapses 264 rows to 8 distinct inputs, invariant to
size and launch config. This is the same blindness in the **numeric** channel,
and worse: 5, not 8. Two independent representations of the same dataset, both
unable to see the two things that determine runtime, and neither could have been
noticed by reading the code — only by running them against real measured rows.
