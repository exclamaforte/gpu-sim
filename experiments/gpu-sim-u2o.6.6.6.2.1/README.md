# gpu-sim-u2o.6.6.6.2.1 — the 264 existing rows are uncached by construction

Every row in `gpu-sim-u2o.6.7.1`'s `batch_rows.jsonl` was sampled by a code path
with **no** `queue_state` cache, so no row's `queue_state` came from a cached
read. This artifact states that and proves it.

## The design's clock premise was false, and saying so is the point

The design asserted the rows predate the cache and that *"the mechanism did not
exist when the rows were taken."* **Both are false.**

```
cache commit 1ede829   2026-09-28T11:49:42Z
rows sampled            2026-09-28T12:19:28Z      <- 29.8 minutes LATER
```

The cache was authored on a **divergent branch 29 minutes before** the rows were
taken. Had I asserted the design's wording it would have **passed and meant
nothing** — the worst kind of assertion. So:

- **CONTENT carries the claim.** `lab/gpu-sim-u2o.6.7.1`'s `queue_state.py`
  contains **zero** `_TERMINAL_CACHE`; master has 9. The mechanism was never on
  the branch that produced the rows.
- **CLOCK bounds WHERE, not WHEN.** What it does establish is narrower and true:
  the rows predate the cache's arrival on **master** (2026-10-03) by days, so no
  row can reflect master's cache.
- **Never ancestry.** The commits share no line, so an ancestry assertion would
  assert nothing that happens to pass.

`cache_existed_elsewhere_at_sample_time` is a **property**, not a note: it is
asserted true, so if the ordering ever changed the artifact would flag it rather
than quietly becoming a different claim.

## Neither half is sufficient alone

- CONTENT alone could be defeated by an **uncommitted** cache in a working tree.
- CLOCK alone could be defeated by a commit date that disagrees with when the code
  actually ran — and here it *does* disagree.

## It fails if the rows are ever cached

Everything is re-derived at run time from the branch and the clock. Nothing is
read from a stored constant. The suite includes a fixture that feeds the detector
a producing branch which *does* have a cache, so the check cannot pass vacuously.

## A test of mine fired on its own documentation

The first version of `test_no_ancestry_assertion_is_USED_between_them` grepped
both files for forbidden strings and failed **on `measure_retro.py`'s explanation
of why ancestry is invalid** — which quotes the very command it must not rely on.

A check that cannot tell a comment from a statement is not a check, and
`gpu-sim-u2o.6.6.6.1`'s scope lint already learned that lesson in this repo. It now
strips comments and docstrings with `tokenize` and inspects the executable body.

## Additive, and it says so

No row was relabelled, edited, or regenerated. The rows are read via
`git show <branch>:<path>`, and the file digest plus a per-row rollup are recorded
so a reader can prove it. The statement names the branch **UNMERGED** while 6.7.1
is unmerged, so it cannot be mistaken for a claim about master.

## Artifacts

CPU only, no GPU, no `ts` job.

```
experiments/gpu-sim-u2o.6.6.6.2.1/measure_retro.py   re-derives every claim at run time
experiments/gpu-sim-u2o.6.6.6.2.1/retro_statement.json
experiments/gpu-sim-u2o.6.6.6.2.1/{run.sh,output.log,job.json}
tests/test_retro_rows_uncached.py                    10 tests
```
