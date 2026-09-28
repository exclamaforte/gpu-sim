# gpu-sim-u2o.7.7 — structural-family split design, on the real data

The 6.15 audit reopened this bead with the note that split logic "may exist in
`structural_split.py` but there is no real dataset with runtime labels for a
split to matter yet." **gpu-sim-u2o.7.1 has now produced that data** — 264
measured rows, ts `20260928t074004-b429f469`. So the split matters, and this
bead ran the existing logic against it rather than against a fixture.

## The pre-existing split is fail-open on the real schema

Measured, not asserted:

```
assign_family(row as 7.1 writes it) -> None
split_by_family(real_rows)           -> train=264  val=0  test=0
analyze_split_quality(that)          -> total_families: 0
                                        leakage_train_test: 0
```

All 264 rows in train. No test set at all. And the leakage analyzer reporting
**zero leakage**.

That last line is the serious one. A leakage check that reports "no leakage" for
a split with no test set is not a weak check — it is a **green result produced
by nothing having been tested**, the same shape as the `8dw` preflight stub and
the `6.3.1` probe that trusted its exit code. The cause is mechanical:
`assign_family()` wants `metadata['family_id']` or a matching `generator`, and
stage1 writes `op_type` at top level and no `generator` at all, so every row
returns `None`, every family pools into one bucket, and the split stops
splitting.

## What a row-level split would have claimed

| quantity | value |
|---|---|
| rows | 264 |
| distinct `kernel_hash` | **264** |
| distinct `token_digest` | **8** |
| rows per digest | 33 |
| test rows sharing a representation key with train | **100%** |
| families shared train/test | **8 / 8** |

A random 85/15 row split leaks **every single test row**, and shares **all
eight** families across the boundary.

Note the two identity columns: `kernel_hash` is distinct on all 264 rows, so a
leakage check keyed on it reports **0%** on a split that leaks 100%. The model
never sees `kernel_hash`; it sees the token sequence. **Keying the check on the
wrong identifier is precisely how the existing analyzer returned zero.**

## A representation finding that only appeared because the split ran

`token_digest` is **1:1 with `op_type`**, and it is **invariant to both
`problem_size` and `launch_config`** (`digest_invariant_to_problem_size: true`,
`..._to_launch_config: true` in `metrics.json`). `token_count` is likewise
constant per family — 56 for every `scale` row regardless of size.

So 264 measured rows collapse to **8 distinct model inputs**, and the two things
that actually determine runtime are invisible to the model. That is a
**gpu-sim-u2o.8** finding, and it was not visible before because nothing had
tried to *split* the data and count what the split could distinguish.

## The split this bead recommends

Families sorted alphabetically; first **2 → test** (`add`, `copy`), next **1 →
val** (`div`), the remaining 5 → train. A **stated rule rather than a random
draw**, so the split is reproducible and auditable instead of lucky. It passes
`validate_split()`: **165 / 33 / 66** rows (train 5 families, val ['div'], test ['add', 'copy']), family-disjoint, representation-disjoint, rows conserved.

AND I MIS-TYPED THAT LINE WHILE FIXING IT. I wrote 231/33/66 -- the artifact says 165/33/66 -- in the very edit whose purpose was to make the prose match the artifact, and I described the numbers as written from metrics.json. That is this bead's exact failure mode performed by me, in the paragraph about it, which is why this line is now interpolated from the JSON rather than typed: a number typed by a person is a number that can be wrong, and the whole bead is an argument that derived numbers cannot be.

## Size-range holdout: recorded as IMPOSSIBLE, not implemented

The bead asks for holding out problem-size ranges too. On this data that cannot
be done honestly: there are **4 sizes** (4096 … 262144) and **all 264 rows are
`measured_regime: launch_bound`**. A size-range holdout is only meaningful if the
held-out sizes sit in a different regime, and 7.1's own finding is that they do
not — every size through 2^18 is launch-bound. Implementing the holdout anyway
would produce a number labelled "unseen problem sizes" that is really just
"unseen size, same regime". Recorded as impossible; it becomes real when 7.1's
dataset extends past 2^18 into memory-bound territory.

## Why every degenerate case RAISES

`validate_split()` refuses: an empty side (the exact fail-open above), rows that
do not conserve, a family straddling the boundary, a **representation key**
shared across the boundary, and a val/test side with fewer than 5 examples — a
one-row test set produces a confident number from a single measurement.

`structural_family()` raises rather than returning `None` when a row has no
family, because `None` is what silently pooled all 264 rows into one bucket.

15 tests, 0 failures, CPU only, no GPU, no ts job.

## Two of my own mistakes, both silent

**Three test fixtures asserted nothing.** My first family-leakage fixture
overlapped its own row lists, so row-conservation fired first and the assertion
never reached the check under test. The representation-key fixture reused one
family across the boundary, so the *family* check fired first. Both would have
passed for the wrong reason. They are now built to be well-formed in every
respect except the one under test, and the clean case is asserted *first* so a
broken guard cannot masquerade as a working one.

**A patch silently did not apply.** One `str.replace` had a search string that
did not match the file (`d[:5]` against a file containing `d[:2]`), so it
no-op'd with no error and the test kept failing against old code. Every
subsequent edit asserts its anchor is present and refuses to write otherwise.
A patch that quietly does nothing is indistinguishable from a patch that worked.
