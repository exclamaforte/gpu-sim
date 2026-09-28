# gpu-sim-u2o.10.3 — three generalization suites, and a check that fails loud

The bead asks for held-out suites analogous to cclm's generalization
distributions. What is actually hard here is not the suite definitions — it is
that a suite must be **claimed only if the corpus can support it**, and the
support check must fail loud rather than report a degenerate split as clean.

That is not hypothetical. `gpu-sim-u2o.7.7` ran the existing splitter on the
264 real measured rows and got:

```
train=264  val=0  test=0   ...   leakage_train_test: 0
```

A split with no test set at all, reported as **zero leakage** — the check was
computing the intersection of two empty sets.

## The suites

| suite | holds out | on the 264 reference rows |
|---|---|---|
| S1 unseen problem size | the two **largest** sizes | train 136 / eval 128 |
| S2 unseen family | 2 of 8 structural families | train 198 / eval 66 |
| S3 unseen occupancy | 1 of 8 launch-config regimes | train 232 / eval 32 |

S1 holds out the largest sizes specifically: holding out a middle size tests
interpolation and would be *reported* as S1 while measuring something easier.
S3's unit is `(block size, coverage class)`, because a full-coverage and a
1-block config at the same block size are different regimes, not two points in
one.

## The result: 3 of 3 are non-degenerate — and I pre-wrote the opposite

My first version of the artifact **asserted in prose** that the family holdout
was *not* supportable at 8 families. Running it says otherwise: S2 clears the
mechanical floors comfortably, 198 train against 66 eval.

That is the "conclusion written before the data" failure catalogued all night,
made in my own artifact one turn after I catalogued it. The only reason it is
visible is that the verdict is now **derived from `feasibility()`** rather than
typed — and there is a test that the summary reports what the suites returned.

## Necessary but not sufficient — and the distinction matters

The mechanical floor asks whether a suite is **non-degenerate**: does it have an
eval side, a train side, enough rows to mean anything. 7.7's split failed that
question and was reported clean. This check answers it.

What it does **not** answer is whether S2 is **informative**. Holding out 2 of 8
elementwise families leaves 6 near-identical templates to train on. The corpus
decision on `gpu-sim-u2o.6.7.1` argued ≥ 24 families are needed for S2 to be a
real test. Both statements stand:

> S2 is **runnable**. S2 is not yet a **meaningful** generalisation test.

The check is silent on informativeness *by design*: deciding that is a
judgement, and a mechanical gate that encoded one would be a gate nobody could
argue with — the `discriminates` boolean, in a different costume.

## Negative controls

The load-bearing tests are the ones that must **fail**:

* empty eval set → raises, and says the leakage number over it is the
  intersection of two empty sets
* empty train set → raises
* train below the floor of 30 → raises
* eval below the floor of 5 → raises
* a corpus too thin to split → reported **unsupported with its reason**, not
  raised, because "this corpus cannot support it" is a result
* shrinking the corpus can only reduce support (monotone)

## One of my own tests was wrong in the way this lab keeps being wrong

The first version reconstructed S3's held-out count by string-matching
`"block=64"` against the label, which counted every block-64 row instead of the
`(block, coverage)` **regime**, and failed `128 != 64`. The test was wrong, not
the module — and wrong by matching a substring of a label instead of the
quantity the label names. Fixed to reconstruct the regime.

## What this does and does not deliver

**Delivers:** the three suite definitions, and a support check that fails loud
on a degenerate split.

**Does not deliver:** a generalization *result*, because there is no model yet.
That is `gpu-sim-u2o.10.4`'s job, and it is blocked on `9.x`.

The 264 reference rows carry no `protocol` field — the corpus condition was
written after they were measured, so they cannot be stamped retroactively
without inventing a label. They are known single-run (7.1's own harness, one
process per invocation) and are treated as the reference throughout. Stated
rather than papered over.
