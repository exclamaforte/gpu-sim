# gpu-sim-u2o.10.2 REPORT: ranking metrics verified + run on real data

CPU-only bead, no GPU used. No implementation changes needed (the module
was correct -- unlike its 10.1 sibling); the work is verification + real
data + doc.

## Verification (tests/test_ranking_102.py, 4 tests)

- 25 randomized trials vs scipy.stats.spearmanr/kendalltau with ties:
  agreement to 9 decimals
- hand-computed tie case ([1,1,2] vs [1,2,2]): tau = rho = 0.5, derived
  by pair counting in the test comment... (counted here: conc=1, t_x=1,
  t_y=1 -> 1/2; ranks [1.5,1.5,3] vs [1,2.5,2.5] -> cov 0.75 / 1.5 = 0.5)
- extremes: reversal -1.0, identity 1.0, top-k 1.0/0.0

## Real data (family_holdout test split, n=57)

- feature-17 single-feature ranker: spearman 0.886 / kendall 0.751 /
  top5 hit 0.4 -- genuine rank signal in one raw feature, no model needed
- per-op train-median is constant on the single-op test split:
  spearman/kendall 0.0 via the all-tie guards (pinned: 0.0, not NaN) --
  the degenerate case is recorded, not hidden

## Backing doc

eval.md: only the 10.2 placeholder line replaced (section above); 10.3/
10.4 lines untouched for their beads. (Note: 10.1's prepend lives on its
own unmerged branch; no textual conflict -- different lines.)
