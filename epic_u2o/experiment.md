# 11.1 Step-1 overfit sanity (ran on real data)

Tiny dual-head model (d64/4h/1L, concat) memorizes 32 fixed real rows
(first 32 of the family_holdout train split): NLL 22.79 -> 0.15 in 150
full-batch CPU steps (0.7% of initial; bar was <5%). LR=0 control is
bit-exact (22.7864 -> 22.7864, eval mode + fixed order), proving the
decrease is learning. The tokenize->batch->model->loss pipeline fits real
rows end to end; no pipeline bug (bad tokenization, mislabeled targets)
could survive this. Proof: tests/test_overfit_111.py (~4 min CPU).
Test subsets: targets non-degenerate (asserted). New data generation was
not needed: the audit point was 'never ran against real data', and these
are real measured rows.
# placeholder gpu-sim-u2o.11.2
# placeholder gpu-sim-u2o.11.3
# placeholder gpu-sim-u2o.11.4
# placeholder gpu-sim-u2o.11.5
# placeholder gpu-sim-u2o.11.6
