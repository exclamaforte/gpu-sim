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
# 11.4 headline (from 10.4's verdict, not a re-run)

The judged question -- does the learned model beat both baselines -- is
answered NO on both head-to-head triples: transformer beats roofline
~4x/~2.6x but loses to the numeric-only GBT ~1.3x/~9.5x (ratios computed
from 10.4's cells by headline_114.py, pinned by test). Axes without a
headline (ranking, calibration, suites, mem->mem) are listed in the
artifact, not silently dropped. Per PI ruling, 11.4 consumes 10.4's
verdict; gap-filling is new-leaf work.
# placeholder gpu-sim-u2o.11.5
# 11.6 experiment-recording protocol (exists AND used)

scripts/bd-exp.sh (adapted from cclm): start (hypothesis+parent, one-active
guard) / finish (outcome+kept+metrics metadata+REQUIRED artifact cite,
closes the bead) / board. Guards kept: no verdict overwrite, start-
signature check, needs-review block. Lab metric flags: --logmae --spearman
--kendall --coverage --ece --ex-sec --topk + --set. Proven in production:
gpu-sim-u2o.10.4.1 (transformer ranking from 11.2 mus: rho 0.886, neutral,
kept=true, artifact-cited) is the first bead with real metrics metadata.
Close rule added to AGENTS.md: no gpu-sim-u2o close without a concrete
artifact cite.
