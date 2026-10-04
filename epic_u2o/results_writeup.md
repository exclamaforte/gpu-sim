# Results, methods, and negative findings (gpu-sim-u2o.11.5)

Status: living document. Numbers below are from lightweight synthetic
fixtures (`epic_u2o/experiment_program.py`); replace with measured values
once real-GPU runs complete. Negative results are part of the deliverable.

## Methods
- Corpus: synthetic stages 1–5 (`generator_stage{1,2,3,4,5}.py`) + mined
  real-world kernels (`corpus_miner.py`), split by structural family
  (`structural_split.py`, `generalization_suites.py`).
- Representation: hybrid source + PTX + numeric
  (`representation_ablation.md`, `tokenizer.py`, `ptx_tokenizer.py`,
  `features.py`).
- Model: transformer encoder + pooling + numeric MLP fusion
  (`model.py`), Gaussian NLL on log-runtime.
- Baselines: analytic roofline (`roofline.py`), GBT on hand features
  (`tree_baseline.py`).
- Metrics: MAPE / RMSLE / R² + calibration (`metrics_eval.py`),
  Spearman/Kendall + top-k (`ranking_metrics.py`); one comparison table
  (`compare_baselines.py`).
- Runs recorded via `scripts/bd-exp.sh` (start/finish/board); dead ends kept.

## Current fixture results (synthetic, NOT headline numbers)
See `/tmp/writeup_draft.md` for the generated table. Transformer stub beats
roofline/GBT stubs by construction; this validates the plumbing only.

## Negative findings (required; fill with real runs)
- (pending) Any suite where the model fails to beat a baseline.
- (pending) Dead ends from `scripts/bd-exp.sh board`.
- Known limits: no real-GPU timings yet; WMMA template is a simplified
  placeholder layout; tree baseline needs sklearn/xgboost installed.

## Conclusion (pending real runs)
- State whether the transformer earns its complexity vs roofline/GBT.
