# Results, methods, and negative findings (gpu-sim-u2o.11.5)

Status: real numbers, every figure sourced from a committed artifact
(generated table block below is pinned by tests/test_writeup_115.py --
prose cannot drift). Synthetic fixtures retired with this writeup.

## Methods (as run)

- Corpus: 456 real measured rows (stage-1 subset), family_holdout_v1
  split -- test relu 57 / val mul 57 / train 342, zero shared digest
  groups (7.7 rule, 9.5; factored for reuse in 9.6).
- Model: transformer encoder d512/8h/12L + concat numeric fusion,
  dual-head (mean, variance) with 6.6 sigma-floor NLL (9.1, 9.2, 9.3);
  38,950,658 params, 200 steps.
- Baselines: analytic roofline (9.4), GBT/HGBR on 33 numeric features
  (9.5). Shared loop proven on real data (9.6).
- Metrics: MAPE/RMSE/RMSLE/R2 + calibration (10.1, variance contract),
  Spearman/Kendall/top-k (10.2). Comparison joined, not retyped (10.4);
  headline derived, not typed (11.4). Runs recorded via
  scripts/bd-exp.sh (11.6).

## Results (generated block, do not hand-edit)

| triple | roofline 9.4 | GBT 9.5 | transformer 9.1 |
| latency->latency | 0.5638 | 0.1072 | 0.1438 |
| latency->memory | 0.6873 | 0.0274 | 0.26 |
Constant train-mean baseline (10.1, n=57 test): log-RMSE 1.134, R2 -0.0012, raw-ms MAPE 109.6503%, coverage_90 0.8947.
Feature-17 single-feature ranker (10.2, n=57 test): spearman 0.8863, kendall 0.7513.

Headline (11.4): NO -- the transformer beats the roofline ~4x/~2.6x and
loses to the numeric-only GBT ~1.3x/~9.5x on both triples. The sequence
channel earns nothing at this scale.

## Negative findings (the deliverable, not an omission)

- Transformer loses to GBT on both head-to-head triples (above).
- Constant baseline raw-ms MAPE 109.65%: the heavy tail dominates naive
  predictors; log-space MAPE (24.99%) is scale-fragile near y=0.
- RMSLE applied to log-space inputs degenerates to 0.0 (10.1): RMSLE is
  a raw-space metric; kept visible as a warning.
- Cross-attention fusion loses to concat on both triples despite +1.3%
  params (9.2): the default stands.
- 20M-scale arms of 9.2 (a dims-threading clobber) were superseded by the
  39M rerun; recorded in 9.2's REPORT, not hidden.
- Dead ends kept: 10.4.1 transformer ranking (neutral) -- model ranking
  equals one raw feature; Sep-28 bottleneck ratios superseded (see 12.1,
  pending review).

## Pending-review extensions (cited by branch, not numerated here)

- 11.1 (lab/gpu-sim-u2o.11.1): 32-row overfit sanity.
- 11.2 (lab/gpu-sim-u2o.11.2): measurable-memory failure concentration,
  real-head calibration, per-row predictions.
- 11.6 (lab/gpu-sim-u2o.11.6): scripts/bd-exp.sh + close rule.
- 12.1 (lab/gpu-sim-u2o.12.1): v1 step-time/MFU profile.

## Conclusion

The transformer does not earn its complexity vs GBT on this holdout
(R2-equivalent: GBT wins ~1.3x/~9.5x). It beats the roofline decisively
and carries a working uncertainty head. Next: failure-mode-driven
representation work (measurable-memory), not scale.
