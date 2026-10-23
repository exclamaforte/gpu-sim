# Evaluation (epic gpu-sim-u2o.10)

## 10.1 core metrics (implemented, run on real data)

`epic_u2o/metrics_eval.py` (numpy-only): MAPE, RMSE, RMSLE, R2, plus
calibration coverage (90% interval hit rate) and 1-sigma ECE over the
predicted-variance head. Variance contract (see 9.3): calibration inputs
are VARIANCES as emitted by DualHeadPredictor, with the 6.6 per-cell floor
added in quadrature via meas_sigma. (The module previously took log_var
and exp()'d it -- same mismatch 9.3 fixed in the loop; corrected here,
no in-repo callers existed.)

Scale care (demonstrated, not assumed): RMSLE is defined on raw
nonnegative runtimes; applied to log-space inputs it degenerates (0.0 on
the test split -- kept visible in metrics.json as a warning, not a
result). Log-space MAPE is scale-fragile near y=0. The interpretable
relative metric is MAPE/RMSLE on raw ms; model-space tracking uses
log-RMSE/R2.

Constant train-mean baseline on the family_holdout test split (n=57):
log-RMSE 1.134 / R2 -0.001 (as expected for a constant), raw-ms MAPE
109.7% (the heavy-tail motivation: slow examples dominate), calibration
coverage_90 0.895 with empirical train variance + per-row floors, ECE
0.175. Reference: 9.1's 39M concat reaches log-MAE 0.144 on the same
split -- the constant's 1.134 log-RMSE is the floor to beat, and the
coverage number shows the calibration machinery works before any model
head is plugged in (39M-head calibration awaits saved GPU predictions --
flagged for 10.4).

# 10.2 ranking metrics (implemented, verified, run on real data)

`epic_u2o/ranking_metrics.py` (numpy-only): Spearman rho (Pearson on
average ranks), Kendall tau-b with tie correction, top-k hit rate
(lower-is-better latency: fraction of true top-k in predicted top-k).
Verified against scipy.stats on 25 random trials with ties (9 decimals)
plus hand-computed tie case (tau = rho = 0.5). Real data (family_holdout
test split, n=57): raw feature-17 ranker reaches spearman 0.886 /
kendall 0.751 / top5 0.4 -- rank signal exists before any model. The
single-op test split degenerates op-conditioned predictors to constants;
the all-tie guards return 0.0 (not NaN), pinned by test. For autotuning
(J3): order first, absolute error second -- this module is the scoreboard.
# placeholder gpu-sim-u2o.10.3

## 10.4 model vs baselines (joined from committed artifacts, no retraining)

Source of truth for the headline experiment (11.4): log-MAE on the
family_holdout test split (relu, n=57), all cells provenance-pinned
(experiments/gpu-sim-u2o.10.4/metrics.json + join_104.py drift guard).
latency->latency: GBT 0.107 < transformer 0.144 < roofline 0.564.
latency->memory: GBT 0.027 < transformer 0.260 < roofline 0.687. GBT wins
both; the transformer beats the roofline ~4x/2.6x and loses to the
numeric-only GBT -- the sequence channel earns nothing at this scale.
NOTE (gpu-sim-tsy, 2026-10-23): 11.4's headline above was measured on a
target that is ~97% dispatch overhead (gpu-sim-u2o.7.1). The re-run on the
sound target (kernel_us, gate threshold) is UNDERPOWERED: only 8 usable rows,
so no model ranking is reported. See experiments/gpu-sim-tsy/metrics.json and
REPORT.md. 11.4's NO is therefore unqualified on the sound target; it stands
only as a result on the original target.
Gaps recorded, not filled: ranking/calibration need per-row predictions
none of the three approaches committed; generalization suites await 10.3;
memory->memory has no learned-approach test cell (roofline 1.1209 is from
its own analysis, different basis).
