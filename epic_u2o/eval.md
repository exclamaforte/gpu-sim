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

# placeholder gpu-sim-u2o.10.2
# placeholder gpu-sim-u2o.10.3
# placeholder gpu-sim-u2o.10.4
