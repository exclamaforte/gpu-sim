# gpu-sim-u2o.10.1 REPORT: core metrics run on real data

CPU-only bead, no GPU used.

## Correction (same mismatch class as 9.3)

`metrics_eval.calibration_coverage` / `expected_calibration_error` /
`core_metrics` took `log_var` and exp()'d it -- but the only producer
(DualHeadPredictor) emits variance. No in-repo callers existed, so the
parameters were renamed to `var` (not shimmed), with `meas_sigma` adding
the 6.6 floor in quadrature. Negative control pins it: y=1, mean=0,
var=0.25 gives coverage 0.0 under variance reading vs 1.0 under the old
exp() reading.

## Real-data run (constant train-mean baseline, test split n=57)

- log space: RMSE 1.134, R2 -0.001, MAPE 24.99 (scale-fragile, noted),
  coverage_90 0.895, ECE 0.175
- raw ms: MAPE 109.65 (heavy-tail motivation made concrete), RMSLE 0.025
- RMSLE-on-log-inputs degenerates to 0.0 (kept visible as a warning):
  RMSLE is a raw-space metric
- coverage 0.895 with empirical variance + per-row floors shows the
  calibration machinery works before any model head is plugged in

## Backing doc

eval.md's placeholder pattern partially retired: 10.1 section written
(definitions, scale care, numbers, variance contract); 10.2/10.3/10.4
lines left for their beads.

Suite check + commit next. run.sh replays (test rewrites metrics.json
deterministically -- same constant baseline, same split).
