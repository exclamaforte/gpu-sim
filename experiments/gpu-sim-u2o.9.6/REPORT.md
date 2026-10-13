# gpu-sim-u2o.9.6 REPORT: shared TrainingLoop on real data

Audit demand (u2o.15): the loop had only trained on random tensors. Closed
here: a tiny dual-head model (d64/4h/1L, concat) trained 30 steps on 9.1's
real 456-row corpus (family_holdout_v1 342/57/57) through the SHARED
TrainingLoop -- CPU, ~52 s, no GPU used.

## Observed

- train loss 11.61 -> 7.07 (first -> last logged)
- val losses at hooks 10/20/30: 11.66 / 7.68 / 6.90, best tracked
- eval_fn hook fired exactly at [10, 20, 30]
- best_model.pt written; rotating ckpts bounded (keep_last_n=2)
- resume round-trip: fresh loop loads best ckpt, step restored to 30,
  all params bit-equal
- real batches carry sigma_log > 0: the 6.6 measurement floor engages
  through _batch_sigma (not the 0.0 default)
- test-set evaluate returns loss/mape/mse/r2/num_samples

## Code change (minimal, audited)

- train_trans_91: the inline family_holdout split factored into
  family_holdout_split(rows, repo_root), run() calls it. Diff is the pure
  move + call; split sizes 342/57/57 reproduced bit-identical downstream.
- No changes to the loop itself (9.3 already corrected its loss); this
  bead proves the loop, it doesn't redesign it.

## Artifacts

- tests/test_training_loop_96.py: the proof (2 tests, CPU)
- experiments/gpu-sim-u2o.9.6/metrics.json: numbers above
- run.sh replays the proof test. Transient ckpts/logs live in
  experiments/gpu-sim-u2o.9.6/_proof/ (gitignored, never committed).
