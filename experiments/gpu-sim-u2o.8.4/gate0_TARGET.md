# Gate 0: target transform statement (gpu-sim-u2o.8.4, PI ruling 2026-10-13)

## Transform

All arms train on `log_target = ln(median_runtime_ms + 1e-6)` with MSE loss,
and are scored in the native unit via `pred_ms = exp(log_pred)` (MAPE).
Set in `build_data_84.py:log_target`; consumed as `target` in
`train_ablation.py:AblationDataset`. The arms NEVER trained raw: the
ruling's "if raw" condition does not trigger; this null is already the
log-target result.

## Log-var analysis (justification)

- `var(ln ms) = 6.209` over n=3261 (mean-predictor MSE floor).
- Raw targets would be worse: ms spans 0.0017..330.5 (5 decades); raw MSE is
  dominated by the top decade, so gradient signal ignores 99% of rows.
- The bead's criterion is MAPE -- scale-free. `MSE(ln)` ~= mean squared
  relative error for small errors, so log-target is the loss most aligned
  with what is scored. Standardization (z-score) would only rescale the
  loss; under Adam + cosine schedule that is ~a no-op, not a fix for a
  2.5-MSE floor.
- After 100 epochs at 199K params: train MSE 4.0-4.3 (floor 2.5),
  val MSE 3.7-6.5. Underfitting persists UNDER the aligned transform.

## Conclusion

Gate 0 exhausted: the null is not a transform artifact. Capacity is the
binding constraint -> path (A) authorized (one bigger-model job).
Fallback (B) pre-committed if the bigger scale is still vacuous.
