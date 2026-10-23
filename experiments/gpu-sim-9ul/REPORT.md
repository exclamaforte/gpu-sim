# gpu-sim-9ul: per-family cost models

Verdict: GAP-REMAINS. CV-selected per-family models still miss the 0.25 bar on
all three gate classes: elementwise 0.369, matmul 0.268, reduction 0.706.
Selection came from 3-fold CV on the training split; test split touched once.
Predictions: experiments/gpu-sim-9ul/predictions_9ul.jsonl (per-family winners).
See experiments/gpu-sim-9ul/metrics.json.
Size-only reconciliation: 9ul's original size-only column was a per-family model, not comparable to bwn's global size-only negative control. Global size-only reproduces bwn (elementwise 0.174, matmul 0.839, reduction 0.200). Load-bearing control is bwn's global size-only. See metrics.json size_only_reconciliation.