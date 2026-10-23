# gpu-sim-9ul: per-family cost models

Verdict: GAP-REMAINS. CV-selected per-family models still miss the 0.25 bar on
all three gate classes: elementwise 0.369, matmul 0.268, reduction 0.706.
Selection came from 3-fold CV on the training split; test split touched once.
Predictions: experiments/gpu-sim-9ul/predictions_9ul.jsonl (per-family winners).
See experiments/gpu-sim-9ul/metrics.json.