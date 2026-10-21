# gpu-sim-5zv AC3: Jitter draws vs model prediction

## Source
- Fabric draws: /tmp/scratchpad/jitter_draws.txt (1000 transfers, jitter_us=10.0)
- Model prediction: epic_cluster/link_model.py predict() with jitter_us=10.0

## Results
- n = 1000
- Fabric mean = 4.8859 us, model predicts 5.0 us (error 2.3%, tol 5%)
- Fabric sd = 2.8988 us, model predicts 2.8868 us (error 0.4%, tol 15%)

## Verdict: MET (within pre-registered tolerance)
