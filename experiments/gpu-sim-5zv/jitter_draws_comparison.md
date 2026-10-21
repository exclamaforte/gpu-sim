# gpu-sim-5zv AC3: Jitter draws vs model prediction

## Source
- Fabric draws: /tmp/scratchpad/jitter_draws.txt (1000 transfers, jitter_us=10.0)
- Model prediction: epic_cluster/link_model.py predict() with jitter_us=10.0

## Results
- n = 1000
- Fabric mean = 4.8859 us, model predicts 5.0 us (error 2.3%, tol 5%)
- Fabric sd = 2.8988 us, model predicts 2.8868 us (error 0.4%, tol 15%)

## Interpretation of 2.3% mean error
The 2.3% mean deviation (largest in this file) is consistent with sampling
noise: N=1000 draws from U(0,10) has expected sample-mean standard error
sigma/sqrt(N) = 2.887/sqrt(1000) = 0.091 us, or 1.8% of the predicted mean.
The observed 2.3% is within ~1.3 standard errors, so no model bias is
indicated. The draws are N=1000.

## Verdict: MET (within pre-registered tolerance)
