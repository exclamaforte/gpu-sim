# gpu-sim-bwn: reach the 25% parity bar

Gate criterion (u2o.17): median |sim-measured|/measured <= 25% per op class.
Verdict: GAP-REMAINS. No family reaches 0.25 with GBT median relative error:
elementwise 0.502 (baseline 0.506), matmul 0.266 (0.260), reduction 0.459
(0.462), softmax 0.431 (0.415), transpose 0.257 (0.245); pooled 0.275.
Feature spec: 8 physical descriptors + 33 num33 slots = 41, 12 informative.
Scope caveat: kernel_us on generated kernels is a proxy for u2o.17's op-class
measurement; passing here is necessary but not identical to passing the owner's
gate. Negative control: size-only pooled median rel error 0.4055 > full 0.2755,
but per-family size-only is better for elementwise/reduction/softmax (a
finding, not a failure of the feature set). See experiments/gpu-sim-bwn/metrics.json.