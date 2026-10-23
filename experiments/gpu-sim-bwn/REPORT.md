# gpu-sim-bwn: reach the 25% parity bar

Gate criterion (u2o.17): median |sim-measured|/measured <= 25% per op class.
Verdict: GAP-REMAINS on the gate's own op classes (elementwise, matmul, reduction); softmax and transpose are extras. No gate class reaches the bar (metrics.json:bar) with GBT median relative error:
elementwise 0.502 (baseline 0.506), matmul 0.266 (0.260), reduction 0.459
(0.462), softmax 0.431 (0.415), transpose 0.257 (0.245); pooled 0.275.
Feature spec: 8 physical descriptors + 33 num33 slots = 41, 12 informative.
Scope caveat: kernel_us on generated kernels is a proxy for u2o.17's op-class
measurement; passing here is necessary but not identical to passing the owner's
gate. Size-only control: WITHDRAWN (PI reconciliation, 2026-10-23): bwn's stated
definition (log1p(size), block excluded, HistGBT 300/6/0.05/5 seed 0, z-scored)
reproduces 9ul's numbers, not bwn's committed ones; four implementations span
0.015-0.839 on matmul. Retained in metrics.json / predictions_sizeonly.jsonl as a
withdrawn control, not a result. See experiments/gpu-sim-bwn/metrics.json.