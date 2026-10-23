# gpu-sim-tsy: re-run 10.4 on a sound target

Verdict: UNDERPOWERED. The sound target is kernel_us (batched scale minus
batched empty, K=40 launches per region, gate threshold
kernel_us >= 5 * max(within-process std, 6.6.1 residual)), on ln(kernel_us).
Only 8 of 25 (family, size) cells pass the gate (elementwise 2^22; reduction
2^22; matmul 2^18/2^22/2^24; transpose 2^22/2^24; softmax 2^22), so the
dataset has 8 usable rows across 5 families -- far below the 500-row floor
and with no model training performed. No model ranking is reported.

11.4's headline (transformer loses to GBT ~1.3x/~9.5x) was measured on a
target that is ~97% dispatch overhead (gpu-sim-u2o.7.1). On the sound target
this bead cannot test that verdict; the honest output is the data requirement:
~492 additional usable rows, estimated ~16 GPU-hours at ~2 min per cell.
See experiments/gpu-sim-tsy/metrics.json.