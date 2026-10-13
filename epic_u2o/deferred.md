# Deferred: cross-GPU generalization
# gpu-sim-u2o.13.1 placeholder
## 13.2 cross-GPU conditioning (schema shipped, data gated)

Shipped: epic_u2o/gpu_spec.py -- fixed-order 6-dim conditioning vector
(sm_count, peak_fp32_tflops, mem_bw_gbs, l2_bytes, shared_mem_per_sm,
max_sm_clock) mapped from sourced hw_5090 constants; missing fields
raise, provenance per entry, rtx5090 vector pinned (sm=170). The unsourced
82.6 peak is carried with its flag, not laundered. Gate (test-pinned): a
second GPU needs its spec entry + measured rows for a holdout test; only
rtx5090 exists today, so training with the vector is gated. Design when
open: concatenate spec vector to the numeric channel, hold one GPU out
entirely, report cross-GPU MAE vs in-GPU MAE.
# gpu-sim-u2o.13.3 placeholder
# gpu-sim-u2o.13.4 placeholder
# gpu-sim-u2o.13.5 placeholder
