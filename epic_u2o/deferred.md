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
## 13.3 autotuning surrogate (search shipped, validation gated)

Shipped: epic_u2o/surrogate_search.py -- greedy/UCB rank over launch
configs with regret vs measured truth (tested: perfect surrogate regret
0, noisy bounded, UCB explores). Gate (test-pinned, 5 files scanned):
max 1 config per kernel_hash corpus-wide -- there is NOTHING to rank
within-kernel today. Opening needs a multi-config measurement campaign
(same kernel x N launch configs, real GPU) + a cost model beating GBT.
Then: rank by mu, validate top-k on GPU, report regret + top-k hit.
# gpu-sim-u2o.13.4 placeholder
# gpu-sim-u2o.13.5 placeholder
