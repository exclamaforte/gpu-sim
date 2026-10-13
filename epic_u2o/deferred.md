# Deferred: cross-GPU generalization
## 13.1 whole-program / multi-kernel prediction (GATED, design only)

Entry gate (machine-checked by tests/test_gate_131.py): the transformer
beats GBT on both head-to-head triples AND D8 multi-stream measured
composition data exists. Gate is CLOSED (beats 0/2 per 11.4) -- building
composition on a per-kernel model that loses to GBT compounds the gap.
Design when open: per-kernel mus summed along the critical path + a
learned overlap term for concurrent streams/memcpy (D8 generator
provides the multi-stream programs; measure end-to-end, fit the
interaction residual, not the total). Deliverable then: composition
error vs sum-of-mus on held-out programs. No compute spent here by
design.
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
## 13.5 active learning (selector shipped, loop gated, retrospective negative)

Shipped: epic_u2o/acquisition.py -- variance-topk (+ sigma floor) and
op-stratified variant with overlap metric (tested). Retrospective on
real 11.2 head outputs (57 rows, read cross-branch): total-variance
top-10 overlaps error-top-10 2/10 (~chance 0.175) and hits the failure
triple 7/10 vs 28/57 base (p~0.2, not significant). NEGATIVE: with an
over-conservative head (11.2 coverage 0.965), uncertainty sampling does
not prioritize better than chance -- calibration must come first; the
closed loop (measure-selected, retrain, re-rank) waits for that + a GPU
campaign. Sidebar: raw-var ranking gave 4/10 + 10/10 triple -- the sigma
floor reshuffles acquisition, so the total-variance contract matters.
