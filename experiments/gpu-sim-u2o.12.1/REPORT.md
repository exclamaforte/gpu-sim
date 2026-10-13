# gpu-sim-u2o.12.1 REPORT: v1 step-time/MFU on the 5090

Job 338826cb exit 0. Synchronized (cuda.synchronize around each step),
median of 20 after 5 warmup, full training step (forward + shared NLL +
backward + AdamW) at batch 32 x seq 512 on the merged v1 shape
(d512/8h/12L, numeric 33, dual-head concat, 38,950,658 params).

## Numbers (RTX 5090)

- step time 145.2 ms, 16,384 tokens/step
- achieved 26.38 TFLOP/s (6*N*S*B upper-bound FLOPs; measured column)
- MFU 31.9% vs in-tree 82.6 peak (flagged UNSOURCED in hw_5090.py;
  inherits that uncertainty -- achieved is the column to trust)
- train consumes 220.4 ex/s; data-gen (6.7.1 batched, read from committed
  artifact in-job) produces 167.3 ex/s

## Verdict (input to 12.2)

Data-gen is still slower, but only 1.32x behind -- the sides are nearly
balanced at v1 scale with batched generation. This CORRECTS the Sep-28
answer (97-6105x), which combined unbatched generation (2.39 ex/s) with
other model sizes: both changes since (v1 shape fixed at 39M, 6.7.1
batching at 167.3 ex/s) moved the ratio by two orders of magnitude.
12.2's framing updates accordingly: data-gen still first, but training
efficiency is no longer negligible -- a 2x train speedup would flip the
bottleneck. Caveats (same as Sep-28): throughput comparison across
separate jobs, not end-to-end wall clock; gen is warm steady-state.

Corroboration: Sep-28's 47M-large rung consumed 233 ex/s; this 39M v1 at
220 ex/s is consistent (different shape/loss, same class).
