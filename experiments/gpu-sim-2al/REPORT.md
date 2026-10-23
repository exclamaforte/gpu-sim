# gpu-sim-2al: stage 2 on powered target

Verdict: CONFIRMED. On the sound target (kernel_us, batched difference, ln transform,
gate threshold as in bk9), with 673 usable rows across 5 families, the primary
protocol (within-family size/config interpolation, pooled) gives log-MAE
latency->latency: roofline 1.0784, GBT 0.3709, transformer 2.4205. The transformer
loses to GBT (2.4205 vs 0.3709), confirming 11.4's NO on a sound target. Numeric-only
(no source tokens). Secondary leave-one-family-out (matmul) is unlearnable by
construction (mean predictor 2.6083 = roofline) because the analytic
roofline has no family descriptor for an unseen family and degenerates to
the training mean (possibility (b)); reported descriptively. Mean-predictor
floor is stated for every cell. Supersedes gpu-sim-u2o.11.4's headline: on a sound target
the transformer still loses to GBT. See experiments/gpu-sim-2al/metrics.json.