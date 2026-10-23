# gpu-sim-1jf: power the sound target

Verdict: STAGE2-SPLIT (stage 1 passes; stage 2 split to gpu-sim-2al). Stage 1 delivered 673 usable rows
across 5 families (elementwise, matmul, reduction, softmax, transpose) in 7.07
GPU-minutes (5714 rows/hour), far above the 100-rows/hour stop threshold.
Target: kernel_us (batched difference), ln transform, gate threshold as in bk9.
Stage 2 (model comparison) is split to gpu-sim-2al; no model number appears here. Supersedes gpu-sim-u2o.11.4's headline only insofar
as the target is now sound; the old NO stands on the original dispatch-heavy target.
See experiments/gpu-sim-1jf/metrics.json.