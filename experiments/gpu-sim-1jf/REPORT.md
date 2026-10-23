# gpu-sim-1jf: power the sound target

Verdict: UNDERPOWERED (stage 2 not executed). Stage 1 delivered 673 usable rows
across 5 families (elementwise, matmul, reduction, softmax, transpose) in 7.07
GPU-minutes (5714 rows/hour), far above the 100-rows/hour stop threshold.
Target: kernel_us (batched difference), ln transform, gate threshold as in bk9.
No model comparison was performed in this bead; the training pipelines (9.4/9.5/9.1)
are not wired to this dataset. Supersedes gpu-sim-u2o.11.4's headline only insofar
as the target is now sound; the old NO stands on the original dispatch-heavy target.
See experiments/gpu-sim-1jf/metrics.json.