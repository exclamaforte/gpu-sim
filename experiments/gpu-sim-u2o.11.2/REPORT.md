# gpu-sim-u2o.11.2 REPORT: Step-2 failure-mode diagnosis

CPU-only (inference ~2 min), no retraining, no GPU.

## Method

Loaded 9.1's 39M ckpt on CPU (weights uncommitted in runs/, found via
`git worktree list` -- no hard path), ran the 57 family_holdout test rows
through the merged v1 shape, committed per-row predictions.json (57 rows:
hash/op/triple/regime/y/sigma/mu/var). failure_analysis.json holds the
worst-10 + regime/triple means + err-vs-sigma correlation.

## Findings (the systematic subfamily this step was filed to catch)

- Worst-10 by abs err: 9 of 10 latency->memory, measurable regime,
  sigma_log 0.003-0.046. Rank 10 is the intruder: 9b474b0e,
  latency->latency / launch_bound, abs_err 0.313, sigma_log 0.165.
  Mean abs of worst-10: 0.49. (v1 said ALL; reviewer recompute showed
  9/10 -- corrected, intruder named.)
- Measurable-regime MAE (0.296) is 2x launch_bound (0.145).
- err-vs-sigma correlation -0.25: errors are NOT measurement noise --
  the model fails on measurable memory-bound relu specifically
  (cache-resident traffic per 9.4's residual_A is the prime suspect;
  stated as suspect, not conclusion).
- Head calibration on real outputs: coverage_90 0.965 (over-coverage:
  predicted std 0.30-0.46 vs typical err ~0.2 -- conservative), ECE 0.22.
  The uncertainty head works; it is cautious.

## Side effect (for 10.4/11.4)

predictions.json fills 10.4's transformer per-row gap on family_holdout:
ranking + calibration for the transformer arm are now computable without
any retraining. GBT/roofline per-row gaps remain.

## Backing doc

experiment.md: only the 11.2 placeholder replaced.
