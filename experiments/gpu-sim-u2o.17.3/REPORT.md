# gpu-sim-u2o.17.3 REPORT: simulated trace from real per-op scaling laws

CPU-only bead, no GPU (consumes 17.5's committed fixture: 28 cases,
5 op classes, 5090, torch 2.14).

## Method

Per-op-class leave-one-size-out log-log fit (size = elements, matmul =
flops) vs naive roofline on the same held-out cases. metrics.json holds
per-case detail.

## Numbers (mean abs rel err, fit vs naive; knee dose merged, 7 sizes)

- matmul (5 sizes): 0.333 vs 0.465 -- fit wins
- reduce (4 sizes): 0.460 vs 3.567 -- fit wins big (naive off by 3.5x+)
- elementwise (7 sizes): 0.964 vs 2.413 -- fit wins AFTER the knee dose
  (was 2.225 vs 1.272 pre-dose). Knee localized: 8M->16M jumps ~6x for 2x
elements (L2 96MiB exhaustion: 16M fp32 x3 tensors = 192MB > L2); 1M->8M
scales near-linearly. Dose: job 72c28c17, 9/9 cases, no throttling.
  away): 1024-1M rows sit on the launch floor (~0.001ms) then jump 30x at
  16M; one slope cannot span the knee. Two-regime (floor+linear) form is
  the named follow-up.
- chain (2 sizes), launch_bound (1): no fit, gaps recorded.

## What this unblocks

17.4's comparison runner now has a real simulated-trace path (this) plus
the naive path, both CPU-rerunnable against the committed fixture. The
elementwise knee is the same launch-bound/measurable regime split the
modeling epics keep hitting -- the trace path and the learned model agree
on where the hard part is.

## No backing-doc edit

No 17.3 placeholder exists in-tree (complete.md is a title line); the
bead lives in the gate script + this artifact.
