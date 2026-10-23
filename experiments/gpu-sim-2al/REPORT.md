# gpu-sim-2al: stage 2 on powered target

Verdict: UNDERPOWERED. 673 usable rows, 5 families, matmul held out. Only
latency->latency head has targets (roofline 2.608, GBT 2.993 log-MAE);
transformer training did not converge (NaN), so no model ranking is reported.
Numeric-only (all rows from measurement harness; no source tokens). Feature
mapping: 33 slots, 24 masked (constant zero) per wiring_check.
Supersedes gpu-sim-u2o.11.4: its NO stands on the original dispatch-heavy
target; this bead does not overturn it. See experiments/gpu-sim-2al/metrics.json.