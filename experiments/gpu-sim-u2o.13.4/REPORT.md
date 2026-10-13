# gpu-sim-u2o.13.4 REPORT: calibrated latencies wired (CI lane)

CPU-only bead, no GPU. The promoted-to-gate item ships in the phase-5
lane: env/config table, no LD_PRELOAD, D_GPUSIM honored.

## Shipped

- op_latency_table.json: 8 ops x 57 measured rows, medians ~0.004 ms,
  coverage + D_GPUSIM in metadata. Built by committed build_table_134.py
  (rerunnable).
- epic_u2o/op_latency.py: load (GPUSIM_OP_LATENCY_JSON override),
  per-op lookup, legacy 0.5 fallback, spec helper.
- safe_harness: one-line wiring (spec op_type -> env var, fallback
  intact). No interpose path touched.

## Honest limits

- Values are measured medians, not model outputs -- "calibrated" means
  data-driven vs the 0.5 guess, nothing more. Model-derived latencies
  wait for 11.4 to flip.
- Coverage is launch-bound small sizes; unknown ops keep 0.5 (visible
  fallback, not a silent claim).

## Backing doc

deferred.md: only the 13.4 placeholder replaced.
