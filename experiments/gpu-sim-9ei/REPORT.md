# gpu-sim-9ei: FAKEGPU size-aware latency
Fallback policy:
- Table source precedence: GPUSIM_OP_LATENCY_JSON env var if set, else experiments/gpu-sim-u2o.13.4/op_latency_table.json (13.4 default).
- In-range size: log-log interpolation between bracketing calibration sizes.
- Out-of-range size: clamp to nearest endpoint.
- Unknown op: legacy default 0.5 ms (documented).
- One-way street: gpu-sim is structural-only; measured table consumed, never training truth.
See epic_u2o/op_latency.py.

Note: mdz calibration entries are keyed by op name only, so the launch_bound relu (64 elements) is intentionally merged into the elementwise relu curve as a launch-floor anchor. This is documented, not accidental. If a launch_bound sigmoid is measured later, it must be keyed separately.

Cross-bead citations: experiments/gpu-sim-u2o.13.4/op_latency_table.json
