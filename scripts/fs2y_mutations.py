"""Mutation registry for the fs2y ledger.

Maps each non-pending ledger mutation name to a short description of intent.
The lab presubmit (x06h) imports this module and checks that every non-pending
ledger row's mutation resolves against this dict; a row whose mutation is absent
is reported as newly unresolved. Descriptions document intent only.
"""
MUTATIONS = {
    "dhfo_memory_check": "dhfo memory-check guard",
    "bch9_fabric_band": "bch9 fabric band robustness",
    "8zqr_generalization": "8zqr cross-class generalization",
    "append_only_guard": "append-only guard for validated records",
    "fs2y_mutations_tracked": "fs2y mutation registry tracked on master",
    "no_pycache_under_experiments": "pycache suppression under experiments/",
    "bk9_gate_flop_aware": "flop-aware bk9 gate calibration",
    "effective_cpu_capacity": "effective CPU capacity helper",
    "ledger_node_ids_validated": "ledger node id validator",
    "no_unconditional_skips": "unconditional-skip guard with allowlist (gpu-sim-1486)",
}