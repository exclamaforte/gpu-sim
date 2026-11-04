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
    "u5q0_lint_fixture_6338": "6.10 prose-ratio fixture pins the derived 6338x ratio (gpu-sim-u5q0)",
    "s2qj_bypass_path": "s2qj bypass path (gpu-sim-s2qj)",
    "wty2_structure_dominated": "wty2 structure-dominated verdict on unseen families (gpu-sim-wty2)",
    "u5r4_size_scaling_arm": "u5r4 size-scaling arm on divergent_branch (gpu-sim-u5r4)",
    "v5l3_pin_coverage": "v5l3 pin coverage lint (gpu-sim-v5l3)",
    "9wza_tracked_bypass_records": "9wza tracked-first bypass records with divergence check (gpu-sim-9wza)",
}


def _apply_ledger_bad_row(root):
    """Mutation for ledger_node_ids_validated: add an unresolvable row to the ledger."""
    import json, os
    ddir = os.path.join(root, "tests", "fs2y_ledger.d")
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, "plant_bad_row.json")
    bad_row = {
        "name": "plant_bad_row",
        "state": "passing",
        "test": "tests/test_nonexistent_file_xyz.py::test_foo",
    }
    # backup the directory listing (just the new file)
    backup = None
    if os.path.exists(path):
        backup = open(path).read()
    with open(path, "w") as f:
        json.dump(bad_row, f, indent=2)
    # return backup and path; restore deletes the file
    return (backup, path)


def _apply_wty2_corrupt(root):
    """Mutation for wty2_structure_dominated: corrupt a scalar in committed metrics.json."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-wty2", "metrics.json")
    backup = open(path).read()
    d = json.loads(backup)
    # corrupt a scalar MAE
    try:
        d["ac1_reconstruction"]["per_family"]["strided_read"]["gbt_mae"] = 999.0
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path


def _apply_u5r4_corrupt(root):
    """Mutation for u5r4_size_scaling_arm: corrupt a scalar in committed metrics.json."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-u5r4", "metrics.json")
    backup = open(path).read()
    d = json.loads(backup)
    try:
        d["size_scaling_arm"]["gbt_mae"] = 999.0
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path

def _apply_s2qj_unbypass(root):
    """Mutation for s2qj_bypass_path: make the checker ignore bypass records."""
    import os
    path = os.path.join(root, "scripts", "check_gate_citations.py")
    backup = open(path).read()
    mutated = backup.replace("if bypass_path is not None:", "if False and bypass_path is not None:", 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_v5l3_unpin(root):
    """Mutation for v5l3_pin_coverage: remove one pin from a run.sh."""
    import os
    path = os.path.join(root, "experiments", "gpu-sim-006j", "run.sh")
    backup = open(path).read()
    mutated = backup.replace("OMP_NUM_THREADS=1", "OMP_NUM_THREADS=2", 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_9wza_disable_lab_fallback(root):
    """Mutation for 9wza_tracked_bypass_records: disable the .lab fallback so tracked-only resolution breaks the .lab-based test."""
    import os
    path = os.path.join(root, "scripts", "check_gate_citations.py")
    backup = open(path).read()
    mutated = backup.replace(
        'bypass_candidates.append(REPO / ".lab" / "gate-bypass" / f"{bead_id}.json")',
        '# bypass_candidates.append(REPO / ".lab" / "gate-bypass" / f"{bead_id}.json")', 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

MUTATION_FUNCTIONS = {
    "ledger_node_ids_validated": _apply_ledger_bad_row,
    "wty2_structure_dominated": _apply_wty2_corrupt,
    "u5r4_size_scaling_arm": _apply_u5r4_corrupt,
    "s2qj_bypass_path": _apply_s2qj_unbypass,
    "v5l3_pin_coverage": _apply_v5l3_unpin,
    "9wza_tracked_bypass_records": _apply_9wza_disable_lab_fallback,
}

