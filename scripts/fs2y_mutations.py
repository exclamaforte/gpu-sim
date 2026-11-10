"""Mutation registry for the fs2y ledger.

Maps each non-pending ledger mutation name to a short description of intent.
The lab presubmit (x06h) imports this module and checks that every non-pending
ledger row's mutation resolves against this dict; a row whose mutation is absent
is reported as newly unresolved. Descriptions document intent only.
"""
MUTATIONS = {
    "fd5i_landing_commit_full_sha": "fd5i landing_commit_for compares full shas (gpu-sim-fd5i)",
    "3v6d1_fabric_rate": "3v6d1 one-rate fabric: bidirectional halving in topology.effective_bw (gpu-sim-3v6d.1)",

    "5uu9_power_arm": "5uu9 power arm reproducibility",
    "55sb_na_rule_disabled": "55sb: disable the 'n/a' refusal so a positive-only row passes again",
    "20ms_drop_grandfather": "20ms: remove wknt from the grandfather list so the row is refused again",
    "qaiu_power_audit": "qaiu power audit reproducibility",
    "8yb4_k_window": "8yb4 K-window: a measured ratio inflated to the 40x currency error the bead rules out",
    "3v6d2_link_calendar": "3v6d.2 per-link calendar: CAS replaced by plain store",
    "rb66_ambient_init": "rb66 fixture: throwaway repo left on the ambient default branch",
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
    "05b3_skip_allowlist_keys": "skip allowlist keyed by file::function:TAG:condition (gpu-sim-05b3)",
    "u5q0_lint_fixture_6338": "6.10 prose-ratio fixture pins the derived 6338x ratio (gpu-sim-u5q0)",
    "s2qj_bypass_path": "s2qj bypass path (gpu-sim-s2qj)",
    "wty2_structure_dominated": "wty2 structure-dominated verdict on unseen families (gpu-sim-wty2)",
    "u5r4_size_scaling_arm": "u5r4 size-scaling arm on divergent_branch (gpu-sim-u5r4)",
    "j1qc_lint_no_bd": "j1qc lint takes comment text as argument, no bd dependency (gpu-sim-j1qc)",
    "v5l3_pin_coverage": "v5l3 pin coverage lint (gpu-sim-v5l3)",
    "9wza_tracked_bypass_records": "9wza tracked-first bypass records with divergence check (gpu-sim-9wza)",
    "fd5i_landing_commit_full_sha": "fd5i landing_commit_for compares full shas (gpu-sim-fd5i)",
    "3v6d3_nccl_enforce": "3v6d.3 ncclAllReduce enforces through fabric calendar (gpu-sim-3v6d.3)",
    "hsuv_manifest_relative": "hsuv manifest relative paths (gpu-sim-hsuv)",
    "3v6d1_fabric_rate": "3v6d1 one-rate fabric: bidirectional halving in topology.effective_bw (gpu-sim-3v6d.1)",
    "nlgf_power_test": "nlgf 3-replicate power test (v1 single-launch target)",
    "e7yw_trunk_resolution": "e7yw trunk resolution prefers main (gpu-sim-e7yw)",
    "trcz_impostor": "trcz impostor probe hardening (gpu-sim-trcz)",
    "ccw5_meta_parity": "ccw5 META/META_KEYS equality (gpu-sim-ccw5)",
    "wknt_error_budget_bounded": "wknt error budget bounded (gpu-sim-wknt)",
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

def _apply_j1qc_lint_no_bd(root):
    """Mutation for j1qc_lint_no_bd: make check_6_6_1_worst_case_mdd a no-op."""
    import os
    path = os.path.join(root, "experiments", "gpu-sim-u2o.6.11.1", "lint_prose_numbers.py")
    backup = open(path).read()
    mutated = backup.replace(
        "def check_6_6_1_worst_case_mdd(findings: list, comments: str | None = None) -> None:",
        "def check_6_6_1_worst_case_mdd(findings: list, comments: str | None = None) -> None:\n    return", 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

def _apply_55sb_na_rule_disabled(root):
    """Mutation for 55sb_positive_only: let mutation 'n/a' through again.

    Disables exactly the rule the ledger row claims to bind, so the bound test
    must FAIL while the plant is applied and pass once it is restored.
    """
    import os
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    mutated = backup.replace(
        'if mutation == "n/a" and not exempt and not grandfathered:',
        'if False and mutation == "n/a" and not exempt and not grandfathered:', 1)
    if mutated == backup:
        raise RuntimeError("55sb mutation anchor not found: the 'n/a' rule moved")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_20ms_drop_grandfather(root):
    """Mutation for 20ms_wknt_grandfather: un-grandfather the wknt row.

    Removing the entry makes the bound test FAIL (the row is refused again),
    which is the only plant that falsifies the claim this bead makes.
    """
    import os
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    marker = '    "wknt_error_budget_bounded": '
    lines = [ln for ln in backup.splitlines(True) if not ln.startswith(marker)]
    mutated = "".join(lines)
    if mutated == backup:
        raise RuntimeError("20ms mutation anchor not found: wknt grandfather entry moved")
    with open(path, "w") as f:
        f.write(mutated)
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

def _apply_fd5i_revert_to_short(root):
    """Mutation for fd5i_landing_commit_full_sha: revert landing_commit_for to %h so the full-sha agreement plant fails."""
    import os
    path = os.path.join(root, "scripts", "tree_state.py")
    backup = open(path).read()
    mutated = backup.replace('log", "--format=%H", "--diff-filter=A"', 'log", "--format=%h", "--diff-filter=A"', 1) + "\n"
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_nlgf_corrupt(root):
    """Mutation for nlgf_power_test: corrupt a scalar in committed metrics_power.json."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-nlgf", "metrics_power.json")
    backup = open(path).read()
    d = json.loads(backup)
    try:
        d["n_cells_total"] = 999
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path


def _apply_3v6d1_unhalve(root):
    """Mutation for 3v6d1_fabric_rate: remove the bidirectional halving in
    topology.effective_bw, so the one-way rate becomes the spec's aggregate
    900 GB/s. The Python rule, the emitted binary and the transport then disagree
    with the model formula in the same way they did before gpu-sim-3v6d.1.

    Raises if the substitution does not apply, so a silent no-op cannot be
    reported as a mutation that failed to fail.
    """
    import os
    path = os.path.join(root, "epic_cluster", "topology.py")
    backup = open(path).read()
    old = 'if link.get("bw_is_bidirectional", False):\n        bw *= 0.5\n'
    # MEASURED 2026-11-04: the mutated text must differ in LENGTH from the
    # original. conftest.py redirects bytecode to .lab/tmp/pycache and CPython
    # validates a .pyc on (mtime, size); a same-length mutation restored inside
    # the same mtime second leaves a stale .pyc that serves the MUTATED code to
    # the restored run, so the checker reports 'restored target did not pass'.
    # The trailing comment makes the mutated file longer, so the signature
    # differs in both directions.
    new = ('if link.get("bw_is_bidirectional", False):\n'
           '        bw *= 1.0  # 3v6d1 mutation: halving removed\n')
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("3v6d1 mutation did not apply: halving block not found "
                           "in epic_cluster/topology.py")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_rb66_ambient_init(root):
    """Mutation for rb66_ambient_init: init the throwaway repo on the WRONG branch.

    The real fix is that the fixture names its branch explicitly, so the
    master/side hazard it builds does not depend on the caller's
    init.defaultBranch. Pointing init at 'main' instead reproduces the bug
    deterministically: the following 'git checkout master' has no such branch and
    the eight tests in the file error out.
    """
    import os
    path = os.path.join(root, "tests", "test_safe_git_add.py")
    backup = open(path).read()
    old = 'git(self.repo, "init", "-q", "-b", "master")'
    new = 'git(self.repo, "init", "-q", "-b", "main")'
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("rb66 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path



def _apply_05b3_delete_key(root):
    """Mutation for 05b3_skip_allowlist_keys: delete one registered key."""
    import json, os
    path = os.path.join(root, "tests", "skip_allowlist.json")
    backup = open(path).read()
    d = json.loads(backup)
    # remove first key
    if d:
        del d[next(iter(d))]
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    return backup, path


def _apply_3v6d2_uncas(root):
    """Mutation for 3v6d2_link_calendar: remove the calendar reservation.

    Targets the ONE semantic line (the reservation start) rather than the whole
    CAS block: the block-splice broke twice today -- once when the fabric was
    refactored, once when a comment and an out-param assignment were added around
    it -- and each break surfaced as "mutation did not apply", which is the
    ledger doing its job but costing a gate cycle. Making the anchor minimal keeps
    the mutation about the BEHAVIOUR (transfers no longer queue) instead of about
    the surrounding text.
    """
    import os
    path = os.path.join(root, "src", "fakegpu_fabric.c")
    backup = open(path).read()
    old = "            start = (busy > 0) ? busy : now;"
    new = "            start = now;  /* mutation: calendar removed -- no serialization */"
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("3v6d2 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    import subprocess
    subprocess.run(["make"], cwd=root, capture_output=True, text=True, timeout=300)
    return backup, path


def _apply_8yb4_corrupt(root):
    """Mutation for 8yb4_k_window: put back the 40x currency error the bead rules out.

    The row's test pins every config's K=1/K=40 ratio to [0.9,1.1]; the defect under
    investigation was the hypothesis that the corpus's one-launch-per-window timing
    and the gate's K=40 pricing are ~40x apart. Inflating one measured ratio to 40
    reproduces that hypothesis as data, so the test must fail.
    """
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-8yb4", "rows.jsonl")
    backup = open(path).read()
    lines = backup.splitlines()
    out = []
    changed = False
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if not changed:
            row["ratio_k1_over_k40"] = 40.0
            changed = True
        out.append(json.dumps(row))
    if not changed:
        raise RuntimeError("8yb4 mutation did not apply")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return backup, path


def _apply_hsuv_manifest_absolute(root):
    """Mutation for hsuv_manifest_relative: write an absolute exe path into the
    committed manifest so the relative-path guard fails."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-u5r4", "build_div_sizescale", "manifest.json")
    backup = open(path).read()
    d = json.loads(backup)
    specs = d.get("specs", [])
    if specs:
        specs[0]["exe"] = "/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-hsuv/experiments/gpu-sim-u5r4/build_div_sizescale/bin/k_divergent_branch_0000.out"
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path


def _apply_3v6d3_nccl_enforce(root):
    """Mutation for 3v6d3_nccl_enforce: revert enforced bw to raw 900 Gbps (8x delta)."""
    import os
    path = os.path.join(root, "src", "fakegpu_nccl.c")
    backup = open(path).read()
    mutated = backup
    mutated = mutated.replace("bw = t->links[bl].eff_bw_gbs;", "bw = 900.0; /* mutation: raw bw */", 1)
    mutated = mutated.replace("bw = t->links[0].eff_bw_gbs;", "bw = 900.0; /* mutation: raw bw */", 1)
    if mutated == backup:
        raise RuntimeError("3v6d3 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    import subprocess
    subprocess.run(["make"], cwd=root, capture_output=True, text=True, timeout=300)
    return backup, path


def _apply_5uu9_corrupt(root):
    """Mutation for 5uu9_power_arm: corrupt a scalar in committed metrics.json."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-5uu9", "metrics.json")
    backup = open(path).read()
    d = json.loads(backup)
    try:
        d["gbt_mae"] = 999.0
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path


def _apply_qaiu_corrupt(root):
    """Mutation for qaiu_power_audit: corrupt a scalar in committed metrics.json."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-qaiu", "metrics.json")
    backup = open(path).read()
    d = json.loads(backup)
    try:
        d["as_landed_over_rows"]["paired_delta_mean"] = 999.0
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path


def _apply_fd5i_revert_to_short(root):
    """Mutation for fd5i_landing_commit_full_sha: revert landing_commit_for to %h so the full-sha agreement plant fails."""
    import os
    path = os.path.join(root, "scripts", "tree_state.py")
    backup = open(path).read()
    mutated = backup.replace('log", "--format=%H", "--diff-filter=A"', 'log", "--format=%h", "--diff-filter=A"', 1) + "\n"
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

def _apply_3v6d1_unhalve(root):
    """Mutation for 3v6d1_fabric_rate: remove the bidirectional halving in
    topology.effective_bw, so the one-way rate becomes the spec's aggregate
    900 GB/s. The Python rule, the emitted binary and the transport then disagree
    with the model formula in the same way they did before gpu-sim-3v6d.1.

    Raises if the substitution does not apply, so a silent no-op cannot be
    reported as a mutation that failed to fail.
    """
    import os
    path = os.path.join(root, "epic_cluster", "topology.py")
    backup = open(path).read()
    old = 'if link.get("bw_is_bidirectional", False):\n        bw *= 0.5\n'
    # MEASURED 2026-11-04: the mutated text must differ in LENGTH from the
    # original. conftest.py redirects bytecode to .lab/tmp/pycache and CPython
    # validates a .pyc on (mtime, size); a same-length mutation restored inside
    # the same mtime second leaves a stale .pyc that serves the MUTATED code to
    # the restored run, so the checker reports 'restored target did not pass'.
    # The trailing comment makes the mutated file longer, so the signature
    # differs in both directions.
    new = ('if link.get("bw_is_bidirectional", False):\n'
           '        bw *= 1.0  # 3v6d1 mutation: halving removed\n')
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("3v6d1 mutation did not apply: halving block not found "
                           "in epic_cluster/topology.py")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_e7yw_resolve_trunk_master(root):
    """Mutation for e7yw_trunk_resolution: make resolve_trunk return None."""
    import os
    path = os.path.join(root, "scripts", "resolve_trunk.py")
    backup = open(path).read()
    mutated = backup.replace('            return ref', '            return None  # mutated', 1)
    if mutated == backup:
        raise RuntimeError("e7yw mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_trcz_remove_assertion(root):
    """Mutation for trcz_impostor: remove the _dirty_at_start/marker assertion."""
    import os
    path = os.path.join(root, "tests", "test_zsd3_janitor_detection.py")
    backup = open(path).read()
    mutated = backup.replace(
        "    assert pytestconfig._dirty_at_start is not None, 'conftest did not run'\n",
        '', 1)
    mutated = mutated.replace(
        '    assert marker.exists(), "probe did not execute (impostor detection):\\n" + out[-1000:]\n',
        '', 1)
    if mutated == backup:
        raise RuntimeError("trcz mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_ccw5_meta_mismatch(root):
    """Mutation for ccw5_meta_parity: remove a key from META in run.sh."""
    import os
    path = os.path.join(root, "experiments", "gpu-sim-5uu9", "run.sh")
    backup = open(path).read()
    mutated = backup.replace('META = ("loadavg_1_5_15", "foreign_gpu_job", "gate")',
                            'META = ("loadavg_1_5_15", "foreign_gpu_job")', 1)
    if mutated == backup:
        raise RuntimeError("ccw5 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_wknt_remove_alarm(root):
    """Mutation for wknt_error_budget_bounded: corrupt the sort key."""
    import os
    path = os.path.join(root, "scripts", "error_budget.py")
    backup = open(path).read()
    mutated = backup.replace('        f = r["fraction_one_sided"]',
                            '        f = r["flip_down"]', 1)
    if mutated == backup:
        raise RuntimeError("wknt mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


def _apply_no_pycache_under_experiments(root):
    """Mutation for no_pycache_under_experiments: drop PYTHONPYCACHEPREFIX from lab.config.json."""
    import json, os
    path = os.path.join(root, "lab.config.json")
    backup = open(path).read()
    d = json.loads(backup)
    cmd = d.get("tests", {}).get("command", [])
    new_cmd = [x for x in cmd if "PYTHONPYCACHEPREFIX" not in str(x)]
    d["tests"]["command"] = new_cmd
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    return backup, path


MUTATION_FUNCTIONS = {
    "fd5i_landing_commit_full_sha": _apply_fd5i_revert_to_short,
    "3v6d1_fabric_rate": _apply_3v6d1_unhalve,
    "5uu9_power_arm": _apply_5uu9_corrupt,
    "qaiu_power_audit": _apply_qaiu_corrupt,
    "8yb4_k_window": _apply_8yb4_corrupt,
    "3v6d2_link_calendar": _apply_3v6d2_uncas,
    "rb66_ambient_init": _apply_rb66_ambient_init,
    "ledger_node_ids_validated": _apply_ledger_bad_row,
    "wty2_structure_dominated": _apply_wty2_corrupt,
    "u5r4_size_scaling_arm": _apply_u5r4_corrupt,
    "j1qc_lint_no_bd": _apply_j1qc_lint_no_bd,
    "s2qj_bypass_path": _apply_s2qj_unbypass,
    "v5l3_pin_coverage": _apply_v5l3_unpin,
    "9wza_tracked_bypass_records": _apply_9wza_disable_lab_fallback,
    "fd5i_landing_commit_full_sha": _apply_fd5i_revert_to_short,
    "3v6d1_fabric_rate": _apply_3v6d1_unhalve,
    "nlgf_power_test": _apply_nlgf_corrupt,
    "3v6d3_nccl_enforce": _apply_3v6d3_nccl_enforce,
    "hsuv_manifest_relative": _apply_hsuv_manifest_absolute,
    "05b3_skip_allowlist_keys": _apply_05b3_delete_key,
    "no_pycache_under_experiments": _apply_no_pycache_under_experiments,
    "55sb_na_rule_disabled": _apply_55sb_na_rule_disabled,
    "20ms_drop_grandfather": _apply_20ms_drop_grandfather,
    "e7yw_trunk_resolution": _apply_e7yw_resolve_trunk_master,
    "trcz_impostor": _apply_trcz_remove_assertion,
    "ccw5_meta_parity": _apply_ccw5_meta_mismatch,
    "wknt_error_budget_bounded": _apply_wknt_remove_alarm,
}
