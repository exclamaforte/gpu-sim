"""Append-only guard tests."""
import json
import unittest
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_append_only.py"


def run_script():
    r = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT),
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


class TestAppendOnly(unittest.TestCase):
    def test_unretired_deletion_fails_naming_key(self):
        mf = ROOT / "docs" / "model_facts.json"
        backup = mf.read_text()
        d = json.loads(backup)
        assert "fabric_band_robustness" in d
        del d["fabric_band_robustness"]
        mf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on unretired deletion"
            assert "fabric_band_robustness" in out, out
        finally:
            mf.write_text(backup)

    def test_retired_deletion_passes(self):
        mf = ROOT / "docs" / "model_facts.json"
        backup = mf.read_text()
        d = json.loads(backup)
        assert "fabric_band_robustness" in d
        del d["fabric_band_robustness"]
        import subprocess as sp
        commit = sp.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 text=True).strip()
        d.setdefault("_retired_fact_blocks", [])
        d["_retired_fact_blocks"].append({
            "key": "fabric_band_robustness",
            "reason": "test retirement",
            "commit": commit,
        })
        mf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc == 0, out
        finally:
            mf.write_text(backup)

    def test_ledger_unretired_deletion_fails_naming_entry(self):
        shard = ROOT / "tests" / "fs2y_ledger.d" / "dhfo_memory_check.json"
        backup = shard.read_text()
        shard.unlink()
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on unretired deletion"
            assert "dhfo_memory_check" in out, out
        finally:
            shard.write_text(backup)

    def test_ledger_retired_deletion_passes(self):
        shard = ROOT / "tests" / "fs2y_ledger.d" / "dhfo_memory_check.json"
        backup = shard.read_text()
        import subprocess as sp
        commit = sp.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 text=True).strip()
        shard.unlink()
        meta = ROOT / "tests" / "fs2y_ledger_meta.json"
        meta_backup = meta.read_text()
        md = json.loads(meta_backup)
        md.setdefault("retired", [])
        md["retired"].append({
            "name": "dhfo_memory_check",
            "reason": "test retirement",
            "commit": commit,
        })
        meta.write_text(json.dumps(md, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc == 0, out
        finally:
            shard.write_text(backup)
            meta.write_text(meta_backup)

    def test_retirement_missing_reason_fails(self):
        mf = ROOT / "docs" / "model_facts.json"
        backup = mf.read_text()
        d = json.loads(backup)
        assert "fabric_band_robustness" in d
        del d["fabric_band_robustness"]
        import subprocess as sp
        commit = sp.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 text=True).strip()
        d.setdefault("_retired_fact_blocks", [])
        d["_retired_fact_blocks"].append({
            "key": "fabric_band_robustness",
            "commit": commit,
        })
        mf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on missing reason"
        finally:
            mf.write_text(backup)



    def test_no_legacy_ledger_json_reappears(self):
        """Regression (gpu-sim-u6ih): the legacy single-file ledger must not reappear."""
        lf = ROOT / "tests" / "fs2y_ledger.json"
        assert not lf.exists(), (
            "tests/fs2y_ledger.json must not reappear (gpu-sim-u6ih): "
            "entries live in tests/fs2y_ledger.d/*.json now")

class TestLedgerNodeIds(unittest.TestCase):
    def test_ledger_node_ids_resolve(self):
        """Every ledger row's test node resolves and collects >=1 test."""
        rc, out = run_script()
        assert rc == 0, out

    def test_plant_class_qualified_module_level_fails(self):
        """Plant (a): class-qualified node for a module-level test must fail."""
        shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_class_qualified.json"
        shard.write_text(json.dumps({
            "name": "plant_class_qualified",
            "beads": ["gpu-sim-u6ih"],
            "state": "passing",
            "test": "tests/test_no_pycache.py::NonexistentClass::test_lab_config_sets_pycache_suppression",
            "bound_calls": 1,
            "mutation": "append_only_guard",
        }, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on class-qualified module-level node"
            assert "does not resolve" in out or "collects 0" in out, out
        finally:
            shard.unlink(missing_ok=True)

    def test_plant_nonexistent_function_fails(self):
        """Plant (b): node naming a function that does not exist must fail."""
        shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_nonexistent_func.json"
        shard.write_text(json.dumps({
            "name": "plant_nonexistent_func",
            "beads": ["gpu-sim-u6ih"],
            "state": "passing",
            "test": "tests/test_no_pycache.py::nonexistent_func_xyz",
            "bound_calls": 1,
            "mutation": "append_only_guard",
        }, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on nonexistent function"
            assert "does not resolve" in out or "collects 0" in out, out
        finally:
            shard.unlink(missing_ok=True)

    def test_plant_nonexistent_file_fails(self):
        """Plant (c): node naming a file that does not exist must fail."""
        shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_nonexistent_file.json"
        shard.write_text(json.dumps({
            "name": "plant_nonexistent_file",
            "beads": ["gpu-sim-u6ih"],
            "state": "passing",
            "test": "tests/test_nonexistent_file_xyz.py::test_foo",
            "bound_calls": 1,
            "mutation": "append_only_guard",
        }, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on nonexistent file"
            assert "does not resolve" in out or "collects 0" in out, out
        finally:
            shard.unlink(missing_ok=True)

    def test_plant_self_referential_fails(self):
        """A row naming a test in test_append_only (imports checker) must be refused."""
        shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_self_ref.json"
        shard.write_text(json.dumps({
            "name": "plant_self_ref",
            "beads": ["gpu-sim-u6ih"],
            "state": "passing",
            "test": "tests/test_append_only.py::TestAppendOnly::test_retired_deletion_passes",
            "bound_calls": 1,
            "mutation": "append_only_guard",
        }, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on self-referential row"
            assert "self-referential" in out, out
        finally:
            shard.unlink(missing_ok=True)

    def test_plant_vacuous_fails(self):
        """A vacuous row (assert True) must fail the mutation check."""
        plant_file = ROOT / "tests" / "test_vacuous_plant.py"
        plant_backup = None
        if plant_file.exists():
            plant_backup = plant_file.read_text()
        plant_file.write_text("def test_vacuous():\n    assert True\n")
        shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_vacuous.json"
        shard.write_text(json.dumps({
            "name": "plant_vacuous",
            "beads": ["gpu-sim-u6ih"],
            "state": "passing",
            "test": "tests/test_vacuous_plant.py::test_vacuous",
            "bound_calls": 1,
            "mutation": "ledger_node_ids_validated",
        }, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on vacuous row"
            assert "did not fail" in out, out
        finally:
            shard.unlink(missing_ok=True)
            if plant_backup is None:
                plant_file.unlink(missing_ok=True)
            else:
                plant_file.write_text(plant_backup)

    def test_spoofed_depth_marker_does_not_skip(self):
        """An ambient/spoofed OI4M_CHECKER_PID must not enable child/skip mode."""
        import os
        env = dict(os.environ)
        env["OI4M_CHECKER_PID"] = "99999"
        r = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT),
                           capture_output=True, text=True, env=env)
        # baseline ledger passes; rc must still be 0 (full run), proving the
        # spoofed marker did not turn the check into a no-op. The marker is
        # rejected because no ancestor matches.
        assert r.returncode == 0, r.stdout + r.stderr


import json as _json


class TestWty2Repro(unittest.TestCase):
    """Plants for the wty2 reproducibility gate (must exit nonzero)."""

    def _corrupt(self, mutate):
        import tempfile
        committed = ROOT / "experiments" / "gpu-sim-wty2" / "metrics.json"
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(committed.read_text())
        tmp.close()
        d = _json.loads(open(tmp.name).read())
        mutate(d)
        open(tmp.name, "w").write(_json.dumps(d, indent=2) + "\n")
        run = ROOT / "experiments" / "gpu-sim-wty2" / "run.sh"
        env = dict(os.environ)
        env["WTY2_METRICS"] = tmp.name
        try:
            r = subprocess.run(["bash", str(run)], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=120,
                               env=env)
            return r.returncode, r.stdout + r.stderr
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def test_corrupted_scalar_fails(self):
        def mutate(d):
            d["ac1_reconstruction"]["per_family"]["strided_read"]["gbt_mae"] = 999.0
        rc, out = self._corrupt(mutate)
        self.assertNotEqual(rc, 0, "expected failure on corrupted scalar")
        self.assertIn("NOT reproducible", out, out)

    def test_reordered_list_fails(self):
        def mutate(d):
            fams = list(d["ac1_reconstruction"]["per_family"].keys())
            # move first key to end by rebuilding dict
            first = fams[0]
            val = d["ac1_reconstruction"]["per_family"].pop(first)
            d["ac1_reconstruction"]["per_family"][first] = val
        rc, out = self._corrupt(mutate)
        self.assertNotEqual(rc, 0, "expected failure on reordered list")
        self.assertIn("NOT reproducible", out, out)

    def test_flipped_verdict_fails(self):
        def mutate(d):
            d["verdict"] = "EXTRAPOLATION_DOMINATED"
        rc, out = self._corrupt(mutate)
        self.assertNotEqual(rc, 0, "expected failure on flipped verdict")
        self.assertIn("NOT reproducible", out, out)
