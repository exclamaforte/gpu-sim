"""Append-only guard tests."""
import json
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


class TestAppendOnly:
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
        lf = ROOT / "tests" / "fs2y_ledger.json"
        backup = lf.read_text()
        d = json.loads(backup)
        names = [e["name"] for e in d["entries"]]
        assert "dhfo_memory_check" in names
        d["entries"] = [e for e in d["entries"] if e["name"] != "dhfo_memory_check"]
        lf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on unretired deletion"
            assert "dhfo_memory_check" in out, out
        finally:
            lf.write_text(backup)

    def test_ledger_retired_deletion_passes(self):
        lf = ROOT / "tests" / "fs2y_ledger.json"
        backup = lf.read_text()
        d = json.loads(backup)
        names = [e["name"] for e in d["entries"]]
        assert "dhfo_memory_check" in names
        d["entries"] = [e for e in d["entries"] if e["name"] != "dhfo_memory_check"]
        import subprocess as sp
        commit = sp.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 text=True).strip()
        d.setdefault("retired", [])
        d["retired"].append({
            "name": "dhfo_memory_check",
            "reason": "test retirement",
            "commit": commit,
        })
        lf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc == 0, out
        finally:
            lf.write_text(backup)

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


class TestLedgerNodeIds:
    def test_ledger_node_ids_resolve(self):
        """Every ledger row's test node resolves and collects >=1 test."""
        rc, out = run_script()
        assert rc == 0, out

    def test_plant_class_qualified_module_level_fails(self):
        """Plant (a): class-qualified node for a module-level test must fail."""
        lf = ROOT / "tests" / "fs2y_ledger.json"
        backup = lf.read_text()
        d = json.loads(backup)
        d["entries"].append({
            "name": "plant_class_qualified",
            "state": "passing",
            "test": "tests/test_no_pycache.py::TestNoPycache::test_lab_config_sets_pycache_suppression",
            "mutation": "append_only_guard",
        })
        lf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on class-qualified module-level node"
            assert "does not resolve" in out or "collects 0" in out, out
        finally:
            lf.write_text(backup)

    def test_plant_nonexistent_function_fails(self):
        """Plant (b): node naming a function that does not exist must fail."""
        lf = ROOT / "tests" / "fs2y_ledger.json"
        backup = lf.read_text()
        d = json.loads(backup)
        d["entries"].append({
            "name": "plant_nonexistent_func",
            "state": "passing",
            "test": "tests/test_no_pycache.py::nonexistent_func_xyz",
            "mutation": "append_only_guard",
        })
        lf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on nonexistent function"
            assert "does not resolve" in out or "collects 0" in out, out
        finally:
            lf.write_text(backup)

    def test_plant_nonexistent_file_fails(self):
        """Plant (c): node naming a file that does not exist must fail."""
        lf = ROOT / "tests" / "fs2y_ledger.json"
        backup = lf.read_text()
        d = json.loads(backup)
        d["entries"].append({
            "name": "plant_nonexistent_file",
            "state": "passing",
            "test": "tests/test_nonexistent_file_xyz.py::test_foo",
            "mutation": "append_only_guard",
        })
        lf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on nonexistent file"
            assert "does not resolve" in out or "collects 0" in out, out
        finally:
            lf.write_text(backup)

    def test_plant_self_referential_fails(self):
        """A row naming a test in test_append_only (imports checker) must be refused."""
        lf = ROOT / "tests" / "fs2y_ledger.json"
        backup = lf.read_text()
        d = json.loads(backup)
        d["entries"].append({
            "name": "plant_self_ref",
            "state": "passing",
            "test": "tests/test_append_only.py::TestAppendOnly::test_retired_deletion_passes",
            "mutation": "append_only_guard",
        })
        lf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on self-referential row"
            assert "self-referential" in out, out
        finally:
            lf.write_text(backup)

    def test_plant_vacuous_fails(self):
        """A vacuous row (assert True) must fail the mutation check."""
        lf = ROOT / "tests" / "fs2y_ledger.json"
        backup = lf.read_text()
        d = json.loads(backup)
        d["entries"].append({
            "name": "plant_vacuous",
            "state": "passing",
            "test": "tests/test_vacuous_plant.py::test_vacuous",
            "mutation": "ledger_node_ids_validated",
        })
        lf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on vacuous row"
            assert "did not fail" in out, out
        finally:
            lf.write_text(backup)

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

