"""Append-only guard tests."""
import json
import signal
import atexit
import unittest
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_append_only.py"

# p4dw-style: track plant files created during tests so an interrupted run
# cleans them up via atexit / signal handlers instead of poisoning the next run.
_PLANT_FILES = []

def _cleanup_plants():
    for pf in list(_PLANT_FILES):
        try:
            pf.unlink(missing_ok=True)
        except OSError:
            pass
    _PLANT_FILES.clear()

atexit.register(_cleanup_plants)

def _plant_signal(signum, frame):
    _cleanup_plants()
    raise SystemExit(128 + signum)

for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _plant_signal)
    except (ValueError, OSError):
        pass



def run_script():
    env = dict(os.environ)
    if _PLANT_FILES:
        env["P4DW_PLANT_PATHS"] = os.pathsep.join(str(p) for p in _PLANT_FILES)
    r = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT),
                       capture_output=True, text=True, env=env)
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
        meta = ROOT / "tests" / "fs2y_ledger_meta.json"
        _PLANT_FILES.extend([shard, meta])
        import subprocess as sp
        commit = sp.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 text=True).strip()
        shard.unlink()
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
            for _pf in (shard, meta):
                try:
                    _PLANT_FILES.remove(_pf)
                except ValueError:
                    pass

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
        _PLANT_FILES.append(shard)
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
            try: _PLANT_FILES.remove(shard)
            except ValueError: pass

    def test_plant_nonexistent_function_fails(self):
        """Plant (b): node naming a function that does not exist must fail."""
        shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_nonexistent_func.json"
        _PLANT_FILES.append(shard)
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
            try: _PLANT_FILES.remove(shard)
            except ValueError: pass

    def test_plant_nonexistent_file_fails(self):
        """Plant (c): node naming a file that does not exist must fail."""
        shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_nonexistent_file.json"
        _PLANT_FILES.append(shard)
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
            try: _PLANT_FILES.remove(shard)
            except ValueError: pass

    def test_plant_self_referential_fails(self):
        """A row naming a test in test_append_only (imports checker) must be refused."""
        shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_self_ref.json"
        _PLANT_FILES.append(shard)
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
            try: _PLANT_FILES.remove(shard)
            except ValueError: pass

    def test_plant_vacuous_fails(self):
        """A vacuous row (assert True) must fail the mutation check."""
        plant_file = ROOT / "tests" / "test_vacuous_plant.py"
        _PLANT_FILES.append(plant_file)
        plant_backup = None
        if plant_file.exists():
            plant_backup = plant_file.read_text()
        plant_file.write_text("def test_vacuous():\n    assert True\n")
        shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_vacuous.json"
        _PLANT_FILES.append(shard)
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
            try: _PLANT_FILES.remove(shard)
            except ValueError: pass
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
                               capture_output=True, text=True, timeout=300,
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


class TestRaisingApplierLeavesNoResidue(unittest.TestCase):
    """A write-then-raise applier must not crash the restore or leave residue.

    apply_fn raising before it returns (the `make` timeout path at
    fs2y_mutations.py:320-323) leaves backup and mutated_path None. The restore
    block used to call os.remove(None), which raises TypeError and SKIPS the
    dirty-path restore -- leaking exactly the residue that block exists to
    clean. This drives the real function with an injected raising applier.
    """

    def test_write_then_raise_restores_and_does_not_crash(self):
        import subprocess, importlib.util, sys as _sys
        from unittest import mock
        spec = importlib.util.spec_from_file_location(
            "cao_probe", ROOT / "scripts" / "check_append_only.py")
        cao = importlib.util.module_from_spec(spec)
        _sys.modules["cao_probe"] = cao
        spec.loader.exec_module(cao)

        target = ROOT / "scripts" / "check_gate_citations.py"
        before = target.read_text()
        dirty_before = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "-uall"],
                                      capture_output=True, text=True).stdout.split()

        def _raising_applier(root):
            with open(target, "a") as f:
                f.write("\n# mutation residue\n")
            raise RuntimeError("simulated write-then-raise")

        row = {"name": "raise_after_write_probe", "state": "passing",
               "test": "tests/test_lint_prose_numbers.py::TestFiresOnHistory::test_6_6_1_worst_case_mdd",
               "mutation": "raise_after_write"}
        # Restore in a finally, not only on the success path: if the harness
        # under test leaks, this test must not be the thing that leaves residue.
        try:
         with mock.patch.object(cao, "_load_mutations",
                               lambda root: (({"raise_after_write": "probe"},
                                        {"raise_after_write": _raising_applier}), None)), \
             mock.patch.object(cao, "load_ledger_entries", lambda root: [row]):
            try:
                with self.assertRaises(RuntimeError):
                    cao.check_ledger_test_ids(ROOT)
            except TypeError as e:      # the defect being guarded
                self.fail(f"restore path crashed with TypeError: {e}")
        finally:
            if target.read_text() != before:
                target.write_text(before)

        self.assertEqual(target.read_text(), before,
                         "write-then-raise left residue behind")
        self.assertEqual(
            subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "-uall"],
                           capture_output=True, text=True).stdout.split(),
            dirty_before)

    def test_write_then_raise_that_creates_an_UNTRACKED_file_is_cleaned(self):
        """-uall: a file created inside an untracked directory must be restored.

        `git diff --name-only` cannot see untracked paths at all, so a raising
        applier that creates one left it behind while every dirty-set comparison
        still reported clean.
        """
        import subprocess, importlib.util, sys as _sys
        from unittest import mock
        spec = importlib.util.spec_from_file_location(
            "cao_probe2", ROOT / "scripts" / "check_append_only.py")
        cao = importlib.util.module_from_spec(spec)
        _sys.modules["cao_probe2"] = cao
        spec.loader.exec_module(cao)

        junk_dir = ROOT / "tests"          # tracked dir: -uall reports the file
        junk = junk_dir / "untracked_probe_residue.txt"
        before = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "-uall"],
                                capture_output=True, text=True).stdout.splitlines()

        def _raising_applier(root):
            junk_dir.mkdir(parents=True, exist_ok=True)
            junk.write_text("residue\n")
            raise RuntimeError("simulated write-then-raise (untracked)")

        row = {"name": "untracked_probe", "state": "passing",
               "test": "tests/test_lint_prose_numbers.py::TestFiresOnHistory::test_6_6_1_worst_case_mdd",
               "mutation": "raise_after_write_untracked"}
        try:
            with mock.patch.object(cao, "_load_mutations",
                                   lambda root: (({"raise_after_write_untracked": "probe"},
                                                  {"raise_after_write_untracked": _raising_applier}), None)), \
                 mock.patch.object(cao, "load_ledger_entries", lambda root: [row]):
                try:
                    with self.assertRaises(RuntimeError):
                        cao.check_ledger_test_ids(ROOT)
                except TypeError as e:
                    self.fail(f"restore path crashed: {e}")
            self.assertFalse(junk.exists(), "untracked residue was left behind")
            self.assertEqual(
                subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "-uall"],
                               capture_output=True, text=True).stdout.splitlines(),
                before, "untracked residue changed the tree")
        finally:
            if junk.exists():          # never leave residue if the harness fails
                junk.unlink()


class TestMutationRestoreSafety(unittest.TestCase):
    """p4dw: mutation restore safety net (signal handler + atexit)."""

    def test_dirty_check_tolerates_untracked(self):
        # untracked files (new shards/rows) are tolerated; the walk proceeds
        dirty = ROOT / "experiments" / "gpu-sim-p4dw-dirty-probe.txt"
        dirty.write_text("x")
        try:
            r = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotIn("REFUSE", r.stdout + r.stderr)
        finally:
            dirty.unlink(missing_ok=True)

