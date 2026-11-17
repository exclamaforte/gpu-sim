"""Guard: self-service grandfathering is refused (gpu-sim-zw79)."""
import importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = "check_" + "append_only.py"


def _checker_path(root):
    return str(Path(root) / "scripts" / CHECKER)


def _clone(tmp):
    clone = Path(tmp) / "clone"
    subprocess.run(["git", "clone", "-q", str(ROOT), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(clone), "branch", "-f", "main", "origin/main"], capture_output=True)
    return clone


def _run_checker(clone):
    r = subprocess.run([sys.executable, _checker_path(clone), "--dirty-ok"],
                       cwd=str(clone), capture_output=True, text=True, timeout=300)
    return r


def test_selfservice_refused():
    """End-to-end: a branch adding a self-service grandfather row is refused."""
    with tempfile.TemporaryDirectory() as td:
        clone = _clone(td)
        import shutil
        shutil.copyfile(_checker_path(ROOT), _checker_path(clone))
        cao_path = Path(_checker_path(clone))
        src = cao_path.read_text()
        src = src.replace(
            '    "zsd3_janitor": "gpu-sim-zsd3: binds a product test, no mutation recorded",\n',
            '    "zsd3_janitor": "gpu-sim-zsd3: binds a product test, no mutation recorded",\n    "plant_selfservice": "gpu-sim-zw79: binds a product test, no mutation recorded",\n', 1)
        cao_path.write_text(src)
        row = clone / "tests" / "fs2y_ledger.d" / "plant_selfservice.json"
        row.write_text(json.dumps({"name": "plant_selfservice", "beads": ["gpu-sim-zw79"],
            "state": "passing", "test": "test_1q34_lab_clean.test_lab_daemon_state_ignored", "bound_calls": 1,
            "mutation": "n/a", "verification": "positive_only"}))
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(clone), "commit", "-qm", "plant"], check=True)
        r = _run_checker(clone)
        assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)
        assert "plant_selfservice" in (r.stdout + r.stderr), (r.stdout, r.stderr)


def test_ordinary_rework_passes():
    """Direct call: ordinary rework (no grandfather edit) is not refused."""
    import importlib.util
    with tempfile.TemporaryDirectory() as td:
        clone = _clone(td)
        import shutil
        shutil.copyfile(_checker_path(ROOT), _checker_path(clone))
        (clone / "README.md").write_text("ordinary rework\n")
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(clone), "commit", "-qm", "ordinary"], check=True)
        spec = importlib.util.spec_from_file_location("cao_zw79_ord", _checker_path(clone))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.ROOT = Path(clone)
        res = mod.check_grandfather_self_service()
        assert res == [], (res,)
        assert isinstance(res, list)
