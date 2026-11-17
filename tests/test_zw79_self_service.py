"""Guard: self-service grandfathering is refused (gpu-sim-zw79)."""
import importlib.util, json, subprocess, sys, tempfile
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


def _load(clone):
    spec = importlib.util.spec_from_file_location("cao_zw79", _checker_path(clone))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = Path(clone)
    return mod


def test_selfservice_refused():
    """Direct call: a branch adding a self-service grandfather row is refused."""
    import shutil
    with tempfile.TemporaryDirectory() as td:
        clone = _clone(td)
        shutil.copyfile(_checker_path(ROOT), _checker_path(clone))
        cao_path = Path(_checker_path(clone))
        src = cao_path.read_text()
        src = src.replace(
            '    "zsd3_janitor": "gpu-sim-zsd3: binds a product test, no mutation recorded",\n',
            '    "zsd3_janitor": "gpu-sim-zsd3: binds a product test, no mutation recorded",\n    "plant_selfservice": "gpu-sim-zw79: binds a product test, no mutation recorded",\n', 1)
        cao_path.write_text(src)
        row = clone / "tests" / "fs2y_ledger.d" / "plant_selfservice.json"
        row.parent.mkdir(parents=True, exist_ok=True)
        row.write_text(json.dumps({"name": "plant_selfservice", "beads": ["gpu-sim-zw79"],
            "state": "passing", "test": "test_1q34_lab_clean.test_lab_daemon_state_ignored", "bound_calls": 1,
            "mutation": "n/a", "verification": "positive_only"}))
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(clone), "commit", "-qm", "plant"], check=True)
        subprocess.run(["git", "-C", str(clone), "update-ref", "refs/heads/main", "origin/main"], check=True)
        mod = _load(clone)
        res = mod.check_grandfather_self_service()
        assert isinstance(res, list)
        assert len(res) == 1, (res,)
        assert "plant_selfservice" in res[0], (res,)
        assert "self-service grandfathering refused" in res[0], (res,)


def test_ordinary_rework_passes():
    """Direct call: ordinary rework (no grandfather edit) is not refused."""
    with tempfile.TemporaryDirectory() as td:
        clone = _clone(td)
        import shutil
        shutil.copyfile(_checker_path(ROOT), _checker_path(clone))
        (clone / "README.md").write_text("ordinary rework\n")
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(clone), "commit", "-qm", "ordinary"], check=True)
        mod = _load(clone)
        res = mod.check_grandfather_self_service()
        assert res == [], (res,)
        assert isinstance(res, list)
