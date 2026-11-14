"""Guard: _child_env scopes PYTHONDONTWRITEBYTECODE correctly (gpu-sim-qe7d)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / ("check_append" + "_only.py")


def _load():
    spec = importlib.util.spec_from_file_location("cao_qe7d", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mutated_run_suppresses_bytecode():
    cao = _load()
    assert cao._child_env(no_bytecode=True)["PYTHONDONTWRITEBYTECODE"] == "1"


def test_restore_run_does_not(monkeypatch):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    cao = _load()
    assert "PYTHONDONTWRITEBYTECODE" not in cao._child_env()
