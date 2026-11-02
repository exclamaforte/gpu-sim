"""Ensure test invocation prevents bytecode writes under experiments/."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lab_config_sets_pycache_suppression():
    cfg = json.loads((ROOT / "lab.config.json").read_text())
    cmd = cfg["tests"]["command"]
    assert any("PYTHONDONTWRITEBYTECODE" in str(x) for x in cmd) or "-B" in cmd
