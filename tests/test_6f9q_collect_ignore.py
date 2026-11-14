"""Guard: bare pytest collects cleanly (gpu-sim-6f9q)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_collect_ignore_glob_set():
    src = (ROOT / "conftest.py").read_text()
    assert 'collect_ignore_glob = ["experiments/*"]' in src
