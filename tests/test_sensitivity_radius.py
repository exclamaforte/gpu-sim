"""Robustness radius pinned test (gpu-sim-zj9a)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from epic_cluster import parallelism_search as ps

FACTS = json.loads((pathlib.Path(__file__).resolve().parents[1] / "docs" / "model_facts.json").read_text())


def test_n8_cp_radius():
    sr = FACTS["sensitivity_radius"]["N8_seq8192"]["cp"]
    # CP tolerates up to ~5.65x overestimate before flipping
    assert sr["k_hi"] is not None and sr["k_hi"] > 5.0
    assert sr["k_lo"] is None  # never flips below
    assert sr["base_winner"] == [4, 1, 2, 1, 0]
    assert sr["flip_above"] == [1, 8, 1, 1, 0]


def test_n8_pp_never_flips():
    sr = FACTS["sensitivity_radius"]["N8_seq8192"]["pp"]
    assert sr["k_hi"] is None
    assert sr["k_lo"] is None


def test_script_runs():
    import subprocess
    r = subprocess.run(["python3", "scripts/sensitivity_radius.py"], capture_output=True, text=True)
    assert r.returncode == 0
