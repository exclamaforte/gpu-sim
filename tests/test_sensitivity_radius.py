"""Robustness radius pinned test (gpu-sim-zj9a)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FACTS = json.loads((pathlib.Path(__file__).resolve().parents[1] / "docs" / "model_facts.json").read_text())


def test_n8_cp_radius():
    sr = FACTS["sensitivity_radius"]["N8_seq8192"]
    assert sr["flip_cp_up"] is not None and sr["flip_cp_up"] > 1.5
    assert sr["flip_cp_down"] is not None
    assert sr["flip_pp_up"] is None  # PP never flips upward
    assert sr["winner"] == [4, 1, 2, 1, 0]
    assert sr["winner_at_high_cp"] == [1, 8, 1, 1, 0]


def test_n8_pp_never_flips():
    sr = FACTS["sensitivity_radius"]["N8_seq8192"]
    assert sr["flip_pp_up"] is None


def test_script_runs():
    import subprocess
    r = subprocess.run(["python3", "scripts/sensitivity_radius.py", "--json"], capture_output=True, text=True)
    assert r.returncode == 0
    import json as _j
    data = _j.loads(r.stdout)
    assert "N8_seq8192" in data
