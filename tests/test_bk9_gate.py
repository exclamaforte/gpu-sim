import json, subprocess, sys, pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-bk9" / "metrics.json"
BUILDER = REPO / "experiments" / "gpu-sim-bk9" / "build_dataset.py"

def test_gate_refuses_unusable_rows():
    r = subprocess.run([sys.executable, str(BUILDER), "--rows", "all",
                        "--out", str(pathlib.Path("/tmp") / "bk9_ds.json")],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode != 0
    assert "REFUSED: target_usable=false rows" in (r.stderr + r.stdout)

def test_gate_allows_usable_subset():
    metrics = json.loads(METRICS.read_text())
    usable = [f"{c['family']}:{c['size']}" for c in metrics["cells"] if c["target_usable"]]
    assert usable, "no usable rows"
    r = subprocess.run([sys.executable, str(BUILDER), "--rows", ",".join(usable),
                        "--out", str(pathlib.Path("/tmp") / "bk9_ds_ok.json")],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr