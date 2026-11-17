"""Guard: fabric spec derives the nccl_cost.FABRICS table (gpu-sim-3v6d.5)."""
import json, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_cluster"))
from fabric_spec import load_spec, derive_fabrics, fabric_name
from nccl_cost import FABRICS
SPECS = sorted((REPO / "specs").glob("*.json"))

def test_derived_table_matches_fabrics():
    for spec_path in SPECS:
        spec = load_spec(spec_path)
        derived = derive_fabrics(spec)
        for kind, vals in derived.items():
            key = fabric_name(kind)
            assert key in FABRICS, f"{spec_path.name}: {key} not in FABRICS"
            assert abs(vals["alpha_s"] - FABRICS[key]["alpha_s"]) < 1e-12
            assert abs(vals["bw_gbps"] - FABRICS[key]["bw_gbps"]) < 1e-9

def test_provenance_required():
    # Missing sources block entirely
    bad = {"version": 1, "models": {"m": {"sm_count": 1}}, "nodes": [], "links": [], "ranks": []}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad, f); p = f.name
    try:
        try:
            load_spec(p)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "sources" in str(e)
    finally:
        Path(p).unlink()

def test_model_provenance_required():
    # sources present but model missing source
    bad = {"version": 1, "sources": {"link.nvlink": "src"}, "models": {"m": {"sm_count": 1}},
           "nodes": [], "links": [], "ranks": []}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad, f); p = f.name
    try:
        try:
            load_spec(p)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "missing provenance" in str(e)
    finally:
        Path(p).unlink()

def test_heterogeneity_raises():
    spec = {"version": 1, "sources": {"link.nvlink": "src"}, "models": {},
            "nodes": [], "links": [
                {"kind": "nvlink", "bw_gbps": 900.0, "bw_is_bidirectional": True, "latency_us": 1.0},
                {"kind": "nvlink", "bw_gbps": 450.0, "bw_is_bidirectional": True, "latency_us": 1.0},
            ], "ranks": []}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f); p = f.name
    try:
        try:
            derive_fabrics(load_spec(p))
            assert False, "expected ValueError"
        except ValueError as e:
            assert "heterogeneous" in str(e)
    finally:
        Path(p).unlink()
