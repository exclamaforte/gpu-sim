"""Tests for cy0-L4: calibration harness."""
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cy0_l4
from cy0_l4 import calibrate


def make_trace(path, n=5, eff=0.5, observed_scale=1.0):
    """Generate a synthetic trace with known efficiency."""
    records = []
    for i in range(n):
        records.append({
            "kind": "compute",
            "op": f"op_{i}",
            "op_class": "gemm",
            "bytes": 1000,
            "flops": 1e9,
            "seconds": 0.01,  # predicted with eff=0.5
            "observed_seconds": 0.01 * observed_scale,
            "provenance": "test",
            "predicted": True,
        })
    # Add a comm record
    records.append({
        "kind": "comm",
        "op": "tp_allreduce",
        "bytes": 1000,
        "flops": 0,
        "seconds": 0.005,
        "observed_seconds": 0.005,
        "provenance": "test",
        "predicted": True,
    })
    Path(path).write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_refuse_fewer_than_n():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"kind": "compute", "op": "x", "op_class": "gemm",
                           "bytes": 1, "flops": 1, "seconds": 1.0,
                           "provenance": "t", "predicted": True}) + "\n")
        f.flush()
        try:
            calibrate(f.name)
            assert False, "should refuse"
        except ValueError:
            pass
        finally:
            import os
            os.unlink(f.name)


def test_recovers_known_param():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        make_trace(f.name, n=5, observed_scale=2.0)  # observed 2x predicted
        f.flush()
        result = calibrate(f.name)
        # Should have refitted eff
        gemm_hist = [h for h in result["param_histories"] if h["name"] == "eff_gemm"][0]
        assert gemm_hist["n_observations"] == 5
        assert gemm_hist["new_value"] != gemm_hist["old_value"]
        import os
        os.unlink(f.name)


def test_residual_history_per_workload():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        make_trace(f.name, n=5)
        f.flush()
        result = calibrate(f.name, workload_id="wl1")
        assert result["workload_id"] == "wl1"
        for h in result["param_histories"]:
            assert len(h["residuals_before"]) > 0
            assert len(h["residuals_after"]) > 0
        import os
        os.unlink(f.name)


def test_structural_finding_reported():
    # Create a trace with wildly different residuals (structural issue)
    records = []
    for i in range(5):
        records.append({
            "kind": "compute",
            "op": f"op_{i}",
            "op_class": "gemm",
            "bytes": 1000,
            "flops": 1e9,
            "seconds": 0.01 * (i + 1),  # distinct residuals
            "observed_seconds": 0.01,
            "provenance": "test",
            "predicted": True,
        })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(r) for r in records) + "\n")
        f.flush()
        result = calibrate(f.name)
        # Should report structural finding
        assert len(result["structural_findings"]) > 0
        import os
        os.unlink(f.name)


def test_provenance_change_recorded():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        make_trace(f.name, n=5)
        f.flush()
        result = calibrate(f.name)
        for h in result["param_histories"]:
            assert h["old_provenance"] != ""
            assert h["new_provenance"].startswith("fitted:")
            assert h["fit_method"] != ""
        import os
        os.unlink(f.name)


def test_cli_runs():
    import subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        make_trace(f.name, n=5)
        f.flush()
        tmp = f.name
    out = subprocess.run(
        [sys.executable, str(REPO / "cy0_l4.py"), "--trace", tmp],
        capture_output=True, text=True
    )
    assert out.returncode == 0
    result = json.loads(out.stdout)
    assert "param_histories" in result
    import os
    os.unlink(tmp)
