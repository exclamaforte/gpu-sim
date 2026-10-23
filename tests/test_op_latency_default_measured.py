import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ensure no env override
os.environ.pop("GPUSIM_OP_LATENCY_JSON", None)

from epic_u2o import op_latency as ol  # noqa: E402


def test_default_table_is_size_aware():
    a = ol.latency_ms_for("relu", size=1024)
    b = ol.latency_ms_for("relu", size=2 ** 24)
    assert a != b, (a, b)


def test_gate_numbers_unchanged():
    import json, subprocess
    subprocess.run([sys.executable, str(REPO / "experiments" / "gpu-sim-m93" / "run_gate.py")],
                   check=True, capture_output=True)
    metrics = json.loads((REPO / "experiments" / "gpu-sim-m93" / "metrics.json").read_text())
    assert round(metrics["per_class"]["elementwise"]["median_abs_rel_err"], 3) == 0.025
    assert round(metrics["per_class"]["matmul"]["median_abs_rel_err"], 3) == 0.038
    assert round(metrics["per_class"]["reduce"]["median_abs_rel_err"], 3) == 0.024
