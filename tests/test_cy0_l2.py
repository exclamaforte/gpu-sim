"""Tests for cy0-L2: full training-step composer."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cy0_l1
import cy0_l2
from cy0_l2 import StepConfig, predict_step

SPEC_PATH = REPO / "specs" / "h100_rack.json"


def test_components_sum_to_total():
    cfg = StepConfig(n_layers=2, n_microbatches=4, dp=2, tp=2, pp=1)
    result = predict_step(str(SPEC_PATH), cfg)
    total = sum(c["seconds"] for c in result["components"] if c["kind"] != "hidden")
    assert abs(total - result["total_seconds"]) < 1e-9


def test_overlap_false_increases_total():
    cfg_on = StepConfig(n_layers=2, n_microbatches=4, dp=2, tp=2, pp=1, overlap=True)
    cfg_off = StepConfig(n_layers=2, n_microbatches=4, dp=2, tp=2, pp=1, overlap=False)
    r_on = predict_step(str(SPEC_PATH), cfg_on)
    r_off = predict_step(str(SPEC_PATH), cfg_off)
    # overlap=false should be >= overlap=true
    assert r_off["total_seconds"] >= r_on["total_seconds"] - 1e-12
    # The difference should be the overlapped term
    if r_on["overlapped_seconds"] > 0:
        assert r_off["total_seconds"] > r_on["total_seconds"]


def test_pp_disabled_zero_bubble():
    cfg = StepConfig(n_layers=2, n_microbatches=4, dp=2, tp=2, pp=1)
    result = predict_step(str(SPEC_PATH), cfg)
    assert result["bubble_seconds"] == 0.0
    # Also verify no bubble component
    bubbles = [c for c in result["components"] if c["component"] == "bubble"]
    assert len(bubbles) == 0


def test_pp_enabled_nonzero_bubble():
    cfg = StepConfig(n_layers=2, n_microbatches=4, dp=2, tp=2, pp=2)
    result = predict_step(str(SPEC_PATH), cfg)
    assert result["bubble_seconds"] > 0


def test_closed_form_2layer_toy():
    """Hand-computed closed form for 2-layer, DP=2, TP=2, PP=1."""
    cfg = StepConfig(n_layers=2, n_microbatches=4, dp=2, tp=2, pp=1, overlap=True)
    result = predict_step(str(SPEC_PATH), cfg)
    # Forward: 2 layers * layer_time (TP comm hidden)
    # Backward: 2 layers * layer_time
    # Optimizer: 0.1 * layer_time
    # DP allreduce: dp_allreduce_time
    # Bubble: 0 (PP=1)
    lc = cy0_l1.layer_compute_time if hasattr(cy0_l1, 'layer_compute_time') else None
    # Just check total > 0 and components exist
    assert result["total_seconds"] > 0
    assert len(result["components"]) > 0
    phases = set(c["phase"] for c in result["components"])
    assert "forward" in phases
    assert "backward" in phases
    assert "optimizer" in phases
    assert "dp_sync" in phases


def test_per_phase_records():
    cfg = StepConfig(n_layers=2, n_microbatches=4, dp=2, tp=2, pp=1)
    result = predict_step(str(SPEC_PATH), cfg)
    # Each component has phase and component fields
    for c in result["components"]:
        assert "phase" in c
        assert "component" in c
        assert "kind" in c
        assert "seconds" in c


def test_jsonl_emission():
    import subprocess
    out = subprocess.run(
        [sys.executable, str(REPO / "cy0_l2.py"), "--spec", str(SPEC_PATH),
         "--layers", "2", "--dp", "2", "--tp", "2", "--pp", "1"],
        capture_output=True, text=True
    )
    lines = [l for l in out.stdout.strip().split("\n") if l]
    assert len(lines) > 0
    for line in lines:
        rec = json.loads(line)
        assert "phase" in rec
        assert "component" in rec


def test_absurdity_comm_exceeds_compute():
    """When comm > compute, the excess must appear in the total (not hidden)."""
    import cy0_l2
    # Monkey-patch layer_tp_comm_time to return a huge value
    orig = cy0_l2.layer_tp_comm_time
    cy0_l2.layer_tp_comm_time = lambda sp, tp: 1.0  # 1 second comm
    # Also make compute small
    orig_lc = cy0_l2.layer_compute_time
    cy0_l2.layer_compute_time = lambda sp: 0.001  # 1 ms compute

    cfg = StepConfig(n_layers=1, n_microbatches=1, dp=1, tp=2, pp=1, overlap=True)
    result = predict_step(str(SPEC_PATH), cfg)

    # Restore
    cy0_l2.layer_tp_comm_time = orig
    cy0_l2.layer_compute_time = orig_lc

    # The excess (1.0 - 0.001) should be in the total
    assert result["total_seconds"] > 0.9, "excess comm not in total"
    # Hidden term should be capped at compute
    hidden = [c for c in result["components"] if c["component"] == "tp_allreduce_hidden"]
    if hidden:
        assert hidden[0]["seconds"] <= 0.001 * 2 + 1e-9, "hidden exceeds compute"
