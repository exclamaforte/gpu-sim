"""Regression: measure.py marks non-positive kernel_us rows (gpu-sim-7m60)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module(relpath):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nonpositive_kernel_marked_and_not_usable():
    measure = _load_module("experiments/gpu-sim-8zqr/measure.py")
    spec = {
        "family": "f", "build_idx": 0, "op_type": "gemm",
        "problem_size": 1024, "block": [256, 1, 1],
        "grid": [256, 256, 1], "metadata": {}, "phys": {},
    }
    # scale time below empty time => negative kernel_us
    scale_trials = [0.5] * 10
    empty_trials = [1.0] * 5
    row = measure.build_row(spec, scale_trials, empty_trials)
    assert row["kernel_us"] <= 0
    assert row["target_usable"] is False
    assert row["rejection_reason"] == "non_positive_kernel"
    assert row["log_kernel_us"] is None


def test_positive_kernel_not_rejected_as_nonpositive():
    measure = _load_module("experiments/gpu-sim-8zqr/measure.py")
    spec = {
        "family": "f", "build_idx": 0, "op_type": "gemm",
        "problem_size": 1024, "block": [256, 1, 1],
        "grid": [256, 256, 1], "metadata": {}, "phys": {},
    }
    scale_trials = [2.0] * 10
    empty_trials = [0.5] * 5
    row = measure.build_row(spec, scale_trials, empty_trials)
    assert row["kernel_us"] > 0
    assert row["rejection_reason"] != "non_positive_kernel"
