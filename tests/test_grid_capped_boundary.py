import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grid_limits import grid_capped as _grid_capped


def _load_module(relpath):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_256x256_accepted():
    assert _grid_capped([256, 256, 1]) is False


def test_y_70000_rejected():
    assert _grid_capped([1, 70000, 1]) is True


def test_product_rule_would_have_rejected_256x256():
    assert (256 * 256 * 1) > 65535


def test_measure_exposes_shared_function_by_identity():
    measure = _load_module("experiments/gpu-sim-8zqr/measure.py")
    grid_limits = sys.modules["scripts.grid_limits"]
    assert getattr(measure, "GRID_CAPPED", None) is grid_limits.grid_capped


def test_parse_powered_exposes_shared_function_by_identity():
    parse_powered = _load_module("experiments/gpu-sim-8zqr/parse_powered.py")
    grid_limits = sys.modules["scripts.grid_limits"]
    assert getattr(parse_powered, "GRID_CAPPED", None) is grid_limits.grid_capped


def test_measure_own_path_256x256_not_capped():
    measure = _load_module("experiments/gpu-sim-8zqr/measure.py")
    assert measure.GRID_CAPPED([256, 256, 1]) is False


def test_measure_own_path_y_70000_capped():
    measure = _load_module("experiments/gpu-sim-8zqr/measure.py")
    assert measure.GRID_CAPPED([1, 70000, 1]) is True


def test_parse_powered_own_path_256x256_not_capped():
    parse_powered = _load_module("experiments/gpu-sim-8zqr/parse_powered.py")
    assert parse_powered.GRID_CAPPED([256, 256, 1]) is False


def test_parse_powered_own_path_y_70000_capped():
    parse_powered = _load_module("experiments/gpu-sim-8zqr/parse_powered.py")
    assert parse_powered.GRID_CAPPED([1, 70000, 1]) is True


class TestGridCappedBoundary:
    def test_measure_build_row_uses_shared_symbol_under_monkeypatch(self):
        measure = _load_module("experiments/gpu-sim-8zqr/measure.py")
        grid_limits = sys.modules["scripts.grid_limits"]
        calls = []
        def fake(grid):
            calls.append(list(grid))
            return True  # deliberately wrong verdict
        original = grid_limits.grid_capped
        grid_limits.grid_capped = fake
        try:
            spec = {
                "family": "gemm_naive",
                "build_idx": 0,
                "op_type": "gemm",
                "problem_size": 1024,
                "block": [256, 1, 1],
                "grid": [256, 256, 1],
                "metadata": {"compute_bound": True},
                "phys": {},
            }
            scale_trials = [1.0] * 10
            empty_trials = [0.1] * 5
            row = measure.build_row(spec, scale_trials, empty_trials)
        finally:
            grid_limits.grid_capped = original
        assert calls and calls[0] == [256, 256, 1], f"shared grid_capped was not called: {calls}"
        assert row["grid_capped"] is True, "fake's verdict did not reach the row"
        assert row["target_usable"] is False, "wrong verdict did not propagate to usable"


def test_parse_powered_row_loop_uses_shared_symbol_under_monkeypatch(tmp_path):
    parse_powered = _load_module("experiments/gpu-sim-8zqr/parse_powered.py")
    grid_limits = sys.modules["scripts.grid_limits"]
    # tiny fixture: one trace file
    tr = tmp_path / "traces_fk"
    tr.mkdir()
    import json as _json
    trace = tr / "gemm_naive_1024_256.txt"
    scale_line = _json.dumps({"summary": True, "mode": "scale", "median_us": 100.0, "p95_us": 101.0, "p05_us": 99.0})
    empty_line = _json.dumps({"summary": True, "mode": "empty", "median_us": 1.0, "p95_us": 1.1, "p05_us": 0.9})
    trace.write_text(scale_line + "\n" + empty_line + "\n")
    calls = []
    def fake(grid):
        calls.append(list(grid))
        return True
    original = grid_limits.grid_capped
    grid_limits.grid_capped = fake
    try:
        rows = parse_powered.build_rows(str(tr))
    finally:
        grid_limits.grid_capped = original
    assert rows, "no rows produced"
    assert calls, f"shared grid_capped was not called: {calls}"
    assert rows[0]["grid_capped"] is True, "fake's verdict did not reach the row"
    assert rows[0]["target_usable"] is False, "wrong verdict did not propagate to usable"
