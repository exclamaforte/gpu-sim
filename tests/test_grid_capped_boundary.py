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

def test_measure_row_path_uses_shared_symbol_under_monkeypatch():
    measure = _load_module("experiments/gpu-sim-8zqr/measure.py")
    grid_limits = sys.modules["scripts.grid_limits"]
    calls = []
    def fake(grid):
        calls.append(list(grid))
        return True  # deliberately wrong verdict
    original = grid_limits.grid_capped
    grid_limits.grid_capped = fake
    try:
        result = measure._row_grid_capped([256, 256, 1])
    finally:
        grid_limits.grid_capped = original
    assert calls == [[256, 256, 1]], f"shared grid_capped was not called: {calls}"
    assert result is True, "fake's verdict did not reach the row path"


def test_parse_powered_row_path_uses_shared_symbol_under_monkeypatch():
    parse_powered = _load_module("experiments/gpu-sim-8zqr/parse_powered.py")
    grid_limits = sys.modules["scripts.grid_limits"]
    calls = []
    def fake(grid):
        calls.append(list(grid))
        return True
    original = grid_limits.grid_capped
    grid_limits.grid_capped = fake
    try:
        result = parse_powered._row_grid_capped([256, 256, 1])
    finally:
        grid_limits.grid_capped = original
    assert calls == [[256, 256, 1]], f"shared grid_capped was not called: {calls}"
    assert result is True, "fake's verdict did not reach the row path"
