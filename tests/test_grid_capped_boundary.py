import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.grid_limits import grid_capped as _grid_capped


def test_256x256_accepted():
    assert _grid_capped([256, 256, 1]) is False


def test_y_70000_rejected():
    assert _grid_capped([1, 70000, 1]) is True


def test_product_rule_would_have_rejected_256x256():
    assert (256 * 256 * 1) > 65535


def test_consumers_use_shared_function():
    # measure.py and parse_powered.py both import grid_capped from grid_limits
    import importlib.util
    for mod_path in [
        "experiments/gpu-sim-8zqr/measure.py",
        "experiments/gpu-sim-8zqr/parse_powered.py",
    ]:
        src = Path(mod_path).read_text()
        assert "scripts.grid_limits" in src or "grid_limits" in src, f"{mod_path} does not import grid_limits"
