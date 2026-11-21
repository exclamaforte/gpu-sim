"""Guard: the launch floor is a warm minimum, not a cold median (gpu-sim-mgxy).

MEASURED BEFORE (experiments/gpu-sim-mgxy/instrument_null.py, 20 repeats on the
RTX 5090, prebuilt 8zqr binaries): the NULL cell -- the empty binary measured
against ITSELF -- came out negative in 10/20 repeats under the median floor,
range [-1.33, +1.62]us. A kernel cannot be faster than itself, so a negative
kernel_us was the estimator's own noise, not a property of the kernel; the real
tiny cell (divergent_branch build 0) was negative in 4/20. With the warm
minimum floor the null cell drops to 1/20 and the real cell to 0/20.

These tests pin the estimator on synthetic series, so they run without a GPU:
the floor must ignore cold launches, a negative must stay negative (never
clamped) and be labelled as inside the floor's noise only when it really is.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _measure():
    path = ROOT / "experiments" / "gpu-sim-8zqr" / "measure.py"
    spec = importlib.util.spec_from_file_location("measure_mgxy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spec(size=1024, grid=(16, 1, 1), block=(64, 1, 1)):
    return {
        "family": "divergent_branch", "build_idx": 0, "op_type": "divergent_branch",
        "problem_size": size, "block": list(block), "grid": list(grid),
        "metadata": {}, "phys": {},
    }


def test_floor_ignores_cold_launches():
    """A cold first launch must not become the floor via the median."""
    m = _measure()
    # 10 trials: one cold 30ms launch, then 8ms, then a warm 3ms band.
    empty = [30.0, 8.0, 8.0, 8.0, 8.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    scale = [5.0] * 10
    row = m.build_row(_spec(), scale, empty)
    # median of the whole series would be 5.5ms (a mixed cold/warm number);
    # min over the warm tail is 3.0ms.
    assert row["estimator"] == "warm_min_floor"
    assert abs(row["empty_floor_ms"] - 3.0) < 1e-9, row["empty_floor_ms"]
    assert abs(row["kernel_us"] - 2000.0) < 1e-6, row["kernel_us"]


def test_null_cell_is_not_a_measurement():
    """empty measured against itself: no positive result, flagged, never usable."""
    m = _measure()
    series = [5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 6.0, 3.0, 3.0, 3.0]
    row = m.build_row(_spec(), list(series), list(series))
    assert row["kernel_us"] <= 0.0, row["kernel_us"]
    assert row["target_usable"] is False
    assert row["within_floor_noise"] is True, row


def test_genuine_below_floor_is_never_clamped():
    """A real negative stays signed and says why; clamping would hide it."""
    m = _measure()
    row = m.build_row(_spec(), [1.0] * 10, [10.0] * 10)
    assert row["kernel_us"] == -9000.0, row["kernel_us"]
    assert row["rejection_reason"] == "non_positive_kernel"
    assert row["log_kernel_us"] is None
    assert row["target_usable"] is False
    # 9ms below a zero-jitter floor is NOT inside the noise band
    assert row["within_floor_noise"] is False, row


def test_above_threshold_still_usable():
    """The floor change must not buy its null-cell result with a false negative."""
    m = _measure()
    row = m.build_row(_spec(), [200.0] * 10, [5.0] * 10)
    assert row["kernel_us"] == 195000.0, row["kernel_us"]
    assert row["target_usable"] is True, row
    assert row["within_floor_noise"] is False


def test_warm_trim_does_not_swallow_a_big_cold_scale_trial():
    """A slow FIRST scale launch must not be what the median is computed on."""
    m = _measure()
    scale = [900.0] + [40.0] * 9          # cold spike only
    row = m.build_row(_spec(), scale, [5.0] * 10)
    # median of the warm tail is 40ms, not the 900ms spike and not 470ms
    assert abs(row["kernel_us"] - 35000.0) < 1e-6, row["kernel_us"]


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
