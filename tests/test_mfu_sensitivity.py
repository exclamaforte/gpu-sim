"""The recommendation must not depend on the MFU assumption nobody could validate (gpu-sim-wv2).

The cost model divides 6*P*T FLOPs by dense peak with NO efficiency factor, which for the
published answer works out to an implied 76.5% MFU (test_implied_mfu below). Real bf16 GEMMs
do not reach that: a measured 52-57% on the shapes this model prices (scripts/measure_mfu.py,
real RTX 5090, 8192x4096x4096 and the MLP pair), and published large-model training runs sit
at 35-50%.

So the absolute durations carry a factor-1.25-to-1.6 uncertainty. The question that decides
whether the published recommendation is usable is not "how fast is 94.12 ms really" but "does
the WINNER survive the correction". This test answers that: it re-runs the search at three
efficiency assumptions and requires the same (tp, pp, cp, dp) shape to win at every one.
"""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from epic_cluster import parallelism_search as ps

MODEL_IMPLIED_MFU = 0.765          # 6*P*T / (N*peak) / compute_term, see test below
PEAK = 989e12


def _hw(mfu, n):
    spec = "specs/h100_rack.json" if n <= 16 else "specs/h100_rack_4node.json"
    return {"mem_gb": 80.0, "spec_path": spec, "peak_flops": PEAK * mfu / MODEL_IMPLIED_MFU}


def test_implied_mfu_is_seventy_six_and_a_half_percent():
    """Pin the implicit efficiency: if the compute model changes, this number moves."""
    _, feasible = ps.search(8, _hw(MODEL_IMPLIED_MFU, 8), ps.MODEL)
    best = feasible[0]
    tokens = ps.MODEL["seq"]
    flops = 6 * 8e9 * tokens                     # 8B params, fwd + bwd
    assert best.tp == 4 and best.cp == 2                   # the published winner
    implied = (flops / (8 * PEAK)) / best.compute
    assert implied == pytest.approx(MODEL_IMPLIED_MFU, abs=0.01), (
        f"compute term implies {implied:.3f} of dense peak, not {MODEL_IMPLIED_MFU}")


@pytest.mark.parametrize("n", [8, 16, 32])
def test_winner_shape_is_invariant_to_mfu(n):
    """Measured 5090 (0.566) and a pessimistic H100 training (0.40) must not move the winner."""
    shapes = {}
    for mfu in (MODEL_IMPLIED_MFU, 0.566, 0.40):
        _, feasible = ps.search(n, _hw(mfu, n), ps.MODEL)
        assert feasible, f"N={n} infeasible at MFU {mfu}"
        b = feasible[0]
        shapes[mfu] = (b.tp, b.pp, b.cp, b.dp, b.zero)
    assert len(set(shapes.values())) == 1, f"winner changed with MFU at N={n}: {shapes}"


def test_absolute_durations_move_but_by_a_bounded_factor():
    _, base = ps.search(8, _hw(MODEL_IMPLIED_MFU, 8), ps.MODEL)
    _, slow = ps.search(8, _hw(0.40, 8), ps.MODEL)
    ratio = slow[0].step_sum / base[0].step_sum
    assert 1.4 < ratio < 1.9, f"degrading MFU 0.765->0.40 scaled the step by {ratio:.2f}x"
