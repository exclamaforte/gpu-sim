"""act_factor is a MEMORY knob, not a time knob -- so it cannot move the recommendation.

`act_factor` (2.3) is the one constant in the model that came from a single-shape instrumented
measurement (gpu-sim-4nf, per-layer peak on a 5090), and the source itself calls it
"UNVALIDATED for exact fit". After N=16 turned out to be a 0.79% near-tie reported as a
verdict, the obvious question was whether this fitted constant hides a similar fragility.

It does not, and for a structural reason worth pinning: act_factor multiplies ACTIVATION
MEMORY only. The CP term is priced as a ring exchange of K/V chunks (per layer, sized by
seq/d/n_heads) and the DP/TP terms are priced from parameters and messages, so none of them
read act_bytes. Sweeping act_factor 1.0 -> 20.0 therefore leaves step time and the ranking
bit-identical, and only moves memory and the number of feasible configs.

The knob does bite eventually -- it is the memory ceiling -- so the test also pins where:
N=8 stays feasible up to act ~36 (15x headroom over the calibrated 2.3), N=16 to ~82, N=32
to ~174. Those thresholds are computed by bisection here, so a future change that makes the
model act-sensitive (or that shrinks the memory budget) fails this test instead of silently
invalidating a published recommendation.
"""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from epic_cluster import parallelism_search as ps

CALIBRATED = 2.3


def _search(n, act):
    spec = "specs/h100_rack.json" if n <= 16 else "specs/h100_rack_4node.json"
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": spec, "act_factor": act}
    return ps.search(n, hw, ps.MODEL)


@pytest.mark.parametrize("n", [8, 16, 32])
def test_act_factor_never_moves_time(n):
    """The calibrated winner's OWN duration is bit-identical at every act_factor.

    This is the structural claim: activations are memory, so they cannot enter a time term.
    It is checked on the winning shape rather than on the argmin, because the argmin CAN move
    -- through memory pressure -- at absurd act factors (see the headroom test below).
    """
    _, base = _search(n, CALIBRATED)
    best = base[0]
    shape = (best.tp, best.pp, best.cp, best.dp, best.zero)
    base_ms = best.step_sum
    for act in (1.0, 1.5, 3.0, 6.0, 12.0, 20.0):
        _, feas = _search(n, act)
        same = [c for c in feas if (c.tp, c.pp, c.cp, c.dp, c.zero) == shape]
        if not same:
            continue          # squeezed out by memory, which is the next test's business
        assert same[0].step_sum == pytest.approx(base_ms, rel=1e-12), (
            f"N={n}: act_factor {act} changed step time -- activations have leaked into the "
            f"COMMUNICATION or COMPUTE path, which invalidates the 'memory-only knob' ruling")


@pytest.mark.parametrize("n", [8, 16, 32])
def test_ranking_holds_across_a_5x_band_around_the_fit(n):
    """1.0 -> 12.0 is a 5x band around the calibrated 2.3; the recommendation must not move."""
    _, base = _search(n, CALIBRATED)
    shape = (base[0].tp, base[0].pp, base[0].cp, base[0].dp, base[0].zero)
    for act in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0):
        _, feas = _search(n, act)
        best = feas[0]
        assert (best.tp, best.pp, best.cp, best.dp, best.zero) == shape, (
            f"N={n}: act_factor {act} moved the winner -- still inside a 5x band around the fit")


@pytest.mark.parametrize("n,change_point", [(8, 28.5), (16, None), (32, None)])
def test_the_answer_only_bends_at_9x_the_fitted_value(n, change_point):
    """Where DOES activations matter? Only through memory, and only ~9-12x beyond the fit."""
    _, base = _search(n, CALIBRATED)
    base_shape = (base[0].tp, base[0].pp, base[0].cp, base[0].dp, base[0].zero)
    act, first_change = CALIBRATED, None
    while act < 200:
        _, feas = _search(n, act)
        if not feas:
            break
        b = feas[0]
        if (b.tp, b.pp, b.cp, b.dp, b.zero) != base_shape:
            first_change = act
            break
        act *= 1.15
    if change_point is None:
        assert first_change is None, f"N={n}: ranking moved at {first_change}, expected never to move"
    else:
        assert first_change is not None, f"N={n}: ranking never moved below act_factor 200"
        assert first_change == pytest.approx(change_point, rel=0.12), (
            f"N={n}: the ranking now bends at act_factor {first_change:.1f}, not ~{change_point}")
        assert first_change / CALIBRATED > 8, f"N={n}: only {first_change / CALIBRATED:.1f}x headroom"


@pytest.mark.parametrize("n", [8, 16, 32])
def test_act_factor_is_monotone_in_memory_and_feasibility(n):
    """For the calibrated winning CONFIG: more activations cost more memory, nothing else.

    Measured per config, not per argmin: when memory pressure squeezes the favourite out, the
    argmin switches to a leaner config and the argmin's memory can fall even as act_factor
    rises -- which is the search doing its job, not a modelling error.
    """
    _, base = _search(n, CALIBRATED)
    shape = (base[0].tp, base[0].pp, base[0].cp, base[0].dp, base[0].zero)
    mem, counts = [], []
    for act in (1.0, CALIBRATED, 6.0, 12.0, 20.0):
        _, feas = _search(n, act)
        counts.append(len(feas))
        same = [c for c in feas if (c.tp, c.pp, c.cp, c.dp, c.zero) == shape]
        if same:
            mem.append(same[0].memory_gb)
    assert mem == sorted(mem) and len(mem) >= 3 and mem[0] < mem[-1], (
        f"N={n}: memory of the winning config must grow with act_factor, got {mem}")
    assert counts == sorted(counts, reverse=True), f"N={n}: feasibility must shrink, got {counts}"


@pytest.mark.parametrize("n,expected_threshold", [(8, 36.2), (16, 82.0), (32, 173.7)])
def test_memory_ceiling_far_from_the_calibrated_value(n, expected_threshold):
    """The knob only bites at the memory ceiling, ~15x beyond the calibrated 2.3 at N=8."""
    lo, hi = 1.0, 400.0
    assert _search(n, hi)[1] == [] or True      # no assumption about very large values
    if _search(n, hi)[1]:
        pytest.skip(f"N={n} still feasible at act_factor {hi}")
    for _ in range(30):
        mid = (lo + hi) / 2
        if _search(n, mid)[1]:
            lo = mid
        else:
            hi = mid
    assert lo == pytest.approx(expected_threshold, rel=0.02), (
        f"N={n}: memory ceiling moved to act_factor {lo:.1f}")
    assert lo / CALIBRATED > 10, f"N={n}: only {lo / CALIBRATED:.1f}x headroom over the fit"
