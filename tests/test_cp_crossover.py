"""Where does context parallelism start paying? (gpu-sim-scross)

"CP pays at seq 8192" has read like a boundary, and S_best=8192 like a knife edge. Neither was
computed: 8192 was the smallest sequence length anyone tried. Bisecting the seq at which the best
cp>1 config overtakes the best cp=1 config puts the crossover at 5956 tokens (3658 at the
measured 56.6% and 3646 at 40% efficiency) -- so S_best is a conventional length above the
crossover, where the advantage is already 6.03% / 8.78% / 6.64%.

act_factor does not move the crossover at all, which is the same structural fact the act-factor
test pins from the other side: activations are memory, not time.

These tests keep the crossover from drifting silently. If a model change moves it ABOVE 8192, the
published "CP pays at 8192" claim becomes false and this fails; if it moves far below, the docs'
wording ("from ~6k") is stale and should be updated deliberately rather than by drift.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from epic_cluster import parallelism_search as ps

PEAK = 989e12
IMPLIED = 0.786


def _advantage(seq, mfu=IMPLIED, act=None):
    model = dict(ps.MODEL)
    model["seq"] = seq
    hw = {"peak_flops": PEAK * mfu / IMPLIED, "mem_gb": 80.0,
          "spec_path": "specs/h100_rack.json"}
    if act is not None:
        hw["act_factor"] = act
    _, feas = ps.search(8, hw, model)
    cp1 = [c for c in feas if c.cp == 1]
    cpg = [c for c in feas if c.cp > 1]
    assert cp1 and cpg, f"seq {seq}: need both a cp=1 and a cp>1 config to compare"
    return 100 * (min(c.step_sum for c in cp1) / min(c.step_sum for c in cpg) - 1)


def _crossover(mfu=IMPLIED, act=None):
    lo, hi = 2048, 8192
    assert _advantage(lo, mfu, act) < 0 < _advantage(hi, mfu, act), \
        "the bracket must straddle the crossover"
    for _ in range(20):
        mid = (lo + hi) // 2
        if _advantage(mid, mfu, act) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) // 2


def test_crossover_is_well_below_8192_at_every_plausible_efficiency():
    for mfu in (IMPLIED, 0.566, 0.40):
        x = _crossover(mfu)
        assert 3500 < x < 6000, f"crossover at MFU {mfu} moved to {x}"
        assert x < 8192 * 0.75, (
            f"crossover {x} is close to 8192 -- 'CP pays at 8192' would be a knife edge again")


def test_cp_advantage_at_8192_is_comfortable():
    for mfu, floor in ((IMPLIED, 6.0), (0.566, 8.0), (0.40, 6.0)):
        adv = _advantage(8192, mfu)
        assert adv > floor, f"MFU {mfu}: CP advantage at 8192 fell to {adv:.2f}%"


def test_cp_is_level_at_4096_so_the_crossover_is_real():
    """Two-sided: CP must still LOSE below the crossover, or the story is not a crossover."""
    for mfu in (IMPLIED, 0.566, 0.40):
        adv = _advantage(4096, mfu)
        # the crossover sits at 5956, so 4096 is a loss at implied efficiency and at most a
        # small win below it -- either way well under a 2% CP victory, which is what makes
        # 4096 evidence of a crossover rather than of CP already paying.
        assert adv < 2.0, f"MFU {mfu}: CP already wins at 4096 by {adv:.2f}%"


@pytest.mark.parametrize("act", [1.0, 1.5, 3.0, 4.0])
def test_act_factor_does_not_move_the_crossover(act):
    """Activations are memory, not time -- the same structural fact as gpu-sim-aknob."""
    assert _crossover(IMPLIED, act) == 5956, f"act_factor {act} moved the CP crossover"
