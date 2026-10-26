"""Where is the recommendation a coin flip? (gpu-sim-frag)

The margin of a winner against the best config with a DIFFERENT (tp,pp,cp,dp) shape is the
number that decides whether a recommendation is a verdict or a tie. Doing this per cell over
N x sequence length x efficiency turned up two things the single-cell analysis could not show:

  1. Fragility is a LONG-SEQUENCE phenomenon, not an N=16 quirk. At seq 8192 only (N=16, 40% MFU)
     is under 2%. From 16384 the N=8 answer itself wins by ~0.2%, and at 32768 it is 0.81% at 40%.
     The CONFIG changes there -- (tp=4 cp=2 dp=1) wins at 8192 and 32768 but (tp=2 cp=4 dp=1)
     takes 16384 -- so the recommendation stands, it is just not decisively separated.
  2. N=16 at 16384 is a three-way tangle: the winner moves from (tp=1 pp=16) to (tp=2 pp=2 cp=4) as
     efficiency drops and the margin never exceeds 0.64%.

N=32 is solid at 8192 but develops ties at longer sequences (down to 0.02% at 32768/typical).

These tests pin the cells so a model change cannot silently turn a decisive answer into a tie,
or quietly repair a tie that the docs warn about.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from epic_cluster import parallelism_search as ps

PEAK = 989e12
IMPLIED_MFU = 0.786
TIGHT = 2.0


def _cell(n, seq, mfu):
    spec = "specs/h100_rack.json" if n <= 16 else "specs/h100_rack_4node.json"
    model = dict(ps.MODEL)
    model["seq"] = seq
    hw = {"peak_flops": PEAK * mfu / IMPLIED_MFU, "mem_gb": 80.0, "spec_path": spec}
    _, feas = ps.search(n, hw, model)
    assert feas, f"N={n} seq={seq} MFU={mfu} has no feasible config"
    b = feas[0]
    alt = next(c for c in feas if (c.tp, c.pp, c.cp, c.dp) != (b.tp, b.pp, b.cp, b.dp))
    return (b.tp, b.pp, b.cp, b.dp), 100 * (alt.step_sum / b.step_sum - 1)


@pytest.mark.parametrize("n,seq", [(8, 8192), (32, 8192)])
def test_decisive_cells_stay_decisive(n, seq):
    """These cells are comfortably separated at every plausible efficiency."""
    for mfu in (IMPLIED_MFU, 0.566, 0.40):
        _, margin = _cell(n, seq, mfu)
        assert margin > TIGHT, f"N={n} seq={seq} MFU={mfu}: margin fell to {margin:.2f}%"


@pytest.mark.parametrize("n,seq", [(32, 16384), (32, 32768)])
def test_n32_long_sequence_is_no_longer_decisive(n, seq):
    """After the batch/PP correction these cells LOST their decisive margins.

    gpu-sim-8pwl review item 3: they used to be the document's only 4-14% cells and the fix dropped
    them to 0.10-2.11%. The test asserts the loss rather than dropping the cells from the suite, so a
    future change that restores decisiveness has to come with an explanation.
    """
    margins = [_cell(n, seq, mfu)[1] for mfu in (IMPLIED_MFU, 0.566, 0.40)]
    assert min(margins) < TIGHT, (
        f"N={n} seq={seq} is decisive again ({min(margins):.2f}%): re-derive the docs claim")
    shape, _ = _cell(n, seq, IMPLIED_MFU)
    assert shape[1] >= 8, f"N={n} seq={seq}: expected a deep-PP winner, got {shape}"


@pytest.mark.parametrize("n,seq,shape", [(8, 32768, (4, 1, 2, 1)), (8, 16384, (2, 1, 4, 1))])
def test_long_sequence_n8_is_known_tight_but_never_switches(n, seq, shape):
    """The N=8 answer is a tie at long sequences -- documented as a tie, not as a verdict."""
    shapes = set()
    margins = []
    for mfu in (IMPLIED_MFU, 0.566, 0.40):
        s, margin = _cell(n, seq, mfu)
        shapes.add(s)
        margins.append(margin)
    assert shapes == {shape}, f"N={n} seq={seq}: the answer stopped being {shape}: {shapes}"
    assert max(margins) < TIGHT, f"N={n} seq={seq}: margin grew to {max(margins):.2f}%"
    assert min(margins) > 0.0, f"N={n} seq={seq}: margin collapsed to {min(margins):.2f}%"


def test_n16_seq16384_is_the_three_way_tangle():
    """Winner moves (tp=2 pp=2 cp=4 dp=1) -> (tp=1 pp=8) as efficiency drops, never separated by 2.5%."""
    shapes = {}
    for mfu in (IMPLIED_MFU, 0.566, 0.40):
        shape, margin = _cell(16, 16384, mfu)
        shapes[mfu] = (shape, margin)
    assert len({v[0] for v in shapes.values()}) >= 2, "the tangle collapsed to one answer"
    assert max(v[1] for v in shapes.values()) < 2.5, "the tangle gained a clear winner"
    assert all(v[0][2] >= 1 for v in shapes.values()), "CP should stay in the picture at N=16/16384"


@pytest.mark.parametrize("n,seq", [(32, 16384), (32, 32768)])
def test_n32_long_sequences_are_no_longer_decisive(n, seq):
    """After the batch/PP fix these cells lost their decisive margin (margin < 2%)."""
    for mfu in (IMPLIED_MFU, 0.566, 0.40):
        _, margin = _cell(n, seq, mfu)
        assert margin < 2.5, f"N={n} seq={seq} MFU={mfu}: margin {margin:.2f}% is still decisive"
