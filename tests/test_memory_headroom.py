"""HBM is modelled as a HARD fit, which endorses configs nobody would deploy (gpu-sim-hroom).

`total_mem_bytes <= mem_limit` treats 91% of HBM and 51% of HBM as equally fine. Real
deployments keep slack for fragmentation, allocator rounding, activation spikes and the CUDA
context, so `mem_headroom` (fraction of HBM kept free) is exposed as an hw knob, default 0.0
so no published number moves.

Requiring only 10% free changes exactly TWO cells of the published grid (evaluated at the
measured 0.566 efficiency): N=16 at seq 32768, where (tp=2 cp=8 dp=1) sits at 73.0 of 80 GB
and is replaced by (tp=2 pp=2 cp=4 dp=1 z0) at 41.4 GB for 423.78 vs 413.98 ms (2.4% slower);
and N=8 at seq 16384, where (tp=2 cp=4 dp=1) at 72.9 GB gives way to (tp=4 cp=2 dp=1 z0) at
40.7 GB for 253.43 vs 253.00 ms (0.2% slower). The N=16 replacement is also what wins at N=16
for seq 16384 at pessimistic efficiency, so the requirement costs no real recommendation -- it
removes artifacts. Nothing else changes even at 20% headroom.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from epic_cluster import parallelism_search as ps

PEAK = 989e12
MEASURED = 0.566 / 0.786          # scale peak so the model runs at the measured efficiency
SPEC = {8: "specs/h100_rack.json", 16: "specs/h100_rack.json", 32: "specs/h100_rack_4node.json"}


def _winner(n, seq, headroom):
    model = dict(ps.MODEL)
    model["seq"] = seq
    hw = {"peak_flops": PEAK * MEASURED, "mem_gb": 80.0, "spec_path": SPEC[n],
          "mem_headroom": headroom}
    _, feas = ps.search(n, hw, model)
    assert feas, f"N={n} seq={seq} headroom={headroom} infeasible"
    b = feas[0]
    return (b.tp, b.pp, b.cp, b.dp, b.zero), b.step_sum, b.memory_gb


def test_default_changes_nothing():
    """headroom 0.0 must reproduce the published winner exactly."""
    shape, _, _ = _winner(16, 32768, 0.0)
    assert shape == (2, 1, 8, 1, 0), "the published (tight) N=16 seq32768 winner moved"


def test_headroom_changes_exactly_two_cells():
    changed = []
    for n in (8, 16, 32):
        for seq in (8192, 16384, 32768):
            base, _, _ = _winner(n, seq, 0.0)
            roomy, _, _ = _winner(n, seq, 0.10)
            if base != roomy:
                changed.append((n, seq))
    assert changed == [(8, 16384), (16, 32768)], f"headroom changed unexpected cells: {changed}"


def test_the_roomy_answers_cost_little_and_fix_the_anomalies():
    for n, seq in ((16, 32768), (8, 16384)):
        tight, tight_ms, tight_gb = _winner(n, seq, 0.0)
        roomy, roomy_ms, roomy_gb = _winner(n, seq, 0.10)
        assert tight_gb / 80.0 > 0.90, f"N={n} seq={seq}: the anomaly should sit above 90% of HBM"
        assert roomy_gb / 80.0 < 0.70, f"N={n} seq={seq}: the replacement should leave generous headroom"
        assert roomy_ms / tight_ms - 1 < 0.06, (
            f"N={n} seq={seq}: the replacement should cost well under 6%")
        assert tight != roomy, f"headroom did not change the N={n} seq{seq} winner"
    # and the N=16 roomy choice is not a new invention: it is also what wins at N=16
    # for seq 16384 at pessimistic efficiency, so the requirement removes an artifact,
    # it does not cost a real recommendation.


@pytest.mark.parametrize("headroom", [0.0, 0.10, 0.20])
def test_robust_cells_survive_any_headroom(headroom):
    """N=32 is decisive and roomy; headroom must never change it."""
    for seq in (16384, 32768):
        shape, _, mem_gb = _winner(32, seq, headroom)
        assert shape in ((1, 8, 4, 1, 0), (1, 16, 2, 1, 0)), f"N=32 seq{seq} moved at headroom {headroom}"
        assert mem_gb / 80.0 < 0.75, f"N=32 seq{seq} is not roomy ({mem_gb:.1f} GB)"
