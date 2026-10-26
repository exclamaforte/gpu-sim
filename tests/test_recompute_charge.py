"""gpu-sim-hz3u: full recompute charges exactly one extra forward pass.

The model already had a recompute knob (none/selective/full, gpu-sim-ooj), but the full
charge added the WHOLE 6P+attn compute a second time (+100%). Full checkpointing replays one
forward pass per step (fwd is 2P of the 6P matmul total; attention fwd is 1/3 of the 12*S^2*d
total because bwd is 2x fwd), i.e. ~+33% -- the standard figure. An overcharged recompute term
would make every with-recompute answer say recompute is more expensive than it is.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps


def _hw(rec):
    return {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
            "batch": 1, "recompute": rec}


def test_full_recompute_charges_one_forward_pass():
    m = dict(ps.MODEL)
    m["seq"] = 2048
    tp, pp, cp, dp, zero = 2, 1, 1, 1, 0
    c_none = ps.score(tp, pp, cp, dp, zero, _hw("none"), m)
    c_full = ps.score(tp, pp, cp, dp, zero, _hw("full"), m)
    P, S, d, L = ps.total_params(m), m["seq"], m["d_model"], m["n_layers"]
    tokens = S  # batch=1, cp=dp=1: one sequence per rank
    fwd_matmul = 2 * (P / (tp * pp)) * tokens
    attn_total = 12.0 * L * (S ** 2) * d / (tp * pp * cp * dp)
    expected = c_none.compute + (fwd_matmul + attn_total / 3.0) / 989e12
    assert abs(c_full.compute - expected) < 1e-12 * expected, (
        c_full.compute, expected)
    # magnitude: about a third extra, not about double
    ratio = (c_full.compute - c_none.compute) / c_none.compute
    assert 0.2 < ratio < 0.5, ratio


def test_full_recompute_memory_below_none():
    # the memory side of the same knob: full keeps one layer, so it must use less
    m = dict(ps.MODEL)
    m["seq"] = 2048
    c_none = ps.score(2, 4, 1, 1, 0, _hw("none"), m)
    c_full = ps.score(2, 4, 1, 1, 0, _hw("full"), m)
    assert c_full.memory_gb < c_none.memory_gb, (c_full.memory_gb, c_none.memory_gb)
