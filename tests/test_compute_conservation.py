import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

def test_compute_invariant_across_splits():
    m = dict(ps.MODEL); m["seq"] = 8192
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100"}
    vals = set()
    for tp in (1,2,4):
        for pp in (1,2):
            for cp in (1,2,4):
                for dp in (1,2,4):
                    if tp*pp*cp*dp != 8:
                        continue
                    for zero in (0,):
                        cfg = ps.score(tp, pp, cp, dp, zero, hw, m)
                        vals.add(round(cfg.compute, 12))
    assert len(vals) == 1, f"compute not invariant: {vals}"

def test_compute_invariant_pp():
    m = dict(ps.MODEL); m["seq"] = 8192
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100"}
    c1 = ps.score(1, 1, 1, 8, 3, hw, m)
    c2 = ps.score(1, 2, 1, 4, 3, hw, m)
    assert abs(c1.compute - c2.compute) < 1e-12, (c1.compute, c2.compute)
