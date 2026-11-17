import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

def test_search_accepts_rank_by():
    hw = {"peak_flops":989e12,"mem_gb":80.0,"fabric":"nvlink4_h100","alpha_s":1.5e-6,"bw_gbps":450.0}
    r1, f1 = ps.search(8, hw, rank_by="sum")
    r2, f2 = ps.search(8, hw, rank_by="max")
    assert len(f1) == len(f2)
    # max ranking winner has step_max <= sum winner step_sum (trivially)
    assert f2[0].step_max <= f1[0].step_sum + 1e-9

def test_default_rank_by_sum():
    hw = {"peak_flops":989e12,"mem_gb":80.0,"fabric":"nvlink4_h100"}
    _, f = ps.search(8, hw)
    # default should be sum: winner step_sum equals min step_sum
    assert f[0].step_sum == min(c.step_sum for c in f)
