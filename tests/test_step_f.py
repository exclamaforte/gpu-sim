import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps
import importlib.util
spec = importlib.util.spec_from_file_location("sens", "experiments/gpu-sim-5o4.5/sensitivity.py")
# sensitivity.py runs on import; instead replicate step_f
def step_f(c, f):
    comm = c.tp_comm + c.cp_comm + c.pp_comm + c.dp_comm
    compute = c.step_sum - comm
    return compute + max(0.0, comm - f * compute)

def test_step_f_one_equals_step_max():
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "alpha_s": 1.5e-6, "bw_gbps": 450.0}
    _, feas = ps.search(8, hw)
    for c in feas:
        assert abs(step_f(c, 1.0) - c.step_max) < 1e-9, (c, step_f(c,1.0), c.step_max)

def test_tie_count_at_f1():
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "alpha_s": 1.5e-6, "bw_gbps": 450.0}
    _, feas = ps.search(8, hw)
    pp1 = [c for c in feas if c.pp == 1]
    assert len(pp1) == 24
    comms = [c.tp_comm + c.cp_comm + c.pp_comm + c.dp_comm for c in pp1]
    computes = [c.step_sum - k for c, k in zip(pp1, comms)]
    # Work conservation (gpu-sim-i95): at fixed N every split has the SAME compute,
    # so a varying compute column would mean a term that does not scale with the
    # split. This is the assertion that catches that class of bug.
    assert max(computes) - min(computes) < 1e-9, (min(computes), max(computes))
    # At 8B / seq 8192 every pp=1 comm term fits under the compute floor, so the
    # whole pp=1 space ties at f=1. Derived, not a literal: the count is len(pp1)
    # exactly while max(comm) <= compute, which is asserted rather than assumed.
    # (It was 22 when the floor was 51.63 ms weight-only; gpu-sim-i95 raised the
    # floor to 64.97 ms and the two stragglers now fit. 24 is not a new truth.)
    assert max(comms) <= min(computes) + 1e-12, (max(comms), min(computes))
    minstep = min(step_f(c, 1.0) for c in feas)
    tied = sum(1 for c in feas if abs(step_f(c,1.0)-minstep) < 1e-9)
    assert tied == len(pp1), tied
