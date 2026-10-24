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
    minstep = min(step_f(c, 1.0) for c in feas)
    tied = sum(1 for c in feas if abs(step_f(c,1.0)-minstep) < 1e-9)
    assert tied == 22, tied
