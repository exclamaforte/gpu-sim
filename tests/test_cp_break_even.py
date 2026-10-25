"""gpu-sim-54m: regression test for CP break-even sequence length."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

def s_best(N=8, model=None):
    if model is None:
        model = {"n_layers":32,"d_model":4096,"n_heads":32,"head_dim":128,"d_ff":14336,"vocab":128256}
    seqs = [8192,10240,12288,14336,16384,20480,24576,32768,65536,131072]
    for seq in seqs:
        m = dict(model); m["seq"] = seq
        hw = {"peak_flops":989e12,"mem_gb":80.0,"fabric":"nvlink4_h100"}
        _, feas = ps.search(N, hw, m=m)
        if feas and feas[0].cp > 1:
            return seq
    return None

def test_s_best_n8_8b():
    # Under the default knobs (act_factor=2.3, crude cp_grad, 80 GB, 8B) the
    # smallest seq where the best config has cp>1 is 32768. The bead's cited
    # 14336-16384 came from an older code state; the current model (with cp_grad
    # and the calibrated activation term) pushes the crossover to 32768.
    # A future change to any term that moves this silently fails here.
    assert s_best() == 32768, s_best()
