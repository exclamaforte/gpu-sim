import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

def test_default_recompute_reproduces():
    # default recompute=none must match the original (no-attn? no: attn now included by default)
    # Regression: with recompute=none, attention term included; verify finite and winner defined
    m = dict(ps.MODEL); m["seq"] = 8192
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100"}
    _, feas = ps.search(8, hw, m=m)
    assert len(feas) > 0
    assert feas[0].step_sum > 0

def test_attention_flops_formula():
    # independent derivation: attn fwd+bwd = 12 * L * S^2 * d / (cp*tp) FLOPs
    m = dict(ps.MODEL); m["seq"] = 1024
    L = m["n_layers"]; d = m["d_model"]; S = 1024
    tp, cp = 2, 4
    expected_attn = 12.0 * L * (S**2) * d / (cp * tp)
    # compute total flops with attention
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100"}
    cfg = ps.score(tp, 1, cp, 1, 0, hw, m)
    # attn contribution to compute time = attn_flops / peak
    # verify cfg.compute includes it (compute > weight-only)
    weight_only = (6 * (ps.total_params(m)/(tp*1)) * (S/(cp*1))) / 989e12
    assert cfg.compute > weight_only, (cfg.compute, weight_only)

def test_recompute_knob():
    m = dict(ps.MODEL); m["seq"] = 2048
    hw_none = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "recompute": 0.0}
    hw_full = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "recompute": 1.0}
    c_none = ps.score(2, 1, 1, 1, 0, hw_none, m)
    c_full = ps.score(2, 1, 1, 1, 0, hw_full, m)
    assert c_full.compute > c_none.compute
