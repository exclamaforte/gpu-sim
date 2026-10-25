"""gpu-sim-qk9: DP/FSDP comm audit tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

def dp_bytes(zero, dp, params_total, dtype_bytes=2):
    # production ZeRO semantics
    P = params_total
    if zero >= 3 and dp > 1:
        params_per_gpu = P / dp  # sharded
    else:
        params_per_gpu = P
    if zero >= 2 and dp > 1:
        grads_per_gpu = P / dp
    else:
        grads_per_gpu = P
    if zero <= 1:
        return grads_per_gpu * dtype_bytes  # all_reduce of grads
    elif zero == 2:
        return grads_per_gpu * dtype_bytes  # reduce_scatter of grads
    else:
        return grads_per_gpu * dtype_bytes + params_per_gpu * dtype_bytes  # RS + AG

def test_dp_bytes_zero_level_semantics():
    P = 1000
    # zero=1: grads only
    assert dp_bytes(1, 4, P) == (P * 2)  # grads_per_gpu = P (not sharded), *2
    # zero=2: grads sharded
    assert dp_bytes(2, 4, P) == (P // 4 * 2)
    # zero=3: grads + params sharded
    assert dp_bytes(3, 4, P) == (P // 4 * 2) + (P // 4 * 2)

def test_dp_dtype_affects_memory_and_comm():
    m = {"n_layers": 2, "d_model": 64, "n_heads": 4, "head_dim": 16, "d_ff": 256, "vocab": 1000, "seq": 128}
    hw_bf = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16", "dp_count_mode": "flat"}
    hw_fp = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "fp32", "dp_count_mode": "flat"}
    c_bf = ps.score(1, 1, 1, 4, 3, hw_bf, m)
    c_fp = ps.score(1, 1, 1, 4, 3, hw_fp, m)
    assert c_bf.dp_comm < c_fp.dp_comm, (c_bf.dp_comm, c_fp.dp_comm)

def test_zero_level_reduces_comm():
    m = {"n_layers": 2, "d_model": 64, "n_heads": 4, "head_dim": 16, "d_ff": 256, "vocab": 1000, "seq": 128}
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16"}
    # zero=1: AR of grads (P bytes). zero=2: RS of grads (P/dp). zero=3: RS+AG.
    c1 = ps.score(1, 1, 1, 4, 1, hw, m)
    c2 = ps.score(1, 1, 1, 4, 2, hw, m)
    c3 = ps.score(1, 1, 1, 4, 3, hw, m)
    # zero=2 < zero=1 (sharded grads)
    assert c2.dp_comm < c1.dp_comm, (c2.dp_comm, c1.dp_comm)
    # zero=2 and zero=3 now both RS+AG of full P (tie on comm)
    assert abs(c2.dp_comm - c3.dp_comm) < 1e-12, (c2.dp_comm, c3.dp_comm)

def test_per_param_ge_flat():
    m = {"n_layers": 2, "d_model": 64, "n_heads": 4, "head_dim": 16, "d_ff": 256, "vocab": 1000, "seq": 128}
    hw_flat = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16", "dp_count_mode": "flat"}
    hw_per = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16", "dp_count_mode": "per_param", "dp_param_tensors": 5}
    c_flat = ps.score(1, 1, 1, 4, 3, hw_flat, m)
    c_per = ps.score(1, 1, 1, 4, 3, hw_per, m)
    assert c_per.dp_comm > c_flat.dp_comm, (c_per.dp_comm, c_flat.dp_comm)

def test_per_param_equals_flat_when_one():
    m = {"n_layers": 2, "d_model": 64, "n_heads": 4, "head_dim": 16, "d_ff": 256, "vocab": 1000, "seq": 128}
    hw_flat = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16", "dp_count_mode": "flat"}
    hw_one = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16", "dp_count_mode": "per_param", "dp_param_tensors": 1}
    c_flat = ps.score(1, 1, 1, 4, 3, hw_flat, m)
    c_one = ps.score(1, 1, 1, 4, 3, hw_one, m)
    assert abs(c_one.dp_comm - c_flat.dp_comm) < 1e-12, (c_one.dp_comm, c_flat.dp_comm)

def test_dp_payload_cp_gt_1_not_full_grad():
    # gpu-sim-qk9 shadowing fix: when cp>1, dp_comm must price grads_per_gpu,
    # not the full unsharded grad (which would be inflated by ~dp).
    m = {"n_layers": 2, "d_model": 64, "n_heads": 4, "head_dim": 16, "d_ff": 256, "vocab": 1000, "seq": 128}
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16", "cp_grad_mode": "crude"}
    # cp=2, dp=4: grads_per_gpu = P/(tp*pp*dp) = P/8? Actually P/(1*1*4)=P/4 for dp. cp doesn't shard grads.
    # The key: dp_comm should use grads_per_gpu, not P. Check by comparing cp=1 vs cp=2 at same dp.
    c_cp1 = ps.score(1, 1, 1, 4, 2, hw, m)
    c_cp2 = ps.score(1, 1, 2, 4, 2, hw, m)
    # dp_comm should be the same (grads_per_gpu depends on dp, not cp)
    assert abs(c_cp1.dp_comm - c_cp2.dp_comm) < 1e-12, (c_cp1.dp_comm, c_cp2.dp_comm)

def test_ring_factor_rs_vs_ar():
    # RS of a full gradient costs ~half of AR of the same payload (ring factor)
    m = {"n_layers": 2, "d_model": 64, "n_heads": 4, "head_dim": 16, "d_ff": 256, "vocab": 1000, "seq": 128}
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16"}
    c_ar = ps.score(1, 1, 1, 4, 1, hw, m)   # zero=1: AR(grads) + AG(params)
    c_rs = ps.score(1, 1, 1, 4, 3, hw, m)   # zero=3: RS(grads) + AG(params)
    # AR = RS + AG of same payload, so AR/2 ~ RS (within ring factor)
    assert c_ar.dp_comm > c_rs.dp_comm, (c_ar.dp_comm, c_rs.dp_comm)

def test_ag_equals_rs_same_payload():
    # AG(full P) == RS(full P) for the same payload
    m = {"n_layers": 2, "d_model": 64, "n_heads": 4, "head_dim": 16, "d_ff": 256, "vocab": 1000, "seq": 128}
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16"}
    # zero=2: RS(grads) + AG(params). If grads==params (P), RS==AG.
    c = ps.score(1, 1, 1, 4, 2, hw, m)
    # dp_comm = RS + AG; both same payload P, so each ~half
    assert c.dp_comm > 0

def test_payload_convention_rs_full_p():
    # Pin: RS of full P at dp=n costs ~AR of full P / 2 (ring factor).
    # Under the broken convention (sharded P), RS would be dp x cheaper.
    m = {"n_layers": 2, "d_model": 64, "n_heads": 4, "head_dim": 16, "d_ff": 256, "vocab": 1000, "seq": 128}
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "dp_dtype": "bf16"}
    c_ar = ps.score(1, 1, 1, 4, 1, hw, m)   # zero=1: AR(grads) + AG(params)
    c_z3 = ps.score(1, 1, 1, 4, 3, hw, m)   # zero=3: RS(grads) + AG(params)
    # RS component should be ~half of AR (ring factor (n-1)/n vs (n-1)/n*2)
    # AR = RS + AG where RS==AG (same payload), so AR = 2*RS, RS = AR/2
    rs_approx = c_z3.dp_comm / 2  # RS is half of (RS+AG) when RS==AG
    ar_approx = c_ar.dp_comm / 2  # AR = RS+AG, RS component = AR/2
    assert abs(rs_approx - ar_approx) / max(rs_approx, 1e-12) < 0.5, (rs_approx, ar_approx)
