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
    # zero=3 > zero=2 (adds AG)
    assert c3.dp_comm > c2.dp_comm, (c3.dp_comm, c2.dp_comm)

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
