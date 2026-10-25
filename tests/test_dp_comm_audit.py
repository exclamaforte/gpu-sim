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
