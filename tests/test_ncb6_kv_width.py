import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

def test_cp_uses_kv_width():
    # CP msg_bytes should use n_kv_heads*head_dim, not d_model
    m_gqa = {"n_layers": 2, "d_model": 4096, "n_heads": 32, "n_kv_heads": 8,
             "head_dim": 128, "d_ff": 14336, "vocab": 128256, "seq": 8192}
    m_mha = {"n_layers": 2, "d_model": 4096, "n_heads": 32, "n_kv_heads": 32,
             "head_dim": 128, "d_ff": 14336, "vocab": 128256, "seq": 8192}
    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "tp_payload": "four_d"}
    c_gqa = ps.score(1, 1, 2, 1, 0, hw, m_gqa)
    c_mha = ps.score(1, 1, 2, 1, 0, hw, m_mha)
    # GQA KV width = 1024, MHA = 4096, so GQA cp_comm should be ~4x smaller
    assert c_gqa.cp_comm < c_mha.cp_comm, (c_gqa.cp_comm, c_mha.cp_comm)
    assert c_gqa.cp_comm * 4 > c_mha.cp_comm * 0.9, "GQA should be ~4x smaller"

def test_comm_bytes_kv_width():
    m = {"n_layers": 2, "d_model": 4096, "n_heads": 32, "n_kv_heads": 8,
         "head_dim": 128, "d_ff": 14336, "vocab": 128256, "seq": 8192}
    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "tp_payload": "four_d"}
    cb = ps.comm_bytes(1, 1, 2, 1, 0, hw, m)
    # cp_exchange should use 1024 width, not 4096
    assert cb["cp_exchange"] > 0
    # 4x smaller than d_model would give
    m2 = dict(m); m2["n_kv_heads"] = 32
    cb2 = ps.comm_bytes(1, 1, 2, 1, 0, hw, m2)
    assert cb["cp_exchange"] < cb2["cp_exchange"]

if __name__ == "__main__":
    test_cp_uses_kv_width()
    test_comm_bytes_kv_width()
    print("OK")
