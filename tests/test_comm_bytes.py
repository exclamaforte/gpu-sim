import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

def test_comm_bytes_returns_dict():
    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "tp_payload": "four_d"}
    m = ps.MODEL
    r = ps.comm_bytes(2, 1, 2, 4, 3, hw, m)
    assert isinstance(r, dict)
    assert "total" in r
    assert r["total"] > 0
    assert r["tp_allreduce"] > 0

def test_comm_bytes_zero_traffic():
    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "tp_payload": "four_d"}
    m = ps.MODEL
    r = ps.comm_bytes(1, 1, 1, 1, 0, hw, m)
    assert r["tp_allreduce"] == 0
    assert r["cp_exchange"] == 0
    assert r["pp_sendrecv"] == 0

if __name__ == "__main__":
    test_comm_bytes_returns_dict()
    test_comm_bytes_zero_traffic()
    print("OK")
