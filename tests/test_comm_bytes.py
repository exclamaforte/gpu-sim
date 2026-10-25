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



def test_comm_bytes_dtype_negative_control():
    hw2 = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
           "tp_payload": "four_d", "dp_dtype": "float32"}
    hw1 = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
           "tp_payload": "four_d", "dp_dtype": "bf16"}
    m = ps.MODEL
    r1 = ps.comm_bytes(2, 1, 2, 4, 3, hw1, m)
    r2 = ps.comm_bytes(2, 1, 2, 4, 3, hw2, m)
    assert r1["dp_reducescatter"] != r2["dp_reducescatter"], "dtype change must affect bytes"


def test_comm_bytes_tp_payload_negative_control():
    hw1 = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
           "tp_payload": "four_d", "dp_dtype": "bf16"}
    hw2 = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
           "tp_payload": "megatron", "dp_dtype": "bf16"}
    m = ps.MODEL
    r1 = ps.comm_bytes(2, 1, 2, 4, 3, hw1, m)
    r2 = ps.comm_bytes(2, 1, 2, 4, 3, hw2, m)
    assert r1["tp_allreduce"] != r2["tp_allreduce"], "tp_payload change must affect bytes"

if __name__ == "__main__":
    test_comm_bytes_returns_dict()
    test_comm_bytes_zero_traffic()
    test_comm_bytes_dtype_negative_control()
    test_comm_bytes_tp_payload_negative_control()
    print("OK")
