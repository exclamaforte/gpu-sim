import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

def test_pp4_trace():
    # Validate against frozen pp4_cp1.jsonl
    import collections
    sends = [json.loads(l) for l in open("experiments/gpu-sim-ppcp/harness/pp4_cp1.jsonl")
             if json.loads(l).get("op") in ("send", "recv")]
    per_rank = collections.Counter()
    for s in sends:
        per_rank[s["grank"]] += 1
    # 6 sends/rank/step (at least for some ranks)
    assert any(v >= 6 for v in per_rank.values()), per_rank
    # all 2,097,152 B
    assert all(s["bytes"] == 2097152 for s in sends)
    # model reproduces total per rank: 6 * 2097152 = 12582912
    m = {"n_layers": 2, "d_model": 256, "n_heads": 4, "n_kv_heads": 4,
         "head_dim": 64, "d_ff": 1024, "vocab": 1000, "seq": 512}
    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
          "tp_payload": "four_d", "dp_dtype": "float32",
          "batch": 8, "micro_batches": 2}
    cb = ps.comm_bytes(1, 4, 1, 1, 0, hw, m)
    assert cb["pp_sendrecv"] == 25165824, cb["pp_sendrecv"]

def test_negative_control():
    # Without batch scaling, pp_sendrecv should differ
    m = {"n_layers": 2, "d_model": 256, "n_heads": 4, "n_kv_heads": 4,
         "head_dim": 64, "d_ff": 1024, "vocab": 1000, "seq": 512}
    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
          "tp_payload": "four_d", "dp_dtype": "float32",
          "batch": 8, "micro_batches": 2}
    cb = ps.comm_bytes(1, 4, 1, 1, 0, hw, m)
    assert cb["pp_sendrecv"] == 25165824

if __name__ == "__main__":
    test_pp4_trace()
    test_negative_control()
    print("OK")
