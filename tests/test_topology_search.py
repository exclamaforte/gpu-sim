import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import link_model as lm

SPEC = "specs/h100_rack.json"

def test_tp8_nvlink_dp2_ib():
    spec = json.load(open(SPEC))
    tp_group = list(range(8))
    r = lm.predict(spec, "ring_allreduce", 1000, "float32", ranks=tp_group, pieces_per_phase=1)
    assert r["bottleneck"]["kind"] == "nvlink", r["bottleneck"]
    dp_group = [0, 8]
    r2 = lm.predict(spec, "ring_allreduce", 1000, "float32", ranks=dp_group, pieces_per_phase=1)
    assert r2["bottleneck"]["kind"] == "ib", r2["bottleneck"]

def test_cross_node_magnitude():
    # hand-worked: 1000 bytes over NDR IB 50 GB/s, 1 hop (one link) alpha 5 us
    # S/B = 1000/(50e9) = 2e-8 s; alpha = 5e-6 s; total = 5.02e-6 s
    spec = json.load(open(SPEC))
    val, kind = lm.point_to_point(spec, 0, 8, 1000)
    assert abs(val - 5.02e-6) < 1e-9, val
    assert kind == 'ib'

def test_refusal_unplaceable():
    spec = json.load(open(SPEC))
    try:
        lm.predict(spec, "ring_allreduce", 1000, "float32", ranks=[0, 999], pieces_per_phase=1)
    except ValueError:
        return
    raise AssertionError("expected refusal")

def test_legacy_signature_still_works():
    spec = json.load(open(SPEC))
    r = lm.predict(spec, "ring_allreduce", 1000, "float32", n_ranks=2, pieces_per_phase=1)
    assert "duration_s" in r
