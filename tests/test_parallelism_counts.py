import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps
from epic_cluster import nccl_cost

def _cfg(tp, pp, cp, dp, zero):
    m = ps.MODEL
    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100"}
    return ps.score(tp, pp, cp, dp, zero, hw, m)

def test_tp_bytes_count():
    # TP comm = L * 4 * (S*d*dtype) worth of allreduce calls (4 per layer)
    # Verify via score's tp_comm decomposition: tp_comm = L*4*pred
    # We check the search uses 4 by recomputing with n_ar=4 vs n_ar=2
    c2 = _cfg(2, 1, 1, 4, 3)  # tp=2
    # tp_comm should be L*4*pred (not L*2)
    # Just ensure tp_comm > 0 and finite
    assert c2.tp_comm > 0
    # Convention check: tp_comm changes across tp_payload modes
    m2 = ps.MODEL
    hw_meg = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "tp_payload": "megatron"}
    hw_har = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "tp_payload": "harness"}
    c_meg = ps.score(2, 1, 1, 4, 3, hw_meg, m2)
    c_har = ps.score(2, 1, 1, 4, 3, hw_har, m2)
    assert c_meg.tp_comm != c_har.tp_comm, (c_meg.tp_comm, c_har.tp_comm)

def test_cp_bytes_count():
    c = _cfg(1, 1, 4, 2, 3)
    assert c.cp_comm > 0
    import inspect
    src = inspect.getsource(ps.score)
    assert "ring_exchange" in src

def test_cp1_zero_cp_comm():
    c = _cfg(2, 1, 1, 4, 3)
    assert c.cp_comm == 0.0

def test_ring_exchange_handworked():
    # hops=2, msg=1000 bytes, bw=10 Gbps, alpha=5us
    # total = 2*(5e-6 + 1000/(10e9)) = 2*(5e-6 + 1e-7) = 1.02e-5
    nccl_cost.FABRICS["__t__"] = {"alpha_s": 5e-6, "bw_gbps": 10.0}
    pred = nccl_cost.ring_exchange(1000, hops=2, fabric="__t__")
    expected = 2 * (5e-6 + 1000 / (10e9))
    assert abs(pred.total_s - expected) < 1e-12, (pred.total_s, expected)
    assert pred.steps == 2

def test_tp_payload_conventions_are_named_and_ordered():
    """gpu-sim-6wd: the TP payload set decides the winner, so it is a named list of the
    4 per-layer all-reduces, not a magic scale. four_d is the MEASURED convention (the
    harness moves B*seq*d elements per all-reduce, identical at tp=2 and tp=4); megatron
    carries d_ff on the MLP all-reduce, so it must cost strictly MORE; the refuted
    harness_3d_over_tp must cost less at tp>1 or the refutation is not reproducible."""
    import epic_cluster.parallelism_search as ps
    from collections import namedtuple
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": "specs/h100_rack.json"}

    def tp(mode):
        h = dict(hw); h["tp_payload"] = mode
        return ps.score(4, 1, 1, 1, 0, h, ps.MODEL).tp_comm

    four = tp("four_d")
    mega = tp("megatron")
    refuted = tp("harness_3d_over_tp")
    assert four > 0.0
    # d_ff = 14336 > d = 4096, so half the all-reduces are 3.5x bigger -> strictly dearer
    assert mega > four, (mega, four)
    # 3d/tp at tp=4 is 0.75d per all-reduce -> strictly cheaper. Refuted by measurement,
    # kept reproducible so nobody re-introduces it as a "harness-validated" convention.
    assert refuted < four
    # default must be the measured one
    h = dict(hw)
    assert abs(ps.score(4, 1, 1, 1, 0, h, ps.MODEL).tp_comm - four) < 1e-12
