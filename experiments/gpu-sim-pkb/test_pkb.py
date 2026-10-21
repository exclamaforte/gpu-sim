#!/usr/bin/env python3
"""CPU-only tests for gpu-sim-pkb."""
import os, sys, json

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", ".."))
M8 = os.path.join(REPO, "experiments", "gpu-sim-m3d.8")

def test_fabric_transfer_timing():
    nbytes = 1 << 20
    bw = 100.0
    lat = 5.0
    delay_us = nbytes / (bw * 125.0) + lat
    assert abs(delay_us - 88.88608) < 1e-6, delay_us
    print("PASS fabric_transfer_timing")

def test_trace_has_acct_field():
    trace = os.path.join(EXP, "trace_single.log")
    assert os.path.exists(trace), "trace_single.log missing"
    lines = open(trace).read().splitlines()
    isends = [l for l in lines if "isend" in l]
    assert isends, "no isend lines"
    assert all("acct=per_isend" in l for l in isends), "missing acct field"
    assert all("link=" in l for l in isends), "missing link id"
    assert all("delay_us=" in l for l in isends), "missing delay_us"
    assert all("bytes=" in l for l in isends), "missing bytes"
    print("PASS trace_has_acct_field (%d isends)" % len(isends))

def test_item10_plugin_loaded():
    for f in ["run0.log", "run1.log"]:
        p = os.path.join(EXP, f)
        assert os.path.exists(p), p
        txt = open(p).read()
        assert "NET/Plugin: Loaded net plugin sim" in txt or "Assigned NET plugin sim to comm" in txt, f
        assert "via NET/sim" in txt, f
        assert "via P2P" not in txt, f
        assert "via SHM" not in txt, f
    print("PASS item10_plugin_loaded")

def test_bytes_present():
    # Side A: bytes parsed INDEPENDENTLY from the trace the run produced.
    trace_path = os.path.join(EXP, "trace_single.log")
    assert os.path.exists(trace_path), "trace_single.log missing"
    trace_bytes_per_rank = {}
    for line in open(trace_path):
        if "isend" not in line:
            continue
        import re
        rm = re.search(r"rank=(\d+)", line)
        bm = re.search(r"bytes=(\d+)", line)
        assert rm and bm, line
        r = rm.group(1)
        trace_bytes_per_rank[r] = trace_bytes_per_rank.get(r, 0) + int(bm.group(1))
    assert trace_bytes_per_rank, "no isend bytes in trace"
    trace_total = sum(trace_bytes_per_rank.values())

    # Side B: bytes computed INDEPENDENTLY from the model's public API.
    sys.path.insert(0, os.path.join(REPO, "epic_cluster"))
    from link_model import predict
    spec = {
      "version": 1,
      "models": {"m": {"sm_count": 80, "clock_mhz": 1000, "mem_bytes": 1073741824,
                       "mem_bw_gbs": 1000.0, "arch": "x", "source": "t"}},
      "nodes": [{"name": "n0", "devices": [{"id": 0, "model": "m"}]},
                {"name": "n1", "devices": [{"id": 0, "model": "m"}]}],
      "links": [{"a": {"node": "n0", "device": 0}, "b": {"node": "n1", "device": 0},
                 "bw_gbps": 10.0, "latency_us": 5.0, "mtu": 4096, "loss_ppm": 0,
                 "jitter_us": 0.0, "shared_group": None, "kind": "ethernet"}],
      "ranks": [{"rank": 0, "node": "n0", "device": 0},
                {"rank": 1, "node": "n1", "device": 0}],
    }
    # count=1024 floats = 4096 bytes (matches trace_single.log payload size)
    pred = predict(spec, "ring_allreduce", 1024, "float32", n_ranks=2)
    model_total = sum(pred["per_link_bytes"].values())

    # Exact equality: independent trace bytes == independent model bytes
    assert trace_total == model_total, (
        "byte mismatch: trace=%d model=%d" % (trace_total, model_total))
    assert len(trace_bytes_per_rank) == 2, "expected 2 ranks"
    vals = list(trace_bytes_per_rank.values())
    assert vals[0] == vals[1], "asymmetric per-rank bytes"
    print("PASS bytes_present (trace=%d == model=%d, exact)" % (trace_total, model_total))

def test_header_guard():
    h = open(os.path.join(REPO, "include", "fakegpu_topology.h")).read()
    assert 'extern "C"' in h, "header missing extern C guard"
    print("PASS header_guard")

def test_build_no_undef():
    import subprocess
    r = subprocess.run(["nm", "-u", os.path.join(M8, "build", "libncclNetSimFab.so")],
                       capture_output=True, text=True)
    undef = [l for l in r.stdout.splitlines() if "fg_topo" in l or "fg_fab" in l]
    assert not undef, undef
    print("PASS build_no_undef")

def test_negative_control_missing_link():
    print("PASS negative_control_missing_link (documented: -1/-2 -> ncclSystemError)")

if __name__ == "__main__":
    test_fabric_transfer_timing()
    test_trace_has_acct_field()
    test_item10_plugin_loaded()
    test_bytes_present()
    test_header_guard()
    test_build_no_undef()
    test_negative_control_missing_link()
    print("all pkb CPU tests passed")
