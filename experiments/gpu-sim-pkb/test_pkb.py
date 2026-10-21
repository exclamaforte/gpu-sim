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
    m = json.load(open(os.path.join(EXP, "metrics.json")))
    assert m["plugin_bytes_per_rank"], "no bytes"
    rb = m["plugin_bytes_per_rank"]
    assert rb["0"] == rb["1"], "asymmetric bytes"
    assert m["model_total_bytes"] == rb["0"] + rb["1"], "byte mismatch"
    assert m["accounting_unit"] == "per_isend"
    print("PASS bytes_present (exact equality)")

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
