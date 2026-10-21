#!/usr/bin/env python3
"""CPU-only tests for gpu-sim-pkb."""
import os, sys, json, re

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

def _parse_phase_table_p():
    """Parse canonical_byte_table.md to get {S: pieces_per_phase}."""
    path = os.path.join(EXP, "canonical_byte_table.md")
    result = {}
    for line in open(path):
        line = line.strip()
        if not line.startswith("|") or "---" in line or "S (bytes)" in line:
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 6:
            continue
        try:
            S = int(cols[0])
            P = int(cols[5])
            result[S] = P
        except ValueError:
            continue
    return result

def test_chunk_rule_from_phase_table():
    table = _parse_phase_table_p()
    assert 2097152 in table, "2 MiB not in phase table"
    expected_p = table[2097152]
    assert expected_p == 8, "expected P=8 at 2 MiB, got %d" % expected_p
    # model must agree with table
    sys.path.insert(0, os.path.join(REPO, "epic_cluster"))
    from link_model import _derived_pieces_per_phase
    model_p = _derived_pieces_per_phase(2, 2097152)
    assert model_p == expected_p, (
        "model P=%d != table P=%d at 2 MiB" % (model_p, expected_p))
    print("PASS chunk_rule_from_phase_table (table P=%d == model P=%d)" % (expected_p, model_p))

def test_chunk_rule_refusal():
    sys.path.insert(0, os.path.join(REPO, "epic_cluster"))
    from link_model import _derived_pieces_per_phase
    # n>=3 must raise (not measured)
    try:
        _derived_pieces_per_phase(3, 2097152)
        assert False, "n=3 should raise"
    except ValueError:
        pass
    try:
        _derived_pieces_per_phase(4, 2097152)
        assert False, "n=4 should raise"
    except ValueError:
        pass
    # uncovered (n=2, S) must raise
    try:
        _derived_pieces_per_phase(2, 4194304)
        assert False, "uncovered S should raise"
    except ValueError:
        pass
    print("PASS chunk_rule_refusal (n>=3 and uncovered S raise)")

def test_c_python_p_agreement_2mib():
    import subprocess
    NCCL_LIB = "/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
    CUDALIB = "/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib"
    env = {
        "LD_LIBRARY_PATH": NCCL_LIB + ":" + CUDALIB,
        "LD_PRELOAD": os.path.join(REPO, "libfakegpu.so"),
        "GPUSIM_TOPOLOGY": "/tmp/topo_2.bin",
        "GPUSIM_TRACE": "/tmp/trace_2m_c_test.jsonl",
        "FAKEGPU_NUM_DEVICES": "2",
    }
    # ensure topo exists
    if not os.path.exists("/tmp/topo_2.bin"):
        import subprocess as sp
        sp.run([sys.executable, "-c",
            "import sys; sys.path.insert(0,'%s/epic_cluster'); "
            "from topology import emit; "
            "spec={'version':1,'models':{'m':{'sm_count':80,'clock_mhz':1000,'mem_bytes':1073741824,'mem_bw_gbs':1000.0,'arch':'x','source':'t'}},"
            "'nodes':[{'name':'n0','devices':[{'id':0,'model':'m'}]},{'name':'n1','devices':[{'id':0,'model':'m'}]}],"
            "'links':[{'a':{'node':'n0','device':0},'b':{'node':'n1','device':0},'bw_gbps':10.0,'latency_us':5.0,'mtu':4096,'loss_ppm':0,'jitter_us':0.0,'shared_group':None,'kind':'ethernet'}],"
            "'ranks':[{'rank':0,'node':'n0','device':0},{'rank':1,'node':'n1','device':0}]}; "
            "open('/tmp/topo_2.bin','wb').write(emit(spec))" % REPO],
            check=True)
    tf = env["GPUSIM_TRACE"]
    if os.path.exists(tf):
        os.remove(tf)
    r = subprocess.run([os.path.join(REPO, "build", "ring_allreduce"), "524288"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    # parse trace
    import json as _j
    recs = [_j.loads(l) for l in open(tf) if l.strip()]
    assert recs, "no trace records"
    c_pred = recs[0]["predicted_s"]
    # Python model
    sys.path.insert(0, os.path.join(REPO, "epic_cluster"))
    from link_model import predict
    spec = {
      "version": 1,
      "models": {"m": {"sm_count": 80, "clock_mhz": 1000, "mem_bytes": 1073741824, "mem_bw_gbs": 1000.0, "arch": "x", "source": "t"}},
      "nodes": [{"name": "n0", "devices": [{"id": 0, "model": "m"}]}, {"name": "n1", "devices": [{"id": 0, "model": "m"}]}],
      "links": [{"a": {"node": "n0", "device": 0}, "b": {"node": "n1", "device": 0}, "bw_gbps": 10.0, "latency_us": 5.0, "mtu": 4096, "loss_ppm": 0, "jitter_us": 0.0, "shared_group": None, "kind": "ethernet"}],
      "ranks": [{"rank": 0, "node": "n0", "device": 0}, {"rank": 1, "node": "n1", "device": 0}],
    }
    py_pred = predict(spec, "ring_allreduce", 524288, "float32", n_ranks=2)
    py_p = py_pred["pieces_per_phase"]
    assert py_p == 8, "Python P=%d, expected 8" % py_p
    assert abs(c_pred - py_pred["duration_s"]) < 1e-12, (
        "C/Python mismatch: C=%s Python=%s" % (c_pred, py_pred["duration_s"]))
    print("PASS c_python_p_agreement_2mib (C=%.10f == Python=%.10f, P=8)" % (c_pred, py_pred["duration_s"]))

def test_negative_control_missing_link():
    import subprocess, os, time
    EXP = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.abspath(os.path.join(EXP, "..", ".."))
    M8 = os.path.join(REPO, "experiments", "gpu-sim-m3d.8")
    NCCL_LIB = "/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
    CUDALIB = "/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib"
    env_base = {
        "LD_LIBRARY_PATH": NCCL_LIB + ":" + CUDALIB,
        "NCCL_NET_PLUGIN": os.path.join(M8, "build", "libncclNetSimFab.so"),
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_DEBUG": "WARN",
    }
    for topo_name, expected_msg in [("topo_missing.bin", "no link"),
                                     ("topo_zerobw.bin", "zero-bandwidth link")]:
        topo_path = os.path.join(EXP, topo_name)
        assert os.path.exists(topo_path), topo_path
        uidfile = "/tmp/pkb-neg-uid"
        for f in [uidfile]:
            try: os.remove(f)
            except FileNotFoundError: pass
        env = dict(env_base)
        env["GPUSIM_TOPOLOGY"] = topo_path
        env["GPUSIM_NRANKS"] = "2"
        env["GPUSIM_RUN_ID"] = "pkb-negtest"
        procs = []
        for i in [0, 1]:
            e = dict(env)
            e["NCCL_HOSTID"] = "node%d" % i
            e["GPUSIM_RANK"] = str(i)
            p = subprocess.Popen(
                [os.path.join(M8, "build", "allreduce_test"), str(i), "2", uidfile],
                env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            procs.append(p)
        # wait for both
        outs = []
        for p in procs:
            try:
                o, _ = p.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                o, _ = p.communicate()
            outs.append(o)
        combined = "\n".join(outs)
        # The plugin must fail loudly: named error string present, no silent OK.
        assert expected_msg in combined, (
            "%s: expected '%s' in output. Got: %s" % (topo_name, expected_msg, combined[:300]))
        assert "allreduce OK" not in combined, (
            "%s: unexpected 'allreduce OK' (silent fallback)" % topo_name)
        # assert named error string visible
        assert expected_msg in combined, (
            "%s: expected '%s' in output, got: %s" % (topo_name, expected_msg, combined[:500]))
        # assert no 'allreduce OK' (no silent fallback)
        assert "allreduce OK" not in combined, (
            "%s: unexpected 'allreduce OK' (silent fallback)" % topo_name)
        print("PASS negative_control_%s (exit=%s, msg='%s' present, no OK)" % (
            topo_name, [p.returncode for p in procs], expected_msg))
    print("PASS negative_control_missing_link (executable)")

if __name__ == "__main__":
    test_fabric_transfer_timing()
    test_trace_has_acct_field()
    test_item10_plugin_loaded()
    test_bytes_present()
    test_chunk_rule_from_phase_table()
    test_chunk_rule_refusal()
    test_c_python_p_agreement_2mib()
    test_header_guard()
    test_build_no_undef()
    test_negative_control_missing_link()
    print("all pkb CPU tests passed")
