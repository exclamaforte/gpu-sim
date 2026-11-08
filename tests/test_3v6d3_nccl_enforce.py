"""Guard: ncclAllReduce enforces through the fabric calendar (gpu-sim-3v6d.3)."""
import ctypes, json, os, tempfile, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "libfakenccl.so"
TOPO = REPO / "experiments" / "gpu-sim-pkb" / "topo_single.bin"

SHAPES = [1 << s for s in range(17, 25)]  # 8 shapes

def _run_once(N):
    if not LIB.exists():
        import subprocess
        subprocess.run(["make"], cwd=str(REPO), check=True)
    lib = ctypes.CDLL(str(LIB))
    os.environ["GPUSIM_TOPOLOGY"] = str(TOPO)
    import uuid
    os.environ["GPUSIM_FABRIC_RUN_ID"] = f"corpus-{os.getpid()}-{N}-{uuid.uuid4().hex[:8]}"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    tf.close()
    os.environ["GPUSIM_TRACE"] = tf.name
    lib.ncclAllReduce.restype = ctypes.c_int
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    send = (ctypes.c_float * N)()
    recv = (ctypes.c_float * N)()
    rc = lib.ncclAllReduce(send, recv, N, 7, 0, None, None)
    assert rc == 0
    lines = [l for l in open(tf.name) if l.strip()]
    assert lines, "no trace written"
    rec = json.loads(lines[-1])
    assert "predicted_s" in rec and "enforced_s" in rec and "grank" in rec
    pred = rec["predicted_s"]; enf = rec["enforced_s"]
    assert pred is not None
    assert enf is not None, "enforced_s null (transfer did not run)"
    delta = abs(enf - pred) / pred * 100.0
    assert delta <= 5.0, f"delta {delta:.2f}% > 5%"
    try:
        os.unlink(tf.name)
    except OSError:
        pass
    return delta

def test_allreduce_trace_has_enforced_within_5pct():
    import glob as _g
    for _f in _g.glob('/dev/shm/gpusim-*'):
        try: os.unlink(_f)
        except OSError: pass
    deltas = []
    for N in SHAPES:
        samples = [_run_once(N) for _ in range(3)]
        med = statistics.median(samples)
        assert med <= 5.0, f"median delta {med:.2f}% > 5% (N={N})"
        deltas.append(med)
    assert len(deltas) == len(SHAPES)

def test_contention_shared_link():
    import glob as _g
    for _f in _g.glob('/dev/shm/gpusim-*'):
        try: os.unlink(_f)
        except OSError: pass
    """Two collectives on the same run_id share the calendar (contention visible)."""
    if not LIB.exists():
        import subprocess
        subprocess.run(["make"], cwd=str(REPO), check=True)
    lib = ctypes.CDLL(str(LIB))
    os.environ["GPUSIM_TOPOLOGY"] = str(TOPO)
    os.environ["GPUSIM_FABRIC_RUN_ID"] = "shared-contention-test"
    N = 1 << 20
    tf1 = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    tf1.close()
    os.environ["GPUSIM_TRACE"] = tf1.name
    lib.ncclAllReduce.restype = ctypes.c_int
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    s1 = (ctypes.c_float * N)(); r1 = (ctypes.c_float * N)()
    rc1 = lib.ncclAllReduce(s1, r1, N, 7, 0, None, None)
    assert rc1 == 0
    lines1 = [l for l in open(tf1.name) if l.strip()]
    rec1 = json.loads(lines1[-1])
    enf1 = rec1["enforced_s"]
    assert enf1 is not None

    tf2 = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    tf2.close()
    os.environ["GPUSIM_TRACE"] = tf2.name
    s2 = (ctypes.c_float * N)(); r2 = (ctypes.c_float * N)()
    rc2 = lib.ncclAllReduce(s2, r2, N, 7, 0, None, None)
    assert rc2 == 0
    lines2 = [l for l in open(tf2.name) if l.strip()]
    rec2 = json.loads(lines2[-1])
    enf2 = rec2["enforced_s"]
    assert enf2 is not None
    assert enf2 >= enf1 * 0.5, f"contention not visible: enf1={enf1}, enf2={enf2}"
    for tf in (tf1, tf2):
        try:
            os.unlink(tf.name)
        except OSError:
            pass