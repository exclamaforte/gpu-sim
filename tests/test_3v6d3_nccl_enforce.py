"""Guard: ncclAllReduce enforces through the fabric calendar (gpu-sim-3v6d.3)."""
import ctypes, json, os, tempfile, statistics, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "libfakenccl.so"
TOPO = REPO / "experiments" / "gpu-sim-pkb" / "topo_single.bin"

SHAPES = [1 << s for s in range(20, 28)]  # 8 shapes
BUSY_OFF = 64  # byte offset of busy_until[0] in fg_header_t

def _run_once(N, nrank=2):
    if not LIB.exists():
        import subprocess
        subprocess.run(["make"], cwd=str(REPO), check=True)
    lib = ctypes.CDLL(str(LIB))
    os.environ["GPUSIM_TOPOLOGY"] = str(TOPO)
    os.environ["GPUSIM_NCCL_LEGACY_TABLE"] = "1"
    os.environ["GPUSIM_ALLOW_EXTRAPOLATE"] = "1"
    import uuid
    rid = f"corpus-{os.getpid()}-{N}-{uuid.uuid4().hex[:8]}"
    os.environ["GPUSIM_FABRIC_RUN_ID"] = rid
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    tf.close()
    os.environ["GPUSIM_TRACE"] = tf.name
    lib.ncclAllReduce.restype = ctypes.c_int
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    send = (ctypes.c_float * N)()
    recv = (ctypes.c_float * N)()
    # create a comm with nrank ranks
    comm_arr = (ctypes.c_void_p * nrank)()
    lib.ncclCommInitAll(comm_arr, nrank, None)
    comm = comm_arr[0]
    rc = lib.ncclAllReduce(send, recv, N, 7, 0, comm, None)
    assert rc == 0
    lines = [l for l in open(tf.name) if l.strip()]
    assert lines, "no trace written"
    rec = json.loads(lines[-1])
    assert "predicted_s" in rec and "enforced_s" in rec and "grank" in rec
    pred = rec["predicted_s"]; enf = rec["enforced_s"]
    assert pred is not None
    assert enf is not None, "enforced_s null (transfer did not run)"
    delta = abs(enf - pred) / pred * 100.0
    assert delta <= 5.0, f"delta {delta:.2f}% > 5% (N={N}, nr={nrank})"
    # the per-sample delta for this call; the test function asserts EVERY one
    # reclaim own segment
    libc = ctypes.CDLL('libc.so.6')
    _ur = libc.shm_unlink(b'/gpusim-' + rid.encode())
    try:
        os.unlink(tf.name)
    except OSError:
        pass
    return delta

def test_allreduce_trace_has_enforced_within_5pct():
    NRANKS = [2, 4, 8]
    for N in SHAPES:
        for nr in NRANKS:
            samples = [_run_once(N, nr) for _ in range(3)]
            # EVERY sample must be within the bound, and the failure names the
            # sample and its value. A median-only assert let one sample exceed
            # 5% unnoticed for three commits (gpu-sim-3v6d.3 rework).
            for i, s in enumerate(samples):
                assert s <= 5.0, (
                    f"sample {i} of {len(samples)} delta {s:.2f}% > 5% "
                    f"(N={N}, nr={nr}, samples={[round(x, 2) for x in samples]})")
            med = statistics.median(samples)
            assert med <= 5.0, f"median delta {med:.2f}% > 5% (N={N}, nr={nr})"

def test_contention_shared_link():
    """Contention: pre-set busy_until so second collective must wait."""
    if not LIB.exists():
        import subprocess
        subprocess.run(["make"], cwd=str(REPO), check=True)
    lib = ctypes.CDLL(str(LIB))
    os.environ["GPUSIM_TOPOLOGY"] = str(TOPO)
    os.environ["GPUSIM_NCCL_LEGACY_TABLE"] = "1"
    os.environ["GPUSIM_ALLOW_EXTRAPOLATE"] = "1"
    rid = "shared-contention-test"
    os.environ["GPUSIM_FABRIC_RUN_ID"] = rid
    N = 1 << 20
    libc = ctypes.CDLL('libc.so.6')
    lib.ncclAllReduce.restype = ctypes.c_int
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    # first collective
    tf1 = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    tf1.close()
    os.environ["GPUSIM_TRACE"] = tf1.name
    s1 = (ctypes.c_float * N)(); r1 = (ctypes.c_float * N)()
    rc1 = lib.ncclAllReduce(s1, r1, N, 7, 0, None, None)
    assert rc1 == 0
    lines1 = [l for l in open(tf1.name) if l.strip()]
    rec1 = json.loads(lines1[-1])
    enf1 = rec1["enforced_s"]
    assert enf1 is not None and enf1 > 0
    # simulate concurrent transfer: push busy_until[0] far into the future
    fd = libc.shm_open(b'/gpusim-' + rid.encode(), 2, 0o666)
    assert fd >= 0
    libc.mmap.restype = ctypes.c_void_p
    ptr = libc.mmap(None, 4096, 3, 1, fd, 0)
    ctypes.c_uint64.from_address(ptr + BUSY_OFF).value = int(time.monotonic_ns()) + 100_000_000
    libc.munmap(ptr, 4096)
    libc.close(fd)
    # second collective: must wait for busy_until
    tf2 = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    tf2.close()
    os.environ["GPUSIM_TRACE"] = tf2.name
    s2 = (ctypes.c_float * N)(); r2 = (ctypes.c_float * N)()
    t0 = time.perf_counter()
    rc2 = lib.ncclAllReduce(s2, r2, N, 7, 0, None, None)
    wall2 = time.perf_counter() - t0
    assert rc2 == 0
    lines2 = [l for l in open(tf2.name) if l.strip()]
    rec2 = json.loads(lines2[-1])
    enf2 = rec2["enforced_s"]
    assert enf2 is not None and enf2 > 0
    # contention: second waited ~0.1s on the pre-set busy_until, so wall time >= 0.05
    assert wall2 >= 0.05, f"contention not visible: wall2={wall2}, enf1={enf1}, enf2={enf2}"
    libc.shm_unlink(b'/gpusim-' + rid.encode())
    for tf in (tf1, tf2):
        try:
            os.unlink(tf.name)
        except OSError:
            pass
