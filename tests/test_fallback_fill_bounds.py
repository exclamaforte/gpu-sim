"""The interposer must never write outside a collective's recv buffer.

gpu-sim-6wd-bytes: src/fakegpu_nccl.c's fallback path (no topology loaded) computed

    bytes = payload * (1 + floor(log2 nrank) / 2)      # "model fan-in/fan-out cost"

and then handed that INFLATED count to fg_nccl_out_fill(), i.e. wrote 1.5x the buffer at
n=2, 2x at n=4 and 2.5x at n=8. Reproduced at n=4 as glibc "malloc(): corrupted top
size" -- heap corruption, not modelling.

Consequences that cost this lab real time before the byte count was understood:
  * the trace `bytes` column carried the inflated factor, which is the whole explanation
    of the "unexplained DP byte accounting" (zero=3 all-gather 32.27 MB at dp=2 =
    21.51 MB payload x 1.5; 43.03 at dp=4 = x 2.0) -- the all-gather total is
    dp-independent, the inflation was not;
  * dp=8 runs died part-way (5-6 of 8 ranks traced) and were written off as a host
    limitation; with the overflow fixed all 8 ranks report.

This test pins both halves: the trace byte count equals the payload, and nothing is
written past the end of the recv buffer.
"""
import ctypes
import json
import os
import pathlib
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = ROOT / "libfakenccl.so"

NCCL_FLOAT32 = 7
COUNT = 1000
PAYLOAD = COUNT * 4          # float32
GUARD = 256                  # bytes of canary past the end of the buffer


def _load():
    if not LIB.exists():
        pytest.skip("libfakenccl.so not built (run `make`)")
    lib = ctypes.CDLL(str(LIB))
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.c_int, ctypes.c_int, ctypes.c_ulonglong]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p]
    lib.ncclAllReduce.restype = ctypes.c_int
    return lib


def _run(lib, nranks, tmpdir):
    trace = os.path.join(tmpdir, f"n{nranks}.jsonl")
    os.environ["GPUSIM_TRACE"] = trace
    os.environ["FAKEGPU_NCCL_NO_FILL"] = ""      # fill ON: that is what writes
    os.environ.pop("GPUSIM_TOPOLOGY", None)      # fallback path
    comm = ctypes.c_void_p()
    assert lib.ncclCommInitRank(ctypes.byref(comm), nranks, 0, 0) == 0
    buf = (ctypes.c_char * (PAYLOAD + GUARD))()
    ctypes.memset(buf, 0, PAYLOAD + GUARD)
    assert lib.ncclAllReduce(buf, buf, COUNT, NCCL_FLOAT32, 0, comm, None) == 0
    past = bytes(buf[PAYLOAD:])
    rows = [json.loads(line) for line in open(trace) if line.strip()]
    return rows[-1], sum(1 for b in past if b != 0)


@pytest.mark.parametrize("nranks", [2, 4, 8])
def test_fallback_collective_writes_only_the_payload(nranks):
    lib = _load()
    with tempfile.TemporaryDirectory() as tmpdir:
        row, past_end = _run(lib, nranks, tmpdir)
    assert row["bytes"] == PAYLOAD, (
        f"n={nranks}: trace bytes {row['bytes']} != payload {PAYLOAD}. The byte column "
        "must carry the payload; a fan-in factor here both corrupts the buffer below "
        "and re-introduces the dp=2/dp=4 byte inflation that took a day to explain.")
    assert past_end == 0, (
        f"n={nranks}: {past_end} bytes written past the end of the recv buffer -- "
        "the out-of-bounds fill that corrupted the heap (gpu-sim-6wd-bytes)")

@pytest.mark.parametrize("nranks", [2, 4, 8])
def test_fallback_reducescatter_reports_the_total(nranks):
    """gpu-sim-smt6: `bytes` must be the payload total on both paths. NCCL's
    ncclReduceScatter takes recvcount, so the fallback path saw the SHARD and recorded
    count*elem = P/dp while the topology path records rc*nr*elem = P. Measured effect
    before the fix: zero=3 read all-gather 21.51 MB but reduce-scatter 5.38 MB at dp=4
    and 2.69 MB at dp=8 -- a spurious dp-dependence the DP audit then had to explain."""
    lib = _load()
    lib.ncclReduceScatter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p]
    lib.ncclReduceScatter.restype = ctypes.c_int
    shard = 250                      # per-rank recv count; payload total = shard * nranks
    total = shard * nranks
    with tempfile.TemporaryDirectory() as tmpdir:
        trace = os.path.join(tmpdir, "rs.jsonl")
        os.environ["GPUSIM_TRACE"] = trace
        os.environ["FAKEGPU_NCCL_NO_FILL"] = ""
        os.environ.pop("GPUSIM_TOPOLOGY", None)
        comm = ctypes.c_void_p()
        assert lib.ncclCommInitRank(ctypes.byref(comm), nranks, 0, 0) == 0
        sendbuf = (ctypes.c_char * (total * 4))()
        recvbuf = (ctypes.c_char * (shard * 4 + GUARD))()
        ctypes.memset(recvbuf, 0, shard * 4 + GUARD)
        assert lib.ncclReduceScatter(sendbuf, recvbuf, shard, NCCL_FLOAT32, 0,
                                     comm, None) == 0
        row = [json.loads(l) for l in open(trace) if l.strip()][-1]
    assert row["bytes"] == total * 4, (
        f"n={nranks}: reduce-scatter bytes {row['bytes']} != total payload {total * 4} "
        "-- the fallback recorded the shard (gpu-sim-smt6)")

def test_point_to_point_fills_the_whole_payload():
    """gpu-sim-smt7: the p2p sites passed an ELEMENT count to APIs that take BYTES.
    fg_simulate_memcpy() divides by bandwidth, so every PP/CP transfer was charged
    1/4 of its real time at fp32, and ncclRecv's fill wrote `count` bytes into a
    count*elem byte buffer -- a quarter of it. Same count-vs-bytes species as the
    three DP-trace byte bugs; p2p had never been audited because it moves no group
    total. This pins the fill: every payload byte must be written, none past the end."""
    lib = _load()
    lib.ncclRecv.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                             ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclRecv.restype = ctypes.c_int
    count = 1024                       # elements
    payload = count * 4                # float32
    with tempfile.TemporaryDirectory() as tmpdir:
        trace = os.path.join(tmpdir, "p2p.jsonl")
        os.environ["GPUSIM_TRACE"] = trace
        os.environ["FAKEGPU_NCCL_NO_FILL"] = ""      # fill ON
        os.environ.pop("GPUSIM_TOPOLOGY", None)
        comm = ctypes.c_void_p()
        assert lib.ncclCommInitRank(ctypes.byref(comm), 2, 0, 0) == 0
        buf = (ctypes.c_char * (payload + GUARD))()
        ctypes.memset(buf, 0xA5, payload + GUARD)     # canary everywhere
        assert lib.ncclRecv(buf, count, NCCL_FLOAT32, 1, comm, None) == 0
        body = bytes(buf[:payload])
        past = bytes(buf[payload:])
        row = [json.loads(l) for l in open(trace) if l.strip()][-1]
    assert row["bytes"] == payload, (
        f"recv trace bytes {row['bytes']} != payload {payload}")
    # fg_fill() writes PSEUDO-RANDOM bytes, so "this byte != canary" is probabilistic:
    # with a 16-byte tail one run in ~40 leaves all 16 unchanged by chance. Assert on
    # the FRACTION of surviving canary instead (a partial fill leaves ~75% of the
    # buffer untouched; a complete one leaves ~0.4% by chance).
    survived = body.count(b"\xa5")
    assert survived < payload // 4, (
        f"{survived}/{payload} payload bytes still hold the canary -- the fill used "
        "the ELEMENT count as a BYTE count (gpu-sim-smt7)")
    assert any(b != 0xA5 for b in body[-16:]), (
        "the tail of the payload was never written -- fill stopped early")
    assert all(b == 0xA5 for b in past), "recv wrote past the end of the buffer"

@pytest.mark.parametrize("nranks", [2, 4])
def test_alltoall_totals_and_fill_cover_every_peer(nranks):
    """gpu-sim-smt7: NCCL's ncclAllToAll `count` is the PER-PEER send count, so a rank
    receives count*nranks elements. The shim recorded and filled count*elem -- 1/nranks
    of the buffer, the same shard-as-total error as the reduce-scatter. No current
    workload uses all-to-all (no MoE), so this guard is preventive."""
    lib = _load()
    lib.ncclAllToAll.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclAllToAll.restype = ctypes.c_int
    per_peer = 64
    total = per_peer * nranks
    with tempfile.TemporaryDirectory() as tmpdir:
        trace = os.path.join(tmpdir, "a2a.jsonl")
        os.environ["GPUSIM_TRACE"] = trace
        os.environ["FAKEGPU_NCCL_NO_FILL"] = ""
        os.environ.pop("GPUSIM_TOPOLOGY", None)
        comm = ctypes.c_void_p()
        assert lib.ncclCommInitRank(ctypes.byref(comm), nranks, 0, 0) == 0
        buf = (ctypes.c_char * (total * 4 + GUARD))()
        ctypes.memset(buf, 0xA5, total * 4 + GUARD)
        assert lib.ncclAllToAll(buf, buf, per_peer, NCCL_FLOAT32, comm, None) == 0
        row = [json.loads(l) for l in open(trace) if l.strip()][-1]
        past = bytes(buf[total * 4:])
        tail = bytes(buf[total * 4 - 16:total * 4])
    assert row["bytes"] == total * 4, (
        f"n={nranks}: alltoall bytes {row['bytes']} != total {total * 4}")
    # pseudorandom fill: use the surviving-canary FRACTION, not per-byte equality
    survived = bytes(buf[:total * 4]).count(b"\xa5")
    assert survived < (total * 4) // 4, (
        f"{survived}/{total * 4} payload bytes still hold the canary -- fill stopped at "
        "the per-peer payload")
    assert any(b != 0xA5 for b in tail), "the last peer's slot was never written"
    assert all(b == 0xA5 for b in past), "fill wrote past the buffer"
