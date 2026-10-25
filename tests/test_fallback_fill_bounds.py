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
