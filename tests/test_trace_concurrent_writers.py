"""Concurrent trace writers must never corrupt or lose a record (gpu-sim-wtorn).

At dp=8 the eight ranks append to ONE trace file. The writer now builds each record in a
buffer and hands it to a SINGLE write() on an O_APPEND descriptor, so lines from concurrent
processes can only interleave at line boundaries (POSIX guarantees the offset update is
atomic and one write() to a regular file lands contiguously).

HONEST NOTE ON THE ORIGIN: the original motivation was a dp=8 run showing 1 torn line in
3186 and unequal per-rank record counts. Reverting the writer to fopen/fprintf does NOT make
this test fail -- a ~200-byte line below BUFSIZ is already emitted as one write by stdio, so
interleaving was never the proven cause; a truncated record from a process killed at the
timeout is the more likely explanation, and unequal counts across time-sliced ranks are not
evidence of loss either. The atomic write is therefore HARDENING plus an explicit guarantee,
not a proven bug fix, and it is recorded as such rather than as a repair.

The value of the test is the invariant, not the archaeology: N processes hammer the same
file through the shim and every record must parse, with exactly one record per call.
"""
import ctypes
import json
import multiprocessing as mp
import os
import pathlib
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB_PATH = str(ROOT / "libfakenccl.so")
TMP_ROOT = ROOT / ".lab" / "tmp"          # never /tmp: the host quota is tight

NCCL_FLOAT32 = 7
NPROC = 8
CALLS = 60


def _worker(path: str, rank: int) -> None:
    os.environ["GPUSIM_TRACE"] = path
    os.environ["FAKEGPU_NCCL_NO_FILL"] = "1"
    os.environ.pop("GPUSIM_TOPOLOGY", None)
    lib = ctypes.CDLL(LIB_PATH)
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                     ctypes.c_int, ctypes.c_ulonglong]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p]
    comm = ctypes.c_void_p()
    assert lib.ncclCommInitRank(ctypes.byref(comm), NPROC, rank, 0) == 0
    buf = (ctypes.c_char * (64 * 4))()
    for _ in range(CALLS):
        assert lib.ncclAllReduce(buf, buf, 64, NCCL_FLOAT32, 0, comm, None) == 0


@pytest.mark.skipif(not os.path.exists(LIB_PATH), reason="libfakenccl.so not built")
def test_concurrent_writers_never_tear_a_line():
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP_ROOT) as d:
        path = os.path.join(d, "trace.jsonl")
        ctx = mp.get_context("fork")
        procs = [ctx.Process(target=_worker, args=(path, r)) for r in range(NPROC)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, f"worker failed: {p.exitcode}"

        torn, ok = [], 0
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    ok += 1
                except json.JSONDecodeError:
                    torn.append((i, line[:80]))
    assert not torn, f"{len(torn)} torn line(s), first: {torn[0]}"
    assert ok == NPROC * CALLS, (
        f"expected one record per call ({NPROC * CALLS}), got {ok} -- records were lost")
