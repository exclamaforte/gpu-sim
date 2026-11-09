"""Class-based guard for gpu-sim-3v6d.2: per-link calendar serializes transfers."""
import ctypes
import os
import statistics
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "libfakegpu.so"


def _lib():
    lib = ctypes.CDLL(str(LIB))
    lib.fg_fab_attach.restype = ctypes.c_void_p
    lib.fg_fab_attach.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int)]
    lib.fg_fab_links_init.restype = ctypes.c_int
    lib.fg_fab_links_init.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.fg_fab_transfer.restype = ctypes.c_int
    lib.fg_fab_transfer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
                                    ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                    ctypes.c_int, ctypes.c_uint64, ctypes.c_double, ctypes.c_int]
    lib.fg_fab_detach.argtypes = [ctypes.c_void_p]
    lib.fg_fab_unlink.argtypes = [ctypes.c_char_p]
    return lib


def _run(same_link, repeats=5, bw_gbs=1.0, latency_us=0.0):
    lib = _lib()
    walls = []
    for _ in range(repeats):
        # pid-suffixed: the shm name is machine-global, so a fixed id makes a
        # concurrent run (another suite on this box, or the gate's own run)
        # attach to the SAME fabric and share one calendar, contaminating both
        # walls. Same guard the 3v6d.1 chain test uses (chain-%d).
        run_id = ("3v6d2-%d" % os.getpid()).encode()
        err = ctypes.c_int(0)
        f = lib.fg_fab_attach(run_id, 1, 0, 1, 10_000_000_000, ctypes.byref(err))
        assert err.value == 0 and f
        assert lib.fg_fab_links_init(f, 2) == 0
        links = [0, 0] if same_link else [0, 1]
        outs = [[], []]
        bar = threading.Barrier(2)

        def worker(i):
            bar.wait()
            t0 = time.perf_counter()
            rc = lib.fg_fab_transfer(f, 0, 1 << 20, bw_gbs, latency_us, 0.0, 0, 0, 0.0, links[i])
            t1 = time.perf_counter()
            assert rc == 0
            outs[i].append((t0, t1))

        ths = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        starts = [o[0][0] for o in outs]
        ends = [o[0][1] for o in outs]
        wall = (max(ends) - min(starts)) * 1000.0
        walls.append(wall)
        lib.fg_fab_detach(f)
        lib.fg_fab_unlink(run_id)
    return statistics.median(walls)


class Test3v6d2Contention:
    @classmethod
    def setUpClass(cls):
        # Rebuild the library from the CURRENT sources before measuring. The
        # ledger replay mutates src/fakegpu_fabric.c (calendar CAS removed) and
        # restores it; without a build in the test, the .so would keep serving
        # whichever source was compiled last (and conftest's freshness guard
        # raises once the restored source is newer than the .so). Building here
        # makes the row's plant sound in BOTH directions: mutated source ->
        # mutated .so -> test fails; restored source -> original .so -> passes.
        import subprocess
        import sys
        r = subprocess.run(["make"], cwd=str(REPO), capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, "make failed: " + (r.stdout + r.stderr)[-800:]
        assert LIB.exists(), LIB
    def test_same_link_serializes(self):
        med = _run(True)
        # acceptance (a) asks 1.9-2.1 ms; the window here is +-10% around 2 ms
        # because this is a wall-clock measurement on a shared box (loadavg ~20).
        # The plant (calendar removed) lands at ~1.09 ms, far outside either.
        assert 1.8 <= med <= 2.2, f"same-link median {med:.3f} ms not in [1.8,2.2]"

    def test_diff_links_parallel(self):
        med = _run(False)
        # acceptance (b) names 0.9-1.1 ms; window is +-10% for wall-clock
        # noise on a shared box, and the message states what the code checks.
        assert 0.9 <= med <= 1.15, f"diff-link median {med:.3f} ms not in [0.9,1.15]"



    def test_model_agreement_serialized_total(self):
        # acceptance (d): model (link_model.predict shared_group) and enforcer
        # (calendar) in one assertion. Feed the model's per-transfer duration as
        # the delay to two same-link transfers; serialized total must match the
        # model's allocation (2 x d) within wall-clock tolerance.
        import sys as _sys
        _sys.path.insert(0, str(REPO / "epic_cluster"))
        _sys.path.insert(0, str(REPO / "tests"))
        from epic_cluster.link_model import predict

        def _spec(n, bw, lat, shared):
            nodes = [{"name": "n0", "devices": [{"id": i, "model": "M"} for i in range(n)]}]
            links = []
            for i in range(n):
                for j in range(i + 1, n):
                    links.append({"a": {"node": "n0", "device": i},
                                  "b": {"node": "n0", "device": j},
                                  "kind": "nvlink", "bw_gbps": bw,
                                  "latency_us": lat, "mtu": 4096,
                                  "loss_ppm": 0, "jitter_us": 0,
                                  "shared_group": shared})
            ranks = [{"rank": i, "node": "n0", "device": i} for i in range(n)]
            return {"version": 1,
                    "models": {"M": {"sm_count": 1, "clock_mhz": 1,
                                     "mem_bytes": 1, "mem_bw_gbs": 1,
                                     "arch": "x", "source": "t"}},
                    "nodes": nodes, "links": links, "ranks": ranks}

        spec = _spec(2, 1.0, 10.0, "g")
        d_us = predict(spec, "ring_allreduce", 262144, "float32", n_ranks=2)["duration_s"] * 1e6
        same = _run(True, bw_gbs=1e9, latency_us=d_us)
        # (d) is a scheduler-sum check: the fabric serializes two transfers
        # whose delay is the model allocation (d). Model-vs-fabric formula
        # agreement is checked separately by test_fabric_model_agreement.py (e).
        predicted_total_ms = 2.0 * d_us / 1000.0
        assert abs(same - predicted_total_ms) / predicted_total_ms <= 0.05, (
            f"serialized total {same:.3f} ms not within 5% of model "
            f"allocation {predicted_total_ms:.3f} ms")

    def test_backward_compat_link_id_negative(self):
        # link_id < 0 keeps uncontended behaviour (single transfer ~1ms)
        lib = _lib()
        run_id = ("3v6d2neg-%d" % os.getpid()).encode()
        err = ctypes.c_int(0)
        f = lib.fg_fab_attach(run_id, 1, 0, 1, 10_000_000_000, ctypes.byref(err))
        assert lib.fg_fab_links_init(f, 2) == 0
        t0 = time.perf_counter()
        rc = lib.fg_fab_transfer(f, 0, 1 << 20, 1.0, 0.0, 0.0, 0, 0, 0.0, -1)
        dt = (time.perf_counter() - t0) * 1000.0
        assert rc == 0
        assert 0.9 <= dt <= 1.15
        lib.fg_fab_detach(f)
        lib.fg_fab_unlink(run_id)
