"""C/Python agreement test for the analytical NCCL cost model (gpu-sim-62i).

Compares fg_nccl_predict (default analytical path) against epic_cluster.nccl_cost
for the same (op, n, S) on a declared topology. The two must agree to 1e-6.
"""
import ctypes
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from epic_cluster import nccl_cost, topology

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(REPO, "libfakegpu.so")


def _spec(n, bw=10.0, lat=5.0):
    if n == 2:
        return {"version": 1, "models": {"M": {"sm_count": 1, "clock_mhz": 1,
                                                "mem_bytes": 1, "mem_bw_gbs": 1,
                                                "arch": "x", "source": "t"}},
                "nodes": [{"name": "n0", "devices": [{"id": 0, "model": "M"},
                                                    {"id": 1, "model": "M"}]}],
                "links": [{"a": {"node": "n0", "device": 0},
                           "b": {"node": "n0", "device": 1}, "kind": "nvlink",
                           "bw_gbps": bw, "latency_us": lat, "mtu": 4096,
                           "loss_ppm": 0, "jitter_us": 0, "shared_group": None}],
                "ranks": [{"rank": 0, "node": "n0", "device": 0},
                          {"rank": 1, "node": "n0", "device": 1}]}
    nodes = [{"name": "n0", "devices": [{"id": i, "model": "M"} for i in range(n)]}]
    links = []
    for i in range(n):
        for j in range(i + 1, n):
            links.append({"a": {"node": "n0", "device": i},
                          "b": {"node": "n0", "device": j}, "kind": "nvlink",
                          "bw_gbps": bw, "latency_us": lat, "mtu": 4096,
                          "loss_ppm": 0, "jitter_us": 0, "shared_group": None})
    ranks = [{"rank": i, "node": "n0", "device": i} for i in range(n)]
    return {"version": 1, "models": {"M": {"sm_count": 1, "clock_mhz": 1,
                                           "mem_bytes": 1, "mem_bw_gbs": 1,
                                           "arch": "x", "source": "t"}},
            "nodes": nodes, "links": links, "ranks": ranks}


OPMAP = {"ring_allreduce": "allreduce", "ring_allgather": "allgather",
         "ring_reduce_scatter": "reducescatter", "ring_alltoall": "alltoall"}


def _c_predict(n, algo, count, bw=10.0, lat=5.0):
    s = _spec(n, bw, lat)
    blob = topology.emit(s)
    fd, path = tempfile.mkstemp(suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        lib = ctypes.CDLL(LIB)
        lib.fg_topo_load.restype = ctypes.c_void_p
        lib.fg_topo_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.fg_nccl_predict.restype = ctypes.c_double
        lib.fg_nccl_predict.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        errbuf = ctypes.create_string_buffer(256)
        topo = lib.fg_topo_load(path.encode(), errbuf, 256)
        assert topo, errbuf.value.decode(errors="replace")
        bl = ctypes.c_int(-1)
        regime = ctypes.create_string_buffer(64)
        dur = lib.fg_nccl_predict(topo, algo.encode(), count, 7, n,
                                  ctypes.byref(bl), regime, 64)
        return dur, bl.value, regime.value.decode()
    finally:
        os.unlink(path)


@pytest.mark.parametrize("algo", list(OPMAP))
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("count", [1024, 100000])
def test_analytical_agreement(algo, n, count):
    bw, lat = 10.0, 5.0
    dur_c, bl, regime_c = _c_predict(n, algo, count, bw, lat)
    nccl_cost.FABRICS["__t__"] = {"alpha_s": lat * 1e-6, "bw_gbps": bw}
    pred = nccl_cost.predict(OPMAP[algo], n, count * 4, fabric="__t__")
    assert abs(dur_c - pred.total_s) < 1e-6 * max(1.0, pred.total_s), (algo, n, count, dur_c, pred.total_s)
    assert regime_c.startswith("analytical_"), (algo, regime_c)
