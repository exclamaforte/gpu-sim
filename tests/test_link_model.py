import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from epic_cluster.link_model import (
    predict, ALGORITHMS, DTYPE_BYTES, _shared_counts,
)


def uniform_spec(n, bw=10.0, lat=10.0, shared=None):
    nodes = [{"name": "n0", "devices": [{"id": i, "model": "M"} for i in range(n)]}]
    links = []
    for i in range(n):
        for j in range(i+1, n):
            links.append({
            "a": {"node": "n0", "device": i},
            "b": {"node": "n0", "device": j},
            "kind": "nvlink",
            "bw_gbps": bw,
            "latency_us": lat,
            "mtu": 4096,
            "loss_ppm": 0,
            "jitter_us": 0,
            "shared_group": shared,
        })
    ranks = [{"rank": i, "node": "n0", "device": i} for i in range(n)]
    return {
        "version": 1,
        "models": {"M": {"sm_count": 1, "clock_mhz": 1, "mem_bytes": 1, "mem_bw_gbs": 1, "arch": "x", "source": "t"}},
        "nodes": nodes,
        "links": links,
        "ranks": ranks,
    }


def test_all_algorithms_closed_form():
    n = 4
    bw = 10.0
    lat = 10.0
    alpha = lat * 1e-6
    B = bw * 1e9 / 8.0
    count = 100000
    dtype = "float32"
    S = count * 4
    spec = uniform_spec(n, bw, lat)
    # ring allreduce
    rec = predict(spec, "ring_allreduce", count, dtype)
    expected = 2*(n-1)/n * S / B + 2*n*alpha
    assert abs(rec["duration_s"] - expected) < 1e-9
    # allgather
    rec = predict(spec, "ring_allgather", count, dtype)
    expected = (n-1)/n * S / B + n*alpha
    assert abs(rec["duration_s"] - expected) < 1e-9
    # reducescatter same
    rec = predict(spec, "ring_reduce_scatter", count, dtype)
    expected = (n-1)/n * S / B + n*alpha
    assert abs(rec["duration_s"] - expected) < 1e-9
    # alltoall
    rec = predict(spec, "ring_alltoall", count, dtype)
    expected = (n-1)/n * S / B + n*alpha
    assert abs(rec["duration_s"] - expected) < 1e-9
    # tree bcast
    rec = predict(spec, "tree_bcast", count, dtype)
    expected = math.log2(n) * (S / B + alpha)
    assert abs(rec["duration_s"] - expected) < 1e-9


def test_monotonicity_n():
    count = 50000
    for S_n in [2, 4, 8]:
        pass
    prev = None
    for n in [2, 3, 4, 8]:
        spec = uniform_spec(n, 10.0, 10.0)
        rec = predict(spec, "ring_allreduce", count, "float32")
        if prev is not None:
            assert rec["duration_s"] >= prev - 1e-12
        prev = rec["duration_s"]


def test_monotonicity_S():
    n = 4
    prev = None
    for count in [1000, 10000, 100000]:
        spec = uniform_spec(n, 10.0, 10.0)
        rec = predict(spec, "ring_allreduce", count, "float32")
        if prev is not None:
            assert rec["duration_s"] >= prev - 1e-12
        prev = rec["duration_s"]


def test_regime_boundary():
    n = 4
    bw = 10.0
    lat = 10.0
    alpha = lat * 1e-6
    B = bw * 1e9 / 8.0
    # S* = alpha * B * n * (n-1) / (2*(n-1)/n) ? derive: bw_term = 2(n-1)/n * S/B; lat_term = 2n*alpha.
    # crossover when bw_term == lat_term => S = (2n*alpha * B) / (2(n-1)/n) = alpha*B*n^2/(n-1)
    S_star = alpha * B * n * n / (n - 1)
    count_star = int(S_star / 4)
    spec = uniform_spec(n, bw, lat)
    rec = predict(spec, "ring_allreduce", count_star, "float32")
    # check regime agrees: if count < star -> latency? Actually for small S, lat_term dominates -> latency.
    # just check regime is one of valid and boundary within 5% by evaluating counts around
    rec_low = predict(spec, "ring_allreduce", int(count_star * 0.5), "float32")
    rec_high = predict(spec, "ring_allreduce", int(count_star * 2.0), "float32")
    assert rec_low["regime"] == "latency"
    assert rec_high["regime"] == "bandwidth"


def test_multihop():
    # A-B-C chain
    spec = {
        "version": 1,
        "models": {"M": {"sm_count": 1, "clock_mhz": 1, "mem_bytes": 1, "mem_bw_gbs": 1, "arch": "x", "source": "t"}},
        "nodes": [{"name": "n0", "devices": [{"id": i, "model": "M"} for i in range(3)]}],
        "links": [
            {"a": {"node": "n0", "device": 0}, "b": {"node": "n0", "device": 1}, "kind": "pcie",
             "bw_gbps": 100.0, "latency_us": 2.0, "mtu": 4096, "loss_ppm": 0, "jitter_us": 0, "shared_group": None},
            {"a": {"node": "n0", "device": 1}, "b": {"node": "n0", "device": 2}, "kind": "pcie",
             "bw_gbps": 10.0, "latency_us": 3.0, "mtu": 4096, "loss_ppm": 0, "jitter_us": 0, "shared_group": None},
        ],
        "ranks": [{"rank": i, "node": "n0", "device": i} for i in range(3)],
    }
    # allreduce between rank0 and rank2 (n=2 subset? but n_ranks=2)
    spec2 = json.loads(json.dumps(spec))
    spec2["ranks"] = [{"rank": 0, "node": "n0", "device": 0}, {"rank": 1, "node": "n0", "device": 2}]
    rec = predict(spec2, "ring_allreduce", 1000, "float32", n_ranks=2)
    # B_eff = min(100,10)=10 Gbps; alpha = 2+3=5 us
    bw = 10.0 * 1e9 / 8.0
    alpha = 5e-6
    S = 1000 * 4
    expected = 1.0 * S / bw + 4 * alpha  # n=2: 2*(1)/2=1 S/B + 2*2*alpha=4 alpha
    assert abs(rec["duration_s"] - expected) < 1e-9
    assert rec["bottleneck"]["link_id"] == 1


def test_shared_group():
    spec = {
        "version": 1,
        "models": {"M": {"sm_count": 1, "clock_mhz": 1, "mem_bytes": 1, "mem_bw_gbs": 1, "arch": "x", "source": "t"}},
        "nodes": [{"name": "n0", "devices": [{"id": i, "model": "M"} for i in range(2)]}],
        "links": [
            {"a": {"node": "n0", "device": 0}, "b": {"node": "n0", "device": 1}, "kind": "nvlink",
             "bw_gbps": 100.0, "latency_us": 1.0, "mtu": 4096, "loss_ppm": 0, "jitter_us": 0, "shared_group": "g"},
            {"a": {"node": "n0", "device": 0}, "b": {"node": "n0", "device": 1}, "kind": "nvlink",
             "bw_gbps": 100.0, "latency_us": 1.0, "mtu": 4096, "loss_ppm": 0, "jitter_us": 0, "shared_group": "g"},
        ],
        "ranks": [{"rank": 0, "node": "n0", "device": 0}, {"rank": 1, "node": "n0", "device": 1}],
    }
    counts = _shared_counts(spec)
    assert counts["g"] == 2
    # each link eff bw = 50
    rec = predict(spec, "ring_allreduce", 1000, "float32", n_ranks=2)
    # n=2: S/B + 4*alpha with B=50 Gbps
    B = 50.0 * 1e9 / 8.0
    alpha = 1e-6
    S = 1000 * 4
    expected = 1.0 * S / B + 4 * alpha
    assert abs(rec["duration_s"] - expected) < 1e-9


def test_cli_invalid_topology():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("{bad json")
        p = f.name
    out = subprocess.run([sys.executable, "-m", "epic_cluster.link_model",
                          "--topology", p, "--op", "ring_allreduce", "--count", "10"],
                         capture_output=True, text=True)
    assert out.returncode != 0


def test_cli_valid():
    spec = uniform_spec(4, 10.0, 10.0)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        p = f.name
    out = subprocess.run([sys.executable, "-m", "epic_cluster.link_model",
                          "--topology", p, "--op", "ring_allreduce", "--count", "1000", "--dtype", "float32"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    rec = json.loads(out.stdout)
    assert "duration_s" in rec
    assert "regime" in rec
    assert "bottleneck" in rec
    assert "link_id" in rec["bottleneck"]
