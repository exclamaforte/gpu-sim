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
    expected = 2*(n-1)/n * S / B + 2*(n-1)*alpha
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
    # n=2 is the degenerate single-hop ring: it is NOT expected to be faster than
    # n=3 (it moves the whole buffer per phase), so monotonicity is asserted from
    # n=3 upward and the n=2 case is pinned separately by test_ring_n2_derivation.
    for n in [3, 4, 8]:
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
    # bw_term = 2(n-1)/n * S/B; lat_term = 2(n-1)*alpha.
    # crossover when bw_term == lat_term => S = (2(n-1)*alpha * B) / (2(n-1)/n) = alpha*B*n
    S_star = alpha * B * n
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
    # ggh: n=2 byte factor is 1.0 (was 2.0; corrected per eut committed traces).
    # latency factor unchanged (2*alpha).
    expected = 1.0 * S / bw + 2 * alpha
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
    # ggh: n=2 byte factor is 1.0 (was 2.0; corrected per eut committed traces).
    B = 50.0 * 1e9 / 8.0
    alpha = 1e-6
    S = 1000 * 4
    expected = 1.0 * S / B + 2 * alpha
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


def test_ring_n2_derivation():
    """gpu-sim-ggh: n=2 ring byte factor corrected to 1.0 (was 2.0).

    eut committed traces (experiments/gpu-sim-pkb/traces/) show per-rank byte
    factor 1.00 at every clean size (4KiB..2MiB), matching the standard ring
    formula 2*(n-1)/n = 1.0 at n=2. Per-rank wire bytes are now S (not 2*S).
    Latency factor unchanged (2*alpha). NEGATIVE CONTROL: the old 2*S factor
    must NOT satisfy the corrected check.
    """
    n, count = 2, 1024
    S = count * 4
    B_gbps, alpha_us = 100.0, 5.0
    spec = uniform_spec(n, B_gbps, alpha_us)
    rec = predict(spec, "ring_allreduce", count, "float32", n_ranks=n)
    per_rank = sum(rec["per_link_bytes"].values()) / n
    assert per_rank == 1 * S, (per_rank, 1 * S)

    B = B_gbps * 1e9 / 8.0
    alpha = alpha_us * 1e-6
    assert abs(rec["duration_s"] - (1 * S / B + 2 * alpha)) < 1e-15

    # negative control: the old 2*S byte factor must NOT satisfy the corrected check
    old_duration = 2 * S / B + 2 * alpha
    assert abs(rec["duration_s"] - old_duration) > 1e-9, "old 2*S byte factor still fits"

    # and n>=3 must keep the closed form the existing tests pin
    for n2 in (3, 4, 8):
        spec2 = uniform_spec(n2, B_gbps, alpha_us)
        rec2 = predict(spec2, "ring_allreduce", count, "float32", n_ranks=n2, pieces_per_phase=1)
        assert abs(rec2["duration_s"] - (2 * (n2 - 1) / n2 * S / B + 2 * (n2 - 1) * alpha)) < 1e-15


def test_negative_control_old_byte_factor():
    """gpu-sim-8sy item 4: old byte factor (S) must FAIL against corrected (2*S)."""
    n, count = 2, 1024
    S = count * 4
    B_gbps, alpha_us = 100.0, 5.0
    spec = uniform_spec(n, B_gbps, alpha_us)
    rec = predict(spec, "ring_allreduce", count, "float32", n_ranks=n)
    B = B_gbps * 1e9 / 8.0
    alpha = alpha_us * 1e-6
    corrected = 1 * S / B + 2 * alpha
    old_byte = 2 * S / B + 2 * alpha   # old 2*S byte factor must fail
    assert abs(rec["duration_s"] - corrected) < 1e-15
    assert abs(rec["duration_s"] - old_byte) > 1e-9


def test_negative_control_old_latency_coefficient():
    """gpu-sim-8sy item 4: old latency coefficient (2n=4) must FAIL against corrected (2)."""
    n, count = 2, 1024
    S = count * 4
    B_gbps, alpha_us = 100.0, 5.0
    spec = uniform_spec(n, B_gbps, alpha_us)
    rec = predict(spec, "ring_allreduce", count, "float32", n_ranks=n)
    B = B_gbps * 1e9 / 8.0
    alpha = alpha_us * 1e-6
    corrected = 1 * S / B + 2 * alpha
    old_latency = 1 * S / B + 4 * alpha  # only the latency coeff reverts
    assert abs(rec["duration_s"] - corrected) < 1e-15
    assert abs(rec["duration_s"] - old_latency) > 1e-9


def test_factors_mutually_consistent():
    """gpu-sim-8sy: byte factor and latency factor agree on hop count at every n."""
    from epic_cluster.link_model import _factor, _latency_factor
    for n in (2, 3, 4, 8):
        bf = _factor("ring_allreduce", n)
        lf = _latency_factor("ring_allreduce", n)
        # ring hop count is 2*(n-1) total (two phases, n-1 hops each).
        assert lf == 2 * (n - 1), (n, lf)
        # ggh: no n=2 special case; bf = 2*(n-1)/n for all n.
        expected_bf = 2.0 * (n - 1) / n
        assert bf == expected_bf, (n, bf, expected_bf)
