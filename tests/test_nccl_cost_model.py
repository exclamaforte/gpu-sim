"""The analytical NCCL model must obey theory, not just produce numbers.

These are property tests, not golden-value tests: the point is that the model's
structure is right, so that changing a fabric constant cannot silently break the
relationship between message size, rank count and time.

Each test states the theoretical claim it is checking.
"""
from __future__ import annotations

import math

import pytest

from epic_cluster.nccl_cost import FABRICS, predict


def test_large_message_approaches_the_ring_bandwidth_limit():
    """As S grows the latency term vanishes and allreduce -> 2(n-1)/n * S/B.

    This is the defining property of a ring all-reduce: it moves 2(n-1)/n of the
    buffer through the fabric no matter how the buffer is split.
    """
    n, S = 8, 1 << 30
    B = FABRICS["nvlink4_h100"]["bw_gbps"] * 1e9
    p = predict("allreduce", n, S)
    theory = 2 * (n - 1) / n * S / B
    assert p.total_s == pytest.approx(theory, rel=0.01)


def test_bandwidth_term_scales_linearly_with_size():
    """Doubling the buffer doubles the bandwidth term (and leaves latency alone)."""
    a = predict("allreduce", 8, 16 << 20)
    b = predict("allreduce", 8, 32 << 20)
    assert b.bandwidth_s == pytest.approx(2 * a.bandwidth_s, rel=1e-9)
    assert b.latency_s == pytest.approx(a.latency_s, rel=1e-9)


def test_latency_term_scales_with_rank_count():
    """A ring pays one hop of latency per step, and steps = n-1 per phase."""
    a = predict("allreduce", 4, 1024)
    b = predict("allreduce", 8, 1024)
    assert b.steps == 2 * (8 - 1)
    assert b.latency_s == pytest.approx(a.latency_s * 7 / 3, rel=1e-9)


def test_regimes_switch_with_size_not_with_magic():
    """Small messages are latency-bound, large ones bandwidth-bound, and the
    crossover is where a hop outlasts the link latency."""
    assert predict("allreduce", 8, 1024).regime == "latency"
    assert predict("allreduce", 8, 64 << 20).regime == "bandwidth"

    B = FABRICS["nvlink4_h100"]["bw_gbps"] * 1e9
    alpha = FABRICS["nvlink4_h100"]["alpha_s"]
    crossover_bytes = alpha * B * 8  # hop time == alpha  =>  (S/n)/B == alpha
    assert predict("allreduce", 8, int(crossover_bytes * 2)).regime == "bandwidth"
    assert predict("allreduce", 8, int(crossover_bytes / 2)).regime == "latency"


def test_allreduce_costs_exactly_two_phases():
    """Ring all-reduce = reduce-scatter + all-gather, so it is 2x either alone."""
    ar = predict("allreduce", 8, 8 << 20)
    ag = predict("allgather", 8, 8 << 20)
    assert ar.total_s == pytest.approx(2 * ag.total_s, rel=1e-9)


def test_point_to_point_has_no_ring_factor():
    """A send moves the whole buffer once; a ring all-gather moves (n-1)/n of it.

    So the point-to-point bandwidth term is LARGER than the all-gather one, which
    is the opposite of the intuition that 'more ranks costs more' -- and it is the
    check that caught a wrong assertion here (the model was right).
    """
    n, S = 8, 8 << 20
    p2p = predict("sendrecv", n, S)
    ag = predict("allgather", n, S)
    B = FABRICS["nvlink4_h100"]["bw_gbps"] * 1e9
    assert p2p.bandwidth_s == pytest.approx(S / B, rel=1e-9)
    assert ag.bandwidth_s == pytest.approx((n - 1) / n * S / B, rel=1e-9)
    assert p2p.bandwidth_s > ag.bandwidth_s


def test_monotonic_in_size_and_rank_count():
    for op in ("allreduce", "allgather", "reducescatter", "alltoall"):
        times = [predict(op, 8, s).total_s for s in (1024, 1 << 20, 16 << 20)]
        assert times == sorted(times), op
        per_n = [predict(op, n, 8 << 20).total_s for n in (2, 4, 8)]
        assert per_n == sorted(per_n), op


def test_single_rank_is_free():
    assert predict("allreduce", 1, 1 << 20).total_s == 0.0


def test_faster_fabric_is_faster():
    fast = predict("allreduce", 8, 64 << 20, fabric="nvlink4_h100")
    slow = predict("allreduce", 8, 64 << 20, fabric="pcie5_x16")
    assert fast.total_s < slow.total_s
    assert fast.regime == slow.regime == "bandwidth"
