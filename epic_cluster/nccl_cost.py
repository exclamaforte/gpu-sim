"""Analytical cost model for NCCL collectives.

Replaces a lookup table (four hand-measured rows, refusing anything larger) with a
closed-form model derived from the algorithm's communication pattern and the
fabric's parameters. Every number it produces is explainable: total time is split
into a latency term, a bandwidth term and the integer step/piece counts that
produced them, so a reader can check the arithmetic instead of trusting it.

THE MODEL, and why it looks like this
-------------------------------------
A ring collective runs `steps` sequential hops. On each hop a rank sends
`bytes_per_step` to its neighbour. Two regimes:

  latency-bound   each hop's transfer is shorter than the link latency, so hops do
                  not overlap:      T = steps * (alpha + bytes_per_step / B)
  bandwidth-bound the transfer is long enough that the next hop starts while the
                  previous is still moving:   T = alpha + steps * bytes_per_step / B

The crossover between them is a real property of the fabric, not a tuned threshold,
and it is what makes a model useful: it predicts WHERE a collective stops being
latency-dominated.

Ring all-reduce is two phases (reduce-scatter, then all-gather), each of which moves
(n-1)/n of the buffer along n-1 hops. That is the source of the familiar 2(n-1)/n
bandwidth factor -- it is DERIVED here, not hard-coded as a magic constant.

WHAT IS NOT MODELLED, deliberately: congestion, contention with compute, per-channel
thread scheduling, and protocol overheads (LL/LL128 packet headers). Those need
measurement, and inventing them would put unearned precision into a number that
reads like a prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

# Declared fabric parameters. alpha is a per-hop latency; B is the one-way
# bandwidth available to a single ring link.
FABRICS = {
    "nvlink4_h100": {"alpha_s": 1.5e-6, "bw_gbps": 450.0},   # 900 GB/s bidir / 2
    "pcie5_x16":    {"alpha_s": 8.0e-6, "bw_gbps": 25.0},    # ~50 GB/s bidir / 2
    "ib_400g":      {"alpha_s": 2.5e-6, "bw_gbps": 45.0},
}


@dataclass
class Prediction:
    total_s: float
    latency_s: float
    bandwidth_s: float
    steps: int
    bytes_per_step: int
    regime: str
    algorithm: str
    n: int
    bytes: int

    @property
    def bandwidth_fraction(self) -> float:
        return self.bandwidth_s / self.total_s if self.total_s > 0 else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bandwidth_fraction"] = round(self.bandwidth_fraction, 4)
        return d


def _ring_phase(n: int, size_bytes: int, alpha: float, bw_bps: float,
                algorithm: str) -> Prediction:
    """One ring phase (reduce-scatter or all-gather): n-1 hops of size/n each."""
    steps = max(n - 1, 1)
    per_step = size_bytes / n
    hop_s = per_step / bw_bps

    no_pipe_s = steps * (alpha + hop_s)
    pipelined_s = alpha + steps * hop_s
    # Pipelining only helps once a hop outlasts the latency; below that the hops
    # serialise and the latency term is paid on every one.
    if hop_s > alpha:
        total, lat, regime = pipelined_s, alpha, "bandwidth"
    else:
        total, lat, regime = no_pipe_s, steps * alpha, "latency"

    return Prediction(
        total_s=total, latency_s=lat, bandwidth_s=total - lat,
        steps=steps, bytes_per_step=int(per_step),
        regime=regime, algorithm=algorithm, n=n, bytes=size_bytes,
    )


def allreduce(n: int, size_bytes: int, fabric: str = "nvlink4_h100",
              algorithm: str = "ring") -> Prediction:
    """Ring all-reduce = reduce-scatter + all-gather, hence the factor of 2."""
    f = FABRICS[fabric]
    if n <= 1:
        return Prediction(0.0, 0.0, 0.0, 0, size_bytes, "trivial", algorithm, n,
                          size_bytes)
    if algorithm == "tree":
        # Binary tree reduce + broadcast: 2*log2(n) hops, but each hop moves the
        # full buffer rather than a 1/n slice.
        hops = 2 * math.ceil(math.log2(n))
        bw_s = hops * size_bytes / f["bw_gbps"] * 1e9
        lat_s = hops * f["alpha_s"]
        return Prediction(bw_s + lat_s, lat_s, bw_s, hops, size_bytes,
                          "bandwidth" if bw_s > lat_s else "latency", algorithm, n,
                          size_bytes)
    p = _ring_phase(n, size_bytes, f["alpha_s"], f["bw_gbps"] * 1e9, "ring_allreduce")
    return Prediction(
        total_s=p.total_s * 2, latency_s=p.latency_s * 2,
        bandwidth_s=p.bandwidth_s * 2,
        # Two phases, so the hop count doubles too -- reporting 7 steps while
        # charging 14 hops of latency would make the breakdown inconsistent with
        # its own total, which is exactly the kind of quiet mismatch that makes a
        # cost model untrustworthy.
        steps=p.steps * 2,
        bytes_per_step=p.bytes_per_step, regime=p.regime,
        algorithm="ring_allreduce", n=n, bytes=size_bytes,
    )


def _one_phase(op: str):
    def f(n: int, size_bytes: int, fabric: str = "nvlink4_h100") -> Prediction:
        fab = FABRICS[fabric]
        return _ring_phase(n, size_bytes, fab["alpha_s"], fab["bw_gbps"] * 1e9, op)
    return f


allgather = _one_phase("ring_allgather")
reducescatter = _one_phase("ring_reduce_scatter")
broadcast = _one_phase("ring_broadcast")


def sendrecv(size_bytes: int, fabric: str = "nvlink4_h100",
             hops: int = 1) -> Prediction:
    """Point-to-point: one transfer along `hops` links, no ring factor."""
    f = FABRICS[fabric]
    lat = hops * f["alpha_s"]
    bw = size_bytes / (f["bw_gbps"] * 1e9)
    return Prediction(bw + lat, lat, bw, hops, size_bytes,
                      "bandwidth" if bw > lat else "latency", "p2p", 2, size_bytes)


def ring_exchange(size_bytes: int, hops: int = 1, fabric: str = "nvlink4_h100") -> Prediction:
    """Serial ring exchange: `hops` dependent point-to-point transfers.

    Unlike sendrecv (one transfer over `hops` links, S/B + alpha*hops), a ring
    exchange performs `hops` SEQUENTIAL transfers, each of `size_bytes`, so the
    cost is hops*(alpha + S/B). This is the CP K/V exchange: (cp-1) ranks'
    blocks must pass through, one at a time, and each transfer depends on the
    previous one completing.
    """
    f = FABRICS[fabric]
    bw_bps = f["bw_gbps"] * 1e9
    alpha = f["alpha_s"]
    hop_s = size_bytes / bw_bps
    total = hops * (alpha + hop_s)
    lat = hops * alpha
    bw = hops * hop_s
    return Prediction(total, lat, bw, hops, size_bytes,
                      "bandwidth" if bw > lat else "latency",
                      "ring_exchange", 2, size_bytes)


def alltoall(n: int, size_bytes: int, fabric: str = "nvlink4_h100") -> Prediction:
    """Ring all-to-all: n-1 hops, each moving 1/n of the buffer."""
    f = FABRICS[fabric]
    return _ring_phase(n, size_bytes, f["alpha_s"], f["bw_gbps"] * 1e9, "ring_alltoall")


OPS = {
    "ring_exchange": ring_exchange, "allreduce": allreduce, "allgather": allgather,
    "reducescatter": reducescatter, "broadcast": broadcast,
    "alltoall": alltoall, "sendrecv": sendrecv,
}


def predict(op: str, n: int, size_bytes: int, fabric: str = "nvlink4_h100",
            algorithm: str = "ring") -> Prediction:
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; known: {sorted(OPS)}")
    if op == "allreduce":
        return allreduce(n, size_bytes, fabric, algorithm)
    if op == "sendrecv":
        # Different signature: point-to-point has a size, not a rank count.
        return sendrecv(size_bytes, fabric)
    return OPS[op](n, size_bytes, fabric)
