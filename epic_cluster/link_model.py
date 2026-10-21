"""gpu-sim-m3d.3: link/collective cost model (pure Python)."""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DTYPE_BYTES = {
    "int8": 1, "uint8": 1, "char": 1, "byte": 1,
    "int16": 2, "uint16": 2, "float16": 2, "half": 2, "bfloat16": 2,
    "int32": 4, "uint32": 4, "float32": 4, "float": 4, "int": 4,
    "int64": 8, "uint64": 8, "float64": 8, "double": 8, "long": 8,
}

ALGORITHMS = {"ring_allreduce", "ring_allgather", "ring_reduce_scatter",
               "tree_bcast", "ring_alltoall"}


def _load_spec(path: str) -> Any:
    # reuse spec validator if available
    try:
        from epic_cluster.spec import validate, load
        return load(path)
    except Exception:
        # fallback: parse json only
        with open(path) as f:
            return json.load(f)


def _dev_index(spec: Any) -> Dict[Tuple[str, int], int]:
    out = {}
    idx = 0
    for node in spec["nodes"]:
        for dev in node["devices"]:
            out[(node["name"], dev["id"])] = idx
            idx += 1
    return out


def _adjacency(spec: Any, dev_idx: Dict[Tuple[str, int], int]):
    n = len(dev_idx)
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for li, link in enumerate(spec["links"]):
        a = dev_idx[(link["a"]["node"], link["a"]["device"])]
        b = dev_idx[(link["b"]["node"], link["b"]["device"])]
        adj[a].append((b, li))
        adj[b].append((a, li))
    return adj


def _shortest_path(adj, link_latency, src, dst):
    """Dijkstra by latency (sum of link latency_us). Returns list of link indices or None."""
    if src == dst:
        return []
    n = len(adj)
    INF = float("inf")
    dist = [INF] * n
    prev = [-1] * n
    prev_link = [-1] * n
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for (v, li) in adj[u]:
            w = link_latency[li]
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                prev_link[v] = li
                heapq.heappush(pq, (nd, v))
    if dist[dst] == INF:
        return None
    links = []
    cur = dst
    while cur != src:
        links.append(prev_link[cur])
        cur = prev[cur]
    links.reverse()
    return links


def _shared_counts(spec: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for link in spec["links"]:
        sg = link["shared_group"]
        if sg is None:
            continue
        counts[sg] = counts.get(sg, 0) + 1
    return counts


def _eff_bw(link: Any, counts: Dict[str, int]) -> float:
    bw = float(link["bw_gbps"])
    sg = link["shared_group"]
    if sg is None:
        return bw
    c = counts.get(sg, 1)
    if c < 1:
        c = 1
    return bw / c


def _path_bw(links_spec, path_link_ids, counts):
    if not path_link_ids:
        return float("inf")
    return min(_eff_bw(links_spec[li], counts) for li in path_link_ids)


def _path_latency(links_spec, path_link_ids):
    return sum(float(links_spec[li]["latency_us"]) for li in path_link_ids)


def _ring_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def _tree_edges(n: int) -> List[Tuple[int, int]]:
    # binary tree by index: parent (i-1)//2 -> child i
    edges = []
    for i in range(1, n):
        edges.append(((i - 1) // 2, i))
    return edges


def _logical_edges(algorithm: str, n: int) -> List[Tuple[int, int]]:
    if algorithm in ("ring_allreduce", "ring_allgather", "ring_reduce_scatter", "ring_alltoall"):
        return _ring_edges(n)
    if algorithm == "tree_bcast":
        return _tree_edges(n)
    raise ValueError(f"unknown algorithm {algorithm}")


def _ring_chunk_factor(n: int) -> float:
    """Fraction of S carried by ONE ring transfer in one phase.

    n >= 3: the ring splits the buffer into n chunks, so a phase moves
    (n-1)/n * S through each rank. n == 2 is degenerate: the "ring" has a single
    peer, one hop per phase, and the phase carries the WHOLE buffer (chunk = S).
    This is the 2x that the m3d.8 gpu-sim-8sy evidence measured (per-rank wire
    bytes 8192 for S=4096, with distinct per-phase digests).
    """
    return 1.0 if n == 2 else (n - 1) / n


def _ring_hops_per_phase(n: int) -> int:
    """Hops a phase takes through the ring: (n-1) for all n >= 2 (the classic
    ring allreduce has n-1 reduce-scatter steps and n-1 allgather steps). At
    n=2 this is 1, matching the observed 2 transfers per direction."""
    return n - 1


def _factor(algorithm: str, n: int) -> float:
    if algorithm == "ring_allreduce":
        # two phases (reduce-scatter, allgather)
        return 2.0 * _ring_chunk_factor(n)
    if algorithm in ("ring_allgather", "ring_reduce_scatter", "ring_alltoall"):
        return (n - 1.0) / n
    if algorithm == "tree_bcast":
        return 1.0
    raise ValueError(f"unknown algorithm {algorithm}")


def _latency_factor(algorithm: str, n: int) -> float:
    if algorithm == "ring_allreduce":
        # two phases, each paying hops_per_phase * alpha
        return 2.0 * _ring_hops_per_phase(n)
    if algorithm in ("ring_allgather", "ring_reduce_scatter", "ring_alltoall"):
        return n
    if algorithm == "tree_bcast":
        return math.log2(n) if n > 1 else 1.0
    raise ValueError(f"unknown algorithm {algorithm}")


def _bw_factor(algorithm: str, n: int) -> float:
    if algorithm == "ring_allreduce":
        return 2.0 * _ring_chunk_factor(n)
    if algorithm in ("ring_allgather", "ring_reduce_scatter", "ring_alltoall"):
        return (n - 1.0) / n
    if algorithm == "tree_bcast":
        return 1.0  # per edge S; log2 handled in latency? Actually bcast: log2(n)*(S/B+alpha)
    raise ValueError(f"unknown algorithm {algorithm}")


FG_FAB_MAX_PAYLOAD = 1 << 20  # 1 MiB, matches include/fakegpu_fabric.h


def _chunk_factor(n: int) -> float:
    """chunk(n): S at n=2, (n-1)/n*S at n>=3 (model's formula)."""
    if n == 2:
        return 1.0
    return (n - 1) / n


# Measured phase-payload table for n=2 (experiments/gpu-sim-pkb/phase_payload_table.md).
# S (bytes) -> pieces per phase per direction.
# Measured: S=4096 -> P=1; S=65536 -> P=4; S=262144 -> P=4; S>=524288 -> P=8.
_N2_P_TABLE = [(4096, 1), (262144, 4), (2097152, 8)]


def _derived_pieces_per_phase(n: int, S: int) -> int:
    if n == 2:
        for threshold, P in _N2_P_TABLE:
            if S <= threshold:
                return P
        raise ValueError(f"n=2, S={S} outside measured range; no cap-based fallback")
    # n>=3: not yet measured; refuse rather than derive from cap
    raise ValueError(f"n={n} pieces_per_phase not measured; provide explicitly")


def predict(spec: Any, algorithm: str, count: int, dtype: str,
            n_ranks: Optional[int] = None,
            pieces_per_phase: Optional[int] = None) -> Dict[str, Any]:
    """Predicted duration for a collective.

    pieces_per_phase (P): number of fabric pieces per logical phase per rank.
    If None (default), P is LOOKED UP from the measured phase-payload table
    (experiments/gpu-sim-pkb/phase_payload_table.md), not derived from the
    fabric's 1 MiB payload cap. For n=2 the measured entries are: 1 at S<=4 KiB,
    4 at 64 KiB <= S <= 256 KiB, 8 at 512 KiB <= S <= 2 MiB. For n>=3 the
    table is unmeasured and _derived_pieces_per_phase raises ValueError (a
    refusal, not a derived number); callers must pass pieces_per_phase explicitly
    or the model refuses. Pass an int to override (recorded in the result as
    pieces_per_phase_override=True).
    With per-isend accounting (pkb candidate A, the operating model), each
    piece pays the link latency, so the latency term scales by P. P depends on
    payload size and the fabric's 1 MiB payload cap (a SIMULATOR constant), so
    alpha (latency_us) is a per-piece charge, not a per-hop network property,
    when P > 1. See gpu-sim-8nt for the fidelity fix that makes alpha a
    per-hop property again.
    """
    override = pieces_per_phase is not None
    if pieces_per_phase is None:
        # derived; n not known yet, placeholder until n resolved below
        pieces_per_phase = 1  # placeholder, resolved after n is known
    if pieces_per_phase < 1:
        pieces_per_phase = 1
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm}")
    dtype = dtype.lower()
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype}")
    S = count * DTYPE_BYTES[dtype]

    ranks = spec.get("ranks", [])
    if n_ranks is None:
        n_ranks = len(ranks) if ranks else None
    if not n_ranks or n_ranks < 1:
        raise ValueError("n_ranks must be >= 1")
    n = n_ranks
    # Derive P if not explicitly provided (override).
    if not override:
        S_tmp = count * DTYPE_BYTES[dtype]
        pieces_per_phase = _derived_pieces_per_phase(n, S_tmp)
    if n == 1:
        # trivial
        return {
            "duration_s": 0.0,
            "algorithm": algorithm,
            "bottleneck": {"link_id": None, "kind": None},
            "per_link_bytes": {},
            "regime": "bandwidth",
        }

    dev_idx = _dev_index(spec)
    adj = _adjacency(spec, dev_idx)
    links_spec = spec["links"]
    counts = _shared_counts(spec)
    link_latency = [float(l["latency_us"]) for l in links_spec]

    # map rank -> device index. Assume ranks sorted by rank id; device from spec.
    # Build list of device indices in rank order.
    ranks_sorted = sorted(ranks, key=lambda r: r["rank"]) if ranks else []
    if len(ranks_sorted) >= n:
        dev_for_rank = [dev_idx[(r["node"], r["device"])] for r in ranks_sorted[:n]]
    else:
        # fallback: first n devices
        dev_for_rank = list(range(n))

    logical_edges = _logical_edges(algorithm, n)

    # For each logical edge compute path
    edge_paths: List[List[int]] = []
    edge_bw: List[float] = []
    edge_lat_us: List[float] = []
    for (u_rank, v_rank) in logical_edges:
        u = dev_for_rank[u_rank]
        v = dev_for_rank[v_rank]
        path = _shortest_path(adj, link_latency, u, v)
        if path is None:
            raise ValueError(f"no path between rank {u_rank} and {v_rank}")
        edge_paths.append(path)
        edge_bw.append(_path_bw(links_spec, path, counts))
        edge_lat_us.append(_path_latency(links_spec, path))

    # worst edge for formula: min bw, then max latency
    worst_bw = min(edge_bw) if edge_bw else float("inf")
    worst_lat_us = max(edge_lat_us) if edge_lat_us else 0.0
    # but for multi-hop test we want the single path's bw/lat; for uniform ring all same.
    # If edges differ, use min bw and corresponding latency? Use the edge with min bw; tie-break max lat.
    worst_idx = 0
    for i in range(1, len(edge_bw)):
        if edge_bw[i] < edge_bw[worst_idx] or (edge_bw[i] == edge_bw[worst_idx] and edge_lat_us[i] > edge_lat_us[worst_idx]):
            worst_idx = i
    B_gbps = edge_bw[worst_idx]
    alpha_s = edge_lat_us[worst_idx] * 1e-6

    # bytes per second
    B_Bps = B_gbps * 1e9 / 8.0

    if algorithm == "tree_bcast":
        lf_t = (math.log2(n) if n > 1 else 1.0)
        bw_term = lf_t * S / B_Bps
        lat_term = lf_t * alpha_s * pieces_per_phase
        dur = bw_term + lat_term
        lat_factor = lf_t
    else:
        bf = _bw_factor(algorithm, n)
        lf = _latency_factor(algorithm, n)
        bw_term = bf * S / B_Bps
        lat_term = lf * alpha_s * pieces_per_phase
        dur = bw_term + lat_term

    # per_link_bytes
    per_link_bytes: Dict[int, int] = {}
    f = _factor(algorithm, n)
    per_edge_bytes = int(f * S) if algorithm != "tree_bcast" else S
    if algorithm == "tree_bcast":
        per_edge_bytes = S
    for path in edge_paths:
        for li in path:
            per_link_bytes[li] = per_link_bytes.get(li, 0) + per_edge_bytes

    # bottleneck link id: the physical link on the worst path with min bw (or max lat contribution)
    worst_path = edge_paths[worst_idx]
    if worst_path:
        # bottleneck is the link with min eff bw along that path, tie max latency
        b_link = worst_path[0]
        for li in worst_path[1:]:
            if _eff_bw(links_spec[li], counts) < _eff_bw(links_spec[b_link], counts):
                b_link = li
            elif _eff_bw(links_spec[li], counts) == _eff_bw(links_spec[b_link], counts) and links_spec[li]["latency_us"] > links_spec[b_link]["latency_us"]:
                b_link = li
    else:
        b_link = None

    bottleneck_kind = links_spec[b_link]["kind"] if b_link is not None else None

    # regime
    if dur <= 0:
        regime = "bandwidth"
    else:
        if lat_term > bw_term:
            regime = "latency"
        else:
            regime = "bandwidth"

    return {
        "duration_s": dur,
        "pieces_per_phase": pieces_per_phase,
        "pieces_per_phase_derived": not override,
        "pieces_per_phase_override": override,
        "algorithm": algorithm,
        "bottleneck": {"link_id": b_link, "kind": bottleneck_kind},
        "per_link_bytes": per_link_bytes,
        "regime": regime,
        "_debug": {"bw_term": bw_term, "lat_term": lat_term, "worst_bw_gbps": B_gbps, "worst_lat_us": edge_lat_us[worst_idx]},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True)
    ap.add_argument("--op", required=True, choices=sorted(ALGORITHMS))
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-ranks", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        spec = _load_spec(args.topology)
    except Exception as e:
        print(f"invalid topology: {e}", file=sys.stderr)
        return 1
    try:
        from epic_cluster.spec import validate as _validate
        _validate(spec)
    except ImportError:
        pass
    except Exception as e:
        print(f"invalid topology: {e}", file=sys.stderr)
        return 1
    # minimal structural validation
    for k in ("version", "nodes", "links", "ranks"):
        if k not in spec:
            print(f"invalid topology: missing {k}", file=sys.stderr)
            return 1

    try:
        rec = predict(spec, args.op, args.count, args.dtype, args.n_ranks)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    # remove debug key
    rec.pop("_debug", None)
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
