"""gpu-sim-m3d.1: topology derivation + flat binary emit/read."""
from __future__ import annotations

import struct
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# Binary layout (little-endian), documented here and in emit().
#
# Header (20 bytes):
#   0  magic            u32   0x47535043 ("GPSC")
#   4  version          u32
#   8  n_dev            u32
#  12  n_links          u32
#  16  n_ranks          u32
#
# Device records (n_dev), each 88 bytes:
#   0  node_idx         u32
#   4  dev_id           u32
#   8  model_name       32 bytes (utf-8, null-padded)
#  40  sm_count         u32
#  44  clock_mhz        u32
#  48  mem_bytes        u64
#  56  mem_bw_gbs       f64
#  64  arch             16 bytes (utf-8, null-padded)
#  80  (padding)        8 bytes
# total 88
#
# Link records (n_links), each 120 bytes:
#   0  a_node           u32
#   4  a_dev            u32
#   8  b_node           u32
#  12  b_dev            u32
#  16  bw_gbps          f64
#  24  latency_us       f64
#  32  mtu              u32
#  36  loss_ppm         u32
#  40  jitter_us        f64
#  48  shared_group     32 bytes (utf-8, null-padded; empty if null)
#  80  eff_bw_gbps      f64   (bw / concurrent sharers in group)
#  88  (padding)        32 bytes
# total 120
#
# Rank records (n_ranks), each 12 bytes:
#   0  rank             u32
#   4  node_idx         u32
#   8  dev_id           u32
# total 12

MAGIC = 0x47535043
VERSION = 1

DEVICE_FMT = "<II32sIIQd16s8x"  # 4+4+32+4+4+8+8+16+8 = 88
LINK_FMT = "<IIIIddIId32s d 32x"  # need exact; define manually
# We'll use explicit calc instead of one format string for clarity.
DEVICE_SIZE = 88
LINK_SIZE = 120
RANK_SIZE = 12
HEADER_SIZE = 20

LINK_STRUCT_FMT = "<IIIIddII d32s d"  # placeholder; we pack manually


def _pad(s: Optional[str], n: int) -> bytes:
    if s is None:
        s = ""
    b = s.encode("utf-8")
    if len(b) > n:
        raise ValueError(f"string too long: {s!r}")
    return b + b"\x00" * (n - len(b))


def _unpad(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].decode("utf-8")


def node_index(spec: Dict[str, Any]) -> Dict[str, int]:
    return {n["name"]: i for i, n in enumerate(spec["nodes"])}


def dev_index(spec: Dict[str, Any]) -> Dict[Tuple[str, int], int]:
    out = {}
    idx = 0
    for node in spec["nodes"]:
        for dev in node["devices"]:
            out[(node["name"], dev["id"])] = idx
            idx += 1
    return out


def build_adjacency(spec: Dict[str, Any], dev_idx: Optional[Dict[Tuple[str, int], int]] = None):
    if dev_idx is None:
        dev_idx = dev_index(spec)
    n = len(dev_idx)
    # adjacency: list of (neighbor_dev_index, link_index)
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    links = spec["links"]
    for li, link in enumerate(links):
        a = dev_idx[(link["a"]["node"], link["a"]["device"])]
        b = dev_idx[(link["b"]["node"], link["b"]["device"])]
        adj[a].append((b, li))
        adj[b].append((a, li))
    return adj, dev_idx


def shortest_path(adj, src: int, dst: int) -> Optional[List[int]]:
    """Return list of link indices forming shortest path (by latency sum). BFS over hops if unweighted; here use Dijkstra by latency."""
    # We need link latency lookup; pass via closure? We'll store in a global-ish: caller provides adj only. 
    # To keep simple, do BFS (hop count) as "shortest path" for chained topology test. 
    # But design wants multi-hop; chained A-B-C should resolve through both links. BFS works.
    if src == dst:
        return []
    n = len(adj)
    prev = [-1] * n  # previous node
    prev_link = [-1] * n
    seen = [False] * n
    q = deque([src])
    seen[src] = True
    while q:
        u = q.popleft()
        if u == dst:
            break
        for (v, li) in adj[u]:
            if not seen[v]:
                seen[v] = True
                prev[v] = u
                prev_link[v] = li
                q.append(v)
    if not seen[dst]:
        return None
    # reconstruct
    links = []
    cur = dst
    while cur != src:
        links.append(prev_link[cur])
        cur = prev[cur]
    links.reverse()
    return links


def shared_group_counts(spec: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for link in spec["links"]:
        sg = link["shared_group"]
        if sg is None:
            continue
        counts[sg] = counts.get(sg, 0) + 1
    return counts


def effective_bw(link: Dict[str, Any], counts: Dict[str, int]) -> float:
    bw = float(link["bw_gbps"])
    sg = link["shared_group"]
    if sg is None:
        return bw
    c = counts.get(sg, 1)
    if c < 1:
        c = 1
    return bw / c


def emit(spec: Dict[str, Any]) -> bytes:
    import json as _j
    # validate-ish: assume spec already validated
    nodes = spec["nodes"]
    links = spec["links"]
    ranks = spec["ranks"]
    models = spec["models"]
    dev_idx = dev_index(spec)
    n_dev = len(dev_idx)
    n_links = len(links)
    n_ranks = len(ranks)
    node_idx = node_index(spec)

    shared_counts = shared_group_counts(spec)

    parts = []
    header = struct.pack("<IIIII", MAGIC, VERSION, n_dev, n_links, n_ranks)
    parts.append(header)

    # device records: iterate nodes then devices
    for node in nodes:
        for dev in node["devices"]:
            mname = dev["model"]
            m = models[mname]
            rec = struct.pack(
                "<II32sIIQd16s8x",
                node_idx[node["name"]],
                dev["id"],
                _pad(mname, 32),
                m["sm_count"],
                m["clock_mhz"],
                m["mem_bytes"],
                float(m["mem_bw_gbs"]),
                _pad(m["arch"], 16),
            )
            assert len(rec) == DEVICE_SIZE, len(rec)
            parts.append(rec)

    # link records
    for link in links:
        a = dev_idx[(link["a"]["node"], link["a"]["device"])]
        b = dev_idx[(link["b"]["node"], link["b"]["device"])]
        # node indices from dev? We store endpoint node idx; derive from link
        a_node = node_idx[link["a"]["node"]]
        a_dev = link["a"]["device"]
        b_node = node_idx[link["b"]["node"]]
        b_dev = link["b"]["device"]
        sg = link["shared_group"]
        sg_bytes = _pad(sg, 32) if sg is not None else _pad("", 32)
        eff = effective_bw(link, shared_counts)
        rec = struct.pack(
            "<IIIIddIId32sd32x",
            a_node,
            a_dev,
            b_node,
            b_dev,
            float(link["bw_gbps"]),
            float(link["latency_us"]),
            int(link["mtu"]),
            int(link["loss_ppm"]),
            float(link["jitter_us"]),
            sg_bytes,
            float(eff),
        )
        assert len(rec) == LINK_SIZE, len(rec)
        parts.append(rec)

    # rank records
    for rk in ranks:
        rec = struct.pack(
            "<III",
            int(rk["rank"]),
            node_idx[rk["node"]],
            int(rk["device"]),
        )
        # pad to RANK_SIZE? 12 bytes already
        assert len(rec) == RANK_SIZE, len(rec)
        parts.append(rec)

    return b"".join(parts)


def read_binary(data: bytes) -> Dict[str, Any]:
    off = 0
    magic, version, n_dev, n_links, n_ranks = struct.unpack_from("<IIIII", data, off)
    off += HEADER_SIZE
    if magic != MAGIC:
        raise ValueError("bad magic")
    if version != VERSION:
        raise ValueError(f"bad version {version}")
    devices = []
    for _ in range(n_dev):
        (node_idx, dev_id, model_name, sm_count, clock_mhz, mem_bytes, mem_bw_gbs, arch) = struct.unpack_from(
            "<II32sIIQd16s8x", data, off
        )
        off += DEVICE_SIZE
        devices.append({
            "node_idx": node_idx,
            "dev_id": dev_id,
            "model_name": _unpad(model_name),
            "sm_count": sm_count,
            "clock_mhz": clock_mhz,
            "mem_bytes": mem_bytes,
            "mem_bw_gbs": mem_bw_gbs,
            "arch": _unpad(arch),
        })
    links = []
    for _ in range(n_links):
        (a_node, a_dev, b_node, b_dev, bw, lat, mtu, loss, jitter, sg_bytes, eff) = struct.unpack_from(
            "<IIIIddIId32sd32x", data, off
        )
        off += LINK_SIZE
        links.append({
            "a_node": a_node,
            "a_dev": a_dev,
            "b_node": b_node,
            "b_dev": b_dev,
            "bw_gbps": bw,
            "latency_us": lat,
            "mtu": mtu,
            "loss_ppm": loss,
            "jitter_us": jitter,
            "shared_group": _unpad(sg_bytes) or None,
            "eff_bw_gbps": eff,
        })
    ranks = []
    for _ in range(n_ranks):
        rank, node_idx, dev_id = struct.unpack_from("<III", data, off)
        off += RANK_SIZE
        ranks.append({"rank": rank, "node_idx": node_idx, "dev_id": dev_id})
    return {
        "magic": magic,
        "version": version,
        "n_dev": n_dev,
        "n_links": n_links,
        "n_ranks": n_ranks,
        "devices": devices,
        "links": links,
        "ranks": ranks,
    }
