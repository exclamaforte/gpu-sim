"""gpu-sim-m3d.1: cluster topology + device-model spec (schema + validator)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_VERSION = 1

# UNIT CONVENTION (gpu-sim-3me): bw_gbps is in GIGABYTES per second (GB/s),
# NOT gigabits.  H100 SXM NVLink = 900 GB/s, NDR IB = 50 GB/s (400 Gbps).
# link_model.py converts GB/s -> bytes/s by multiplying by 1e9 (no /8).
# Every spec in the tree carries GB/s values.

REQUIRED_MODEL_KEYS = {"sm_count", "clock_mhz", "mem_bytes", "mem_bw_gbs", "arch", "source"}
# bw_is_bidirectional (gpu-sim-smt8, optional): true when bw_gbps is the AGGREGATE
# BIDIRECTIONAL figure (NVLink 4 "900 GB/s per GPU (aggregate bidirectional)"), in which
# case the one-directional rate used for a ring phase or a p2p send is half of it. Absent
# means per-direction -- NDR IB "400 Gb/s = 50 GB/s per GPU NIC" is already per direction.
OPTIONAL_LINK_KEYS = {"bw_is_bidirectional"}
REQUIRED_LINK_KEYS = {"a", "b", "kind", "bw_gbps", "latency_us", "mtu", "loss_ppm", "jitter_us", "shared_group"}
REQUIRED_ENDPOINT_KEYS = {"node", "device"}
REQUIRED_NODE_KEYS = {"name", "devices"}
REQUIRED_DEVICE_KEYS = {"id", "model"}
REQUIRED_RANK_KEYS = {"rank", "node", "device"}

TOP_LEVEL_KEYS = {"version", "models", "nodes", "links", "ranks", "sources"}
LINK_KINDS = {"nvlink", "pcie", "ib", "roce", "tcp"}


class SpecError(ValueError):
    """Validation error naming the offending JSON path."""


def _err(path: str, msg: str) -> SpecError:
    return SpecError(f"{path}: {msg}")


def _check_keys(obj: Any, allowed: set, path: str) -> None:
    if not isinstance(obj, dict):
        raise _err(path, "expected object")
    for k in obj:
        if k not in allowed:
            raise _err(f"{path}.{k}", "unknown key")


def validate(spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise _err("$", "top-level must be object")
    _check_keys(spec, TOP_LEVEL_KEYS, "$")
    if "version" not in spec:
        raise _err("$.version", "missing")
    if spec["version"] != SCHEMA_VERSION:
        raise _err("$.version", f"expected {SCHEMA_VERSION}, got {spec['version']!r}")

    models = spec.get("models")
    if not isinstance(models, dict) or not models:
        raise _err("$.models", "must be non-empty object")
    for mname, m in models.items():
        _check_keys(m, REQUIRED_MODEL_KEYS, f"$.models.{mname}")
        if not isinstance(m["sm_count"], int) or m["sm_count"] <= 0:
            raise _err(f"$.models.{mname}.sm_count", "must be positive int")
        if not isinstance(m["clock_mhz"], (int, float)) or m["clock_mhz"] <= 0:
            raise _err(f"$.models.{mname}.clock_mhz", "must be positive number")
        if not isinstance(m["mem_bytes"], int) or m["mem_bytes"] <= 0:
            raise _err(f"$.models.{mname}.mem_bytes", "must be positive int")
        if not isinstance(m["mem_bw_gbs"], (int, float)) or m["mem_bw_gbs"] <= 0:
            raise _err(f"$.models.{mname}.mem_bw_gbs", "must be positive number")
        if not isinstance(m["arch"], str) or not m["arch"]:
            raise _err(f"$.models.{mname}.arch", "must be non-empty string")
        if not isinstance(m["source"], str) or not m["source"]:
            raise _err(f"$.models.{mname}.source", "model without source (provenance required)")

    nodes = spec.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise _err("$.nodes", "must be non-empty array")
    node_names = set()
    devices: Dict[str, set] = {}  # node -> set of device ids
    for i, node in enumerate(nodes):
        _check_keys(node, REQUIRED_NODE_KEYS, f"$.nodes[{i}]")
        name = node["name"]
        if not isinstance(name, str) or not name:
            raise _err(f"$.nodes[{i}].name", "must be non-empty string")
        if name in node_names:
            raise _err(f"$.nodes[{i}].name", "duplicate node name")
        node_names.add(name)
        devs = node["devices"]
        if not isinstance(devs, list):
            raise _err(f"$.nodes[{i}].devices", "must be array")
        ids = set()
        for j, dev in enumerate(devs):
            _check_keys(dev, REQUIRED_DEVICE_KEYS, f"$.nodes[{i}].devices[{j}]")
            did = dev["id"]
            if not isinstance(did, int) or did < 0:
                raise _err(f"$.nodes[{i}].devices[{j}].id", "must be non-negative int")
            if did in ids:
                raise _err(f"$.nodes[{i}].devices[{j}].id", "duplicate device id")
            ids.add(did)
            mname = dev["model"]
            if mname not in models:
                raise _err(f"$.nodes[{i}].devices[{j}].model", f"unknown model {mname!r}")
        devices[name] = ids

    def endpoint_exists(ep, path):
        if not isinstance(ep, dict):
            raise _err(path, "expected object")
        _check_keys(ep, REQUIRED_ENDPOINT_KEYS, path)
        n = ep["node"]; d = ep["device"]
        if n not in devices:
            raise _err(f"{path}.node", f"unknown node {n!r}")
        if not isinstance(d, int) or d not in devices[n]:
            raise _err(f"{path}.device", f"unknown device {d!r} on node {n!r}")

    links = spec.get("links")
    if not isinstance(links, list):
        raise _err("$.links", "must be array")
    for i, link in enumerate(links):
        p = f"$.links[{i}]"
        _check_keys(link, REQUIRED_LINK_KEYS | OPTIONAL_LINK_KEYS, p)
        endpoint_exists(link["a"], f"{p}.a")
        endpoint_exists(link["b"], f"{p}.b")
        kind = link["kind"]
        if kind not in LINK_KINDS:
            raise _err(f"{p}.kind", f"unknown kind {kind!r}")
        bw = link["bw_gbps"]
        if not isinstance(bw, (int, float)) or isinstance(bw, bool) or bw <= 0:
            raise _err(f"{p}.bw_gbps", "must be > 0")
        lat = link["latency_us"]
        if not isinstance(lat, (int, float)) or isinstance(lat, bool) or lat < 0:
            raise _err(f"{p}.latency_us", "must be >= 0")
        mtu = link["mtu"]
        if not isinstance(mtu, int) or isinstance(mtu, bool) or mtu < 1:
            raise _err(f"{p}.mtu", "must be >= 1")
        loss = link["loss_ppm"]
        if not isinstance(loss, (int, float)) or isinstance(loss, bool) or loss < 0 or loss > 1e6:
            raise _err(f"{p}.loss_ppm", "must be in [0,1e6]")
        jitter = link["jitter_us"]
        if not isinstance(jitter, (int, float)) or isinstance(jitter, bool) or jitter < 0:
            raise _err(f"{p}.jitter_us", "must be >= 0")
        sg = link["shared_group"]
        if sg is not None and not isinstance(sg, str):
            raise _err(f"{p}.shared_group", "must be string or null")
        bib = link.get("bw_is_bidirectional")
        if bib is not None and not isinstance(bib, bool):
            raise _err(f"{p}.bw_is_bidirectional", "must be boolean if present")

    sources = spec.get("sources")
    if sources is not None and not isinstance(sources, dict):
        raise _err("$.sources", "must be object if present")

    ranks = spec.get("ranks")
    if not isinstance(ranks, list):
        raise _err("$.ranks", "must be array")
    rank_ids = set()
    rank_devices = []  # list of (node, device)
    for i, rk in enumerate(ranks):
        p = f"$.ranks[{i}]"
        _check_keys(rk, REQUIRED_RANK_KEYS, p)
        r = rk["rank"]
        if not isinstance(r, int) or isinstance(r, bool) or r < 0:
            raise _err(f"{p}.rank", "must be non-negative int")
        if r in rank_ids:
            raise _err(f"{p}.rank", "duplicate rank")
        rank_ids.add(r)
        n = rk["node"]; d = rk["device"]
        if n not in devices:
            raise _err(f"{p}.node", f"unknown node {n!r}")
        if not isinstance(d, int) or d not in devices[n]:
            raise _err(f"{p}.device", f"unknown device {d!r} on node {n!r}")
        rank_devices.append((n, d))

    # device pair used by ranks with no path -> error
    # build adjacency
    from epic_cluster.topology import build_adjacency, shortest_path

    n_dev_list = []
    for node in nodes:
        for dev in node["devices"]:
            n_dev_list.append((node["name"], dev["id"]))
    # map (node,device) -> index
    dev_index = {nd: i for i, nd in enumerate(n_dev_list)}

    # adjacency
    adj, dev_index = build_adjacency(spec, dev_index)
    # for each pair of ranks? "a device pair used by ranks with no path"
    # Check all distinct device pairs that appear among ranks have a path.
    rank_dev_indices = [dev_index[(n, d)] for (n, d) in rank_devices]
    for i in range(len(rank_dev_indices)):
        for j in range(i + 1, len(rank_dev_indices)):
            a = rank_dev_indices[i]; b = rank_dev_indices[j]
            if a == b:
                continue
            if shortest_path(adj, a, b) is None:
                raise _err("$.ranks", f"no path between devices used by ranks {i} and {j}")

    return spec


def load(path) -> Dict[str, Any]:
    p = Path(path)
    with p.open() as f:
        spec = json.load(f)
    return validate(spec)


def loads(text: str) -> Dict[str, Any]:
    return validate(json.loads(text))
