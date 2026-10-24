"""Build a declared cluster for N ranks, and load it into the interposer.

The interposer times collectives from a topology blob handed to it in
`GPUSIM_TOPOLOGY` (see src/fakegpu_nccl.c). Without one, every collective takes
the fallback path and is traced with `predicted_s: null` -- bytes but no cost, so
a sweep can say how much traffic a configuration generates but not what it costs.

This module closes that gap: it emits a spec for the declared world size, encodes
it with epic_cluster.topology.emit(), and returns the path to hand to a
subprocess. Changing fabric (NVLink vs PCIe) is then a parameter, not a code
change, which is what makes "same parallelism, different interconnect" a runnable
comparison rather than an assertion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from epic_cluster import spec as cluster_spec
from epic_cluster import topology

REPO = Path(__file__).resolve().parents[2]

# Fabric presets. The numbers are declared, not measured: they exist so two runs
# differ in a way that is visible in the trace. Replacing them with measured
# values is a one-line change per preset and touches nothing else.
# Device presets. Counts are for the DECLARED cluster: the hardware on this box is
# one consumer GPU, so an "8xH100" run is a claim about a fabric we describe, not
# one we own. That is the point of the tool and it must be labeled as such
# wherever the trace is shown.
DEVICES: dict[str, dict[str, Any]] = {
    "h100_sxm": {"sm_count": 132, "clock_mhz": 1830, "mem_bytes": 80 << 30,
                 "mem_bw_gbs": 3350, "arch": "sm_90",
                 "source": "NVIDIA H100 SXM datasheet: 132 SMs, 1830 MHz, 80GB HBM3, 3.35 TB/s"},
    "generic":  {"sm_count": 128, "clock_mhz": 1800, "mem_bytes": 32 << 30,
                 "mem_bw_gbs": 1792, "arch": "sm_120", "source": "declared placeholder"},
}

FABRICS: dict[str, dict[str, Any]] = {
    "nvlink": {"kind": "nvlink", "bw_gbps": 900.0, "latency_us": 1.0, "mtu": 4096},
    "pcie":   {"kind": "pcie",   "bw_gbps": 25.0,  "latency_us": 8.0, "mtu": 4096},
    "ib":     {"kind": "ib",     "bw_gbps": 400.0, "latency_us": 2.0, "mtu": 4096},
}


def build_spec(n_ranks: int, fabric: str = "nvlink",
               device_model: str = "generic") -> dict[str, Any]:
    """A single-node cluster of `n_ranks` devices, fully connected."""
    if n_ranks < 1:
        raise ValueError("n_ranks must be >= 1")
    if fabric not in FABRICS:
        raise ValueError(f"unknown fabric {fabric!r}; known: {sorted(FABRICS)}")
    f = FABRICS[fabric]

    links = []
    for i in range(n_ranks):
        for j in range(i + 1, n_ranks):
            links.append({
                "a": {"node": "node0", "device": i},
                "b": {"node": "node0", "device": j},
                "kind": f["kind"],
                "bw_gbps": f["bw_gbps"],
                "latency_us": f["latency_us"],
                "mtu": f["mtu"],
                "loss_ppm": 0,
                "jitter_us": 0,
                "shared_group": None,
            })

    return {
        "version": 1,
        "sources": {f"link.{f['kind']}": f"harness preset '{fabric}': declared, not measured",
                    f"model.{device_model}": "declared placeholder for a generic device"},
        "models": {device_model: DEVICES.get(device_model, DEVICES["generic"])},
        "nodes": [{"name": "node0",
                   "devices": [{"id": i, "model": device_model}
                               for i in range(n_ranks)]}],
        "links": links,
        "ranks": [{"rank": i, "node": "node0", "device": i} for i in range(n_ranks)],
    }


def emit(n_ranks: int, out_path: str | Path, fabric: str = "nvlink",
         device_model: str = "generic") -> Path:
    """Validate, encode, and write the topology blob. Returns the path."""
    spec = cluster_spec.validate(build_spec(n_ranks, fabric, device_model))
    blob = topology.emit(spec)
    out = Path(out_path)
    out.write_bytes(blob)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="emit a topology blob for N ranks")
    ap.add_argument("--ranks", type=int, required=True)
    ap.add_argument("--fabric", default="nvlink", choices=sorted(FABRICS))
    ap.add_argument("--device", default="generic", choices=sorted(DEVICES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = emit(a.ranks, a.out, a.fabric, a.device)
    print(json.dumps({"path": str(p), "ranks": a.ranks, "fabric": a.fabric,
                      "bytes": p.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
