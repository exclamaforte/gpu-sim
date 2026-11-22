#!/usr/bin/env python3
"""Generate specs/b300_8x8.json -- D1, the ONE machine spec every 7q3w.4 shim reads.

The values and their sources are the PI-sourced table in the bead comment of
2026-11-22T17:08Z.  Nothing here is guessed: a field the PI marked UNVERIFIED stays
null (or carries an explicit "UNVERIFIED" source) and must not be used in an
assertion.  The per-link latency is a MODELLING CHOICE, not a vendor figure: the
PI could not source a B300 NVLink or ConnectX-8 per-hop latency and required the
derivation to be named.  D1 uses the fabric calendar's existing per-hop alphas
(1.0 us NVLink, 5.0 us IB/NIC; epic_cluster/nccl_cost.py FABRICS) and marks every
such field with that provenance; the clock remains UNVERIFIED.

    python3 experiments/gpu-sim-7q3w.4/make_b300_spec.py
    # writes specs/b300_8x8.json, deterministic (no timestamps, no wall clock)
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "specs", "b300_8x8.json")

GIB = 1 << 30
NODES = 8
GPUS_PER_NODE = 8
# MODELLING CHOICE, not vendor numbers: the lab fabric calendar's per-hop alphas.
NV_LATENCY_US = 1.0
NIC_LATENCY_US = 5.0

SOURCES = {
    "model.b300_sxm": (
        "NVIDIA Blackwell Ultra / B300: 288 GB HBM3e, 8 TB/s, 160 SMs, 1.8 TB/s "
        "NVLink, 1400 W, compute capability 10.3. "
        "developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-"
        "the-ai-factory-era/ ; developer.nvidia.com/cuda/gpus (CC 10.3 for GB300/B300)"
    ),
    "node.hgx_b300": (
        "HGX B300 / DGX B300: 8 x B300 SXM per node, 8 x 288 GB = 2304 GB, "
        "14.4 TB/s aggregate NVLink, 8 x ConnectX-8 800 Gb/s. "
        "nvidia.com/en-us/data-center/dgx-b300/ ; "
        "supermicro.com/datasheet/datasheet_SuperCluster_B300_Front_IO.pdf"
    ),
    "link.nvlink": (
        "NVIDIA B300 NVLink 5: 1.8 TB/s bidirectional per GPU (18 links x 100 GB/s). "
        "developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-"
        "the-ai-factory-era/"
    ),
    "link.nic": (
        "ConnectX-8 SuperNIC: 800 Gb/s per GPU, 1:1 GPU-to-NIC. "
        "supermicro.com/datasheet/datasheet_SuperCluster_B300_Front_IO.pdf"
    ),
    "latency.nvlink": (
        "MODELLING CHOICE, not a vendor figure: derived from the fabric calendar's "
        "per-hop alpha for NVLink, 1.0 us (epic_cluster/nccl_cost.py "
        "FABRICS['nvlink4_h100']). The B300 NVLink per-link latency is unsourced; "
        "this is a model parameter, not a measurement."
    ),
    "latency.nic": (
        "MODELLING CHOICE, not a vendor figure: derived from the fabric calendar's "
        "per-hop alpha for IB/NIC, 5.0 us (epic_cluster/nccl_cost.py "
        "FABRICS['ib_400g']). The ConnectX-8 per-hop latency is unsourced; this "
        "is a model parameter, not a measurement."
    ),
    "clock.b300_sxm": (
        "UNVERIFIED: the sourced B300 table gives no SM clock. Do not use in an "
        "assertion."
    ),
}


def model():
    return {
        "name": "NVIDIA B300",
        "uuid_prefix": "GPU-VDEV",   # D1 names the UUID family; node/device appended by the shims
        "sm_count": 160,
        "mem_bytes": 288 * GIB,
        "mem_bw_gbs": 8000,
        "compute_capability": [10, 3],
        "arch": "sm_103",
        "nvlink_bw_gbps": 1800.0,
        "pcie_gen": 6,
        "pcie_width": 16,
        "pci_bus_base": 11,          # 0x0b; every modelled device sits on this node-local base
        "pci_bus_node_stride": 32,   # 0x20; eight devices per node, next node starts 0x20 later
        "pci_bus_source": "modelled: node n device i -> bus 0x0b + 0x20*n + i; the 0x20 stride keeps the eight devices of a node contiguous and the next node disjoint", 
        "tdp_w": 1400,
        "clock_mhz": None,
        "source": SOURCES["model.b300_sxm"],
        "unverified": ["clock_mhz"],
    }


def nodes():
    out = []
    for n in range(NODES):
        out.append({
            "name": f"node{n}",
            "devices": [{"id": i, "model": "b300_sxm"}
                        for i in range(GPUS_PER_NODE)],
            "nics": [{"id": i, "kind": "connectx-8", "line_rate_gbps": 800,
                      "bw_gbs": 100.0,
                      "latency_us": NIC_LATENCY_US,
                      "latency_source": SOURCES["latency.nic"],
                      "source": SOURCES["link.nic"]}
                     for i in range(GPUS_PER_NODE)],
            "gpu_mem_bytes": GPUS_PER_NODE * 288 * GIB,
            "nvlink_aggregate_gbps": GPUS_PER_NODE * 1800.0,
            "source": SOURCES["node.hgx_b300"],
        })
    return out


def links():
    out = []
    for n in range(NODES):
        for a in range(GPUS_PER_NODE):
            for b in range(a + 1, GPUS_PER_NODE):
                out.append({
                    "a": {"node": f"node{n}", "device": a},
                    "b": {"node": f"node{n}", "device": b},
                    "kind": "nvlink",
                    "bw_gbps": 1800.0,
                    "bw_is_bidirectional": True,
                    "latency_us": NV_LATENCY_US,
                    "latency_source": SOURCES["latency.nvlink"],
                    "mtu": 4096,
                    "loss_ppm": 0,
                    "jitter_us": 0,
                    "shared_group": None,
                })
    return out


def main():
    spec = {
        "version": 1,
        "bead": "gpu-sim-7q3w.4",
        "sources": SOURCES,
        "models": {"b300_sxm": model()},
        "nodes": nodes(),
        "links": links(),
        "unverified": {
            "sm_clock_mhz": SOURCES["clock.b300_sxm"],
        },
        "modelling_choices": {
            "nvlink_latency_us": SOURCES["latency.nvlink"],
            "nic_latency_us": SOURCES["latency.nic"],
        },
    }
    with open(OUT, "w") as f:
        json.dump(spec, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {OUT}: {len(spec['nodes'])} nodes x {GPUS_PER_NODE} GPUs, "
          f"{len(spec['links'])} NVLink edges, "
          f"{sum(len(n['nics']) for n in spec['nodes'])} NICs")


if __name__ == "__main__":
    main()
