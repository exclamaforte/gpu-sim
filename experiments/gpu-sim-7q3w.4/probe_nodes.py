#!/usr/bin/env python3
"""D2 first measured line: eight node processes, each with its OWN eight GPUs.

A1 proved one process sees eight B300s.  D2's extra claim is isolation: node n's
process must see node n's eight PCI bus ids and must not see node m's.  The
shims read VDEV_NODE and offset the PCI bus base by 0x20 per node, so the
identity of a device is a function of (node, local index) -- the same single
source, D1 specs/b300_8x8.json, decides the base.

Eight children run CONCURRENTLY (the design's real shape: one process per
simulated node).  Each enumerates through the same three shims as
probe_vdev.py and prints one NODE line.  No arithmetic is run, so the GPU is
only enumerated; the parent checks the union of the 64 bus ids is 64 distinct
ids, which is exactly "no node can see another node's device".

    python3 experiments/gpu-sim-7q3w.4/probe_nodes.py [--nodes 8]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_vdev import BUILD, REPO, build  # noqa: E402  (same build, one shim set)

CHILD = r'''
import ctypes, json, os, resource
import torch

node = int(os.environ["VDEV_NODE"])
out = {"node": node}
out["device_count"] = torch.cuda.device_count()
out["names"] = [torch.cuda.get_device_name(i) for i in range(out["device_count"])]

# NVML: the per-node PciInfo.  nvmlPciInfo_t is the v3 layout from nvml.h:
# char busIdLegacy[16]; uint domain, bus, device, pciDeviceId, pciSubSystemId;
# char busId[32].
class PciInfo(ctypes.Structure):
    _fields_ = [
        ("busIdLegacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pciDeviceId", ctypes.c_uint),
        ("pciSubSystemId", ctypes.c_uint),
        ("busId", ctypes.c_char * 32),
    ]

nv = ctypes.CDLL("libnvidia-ml.so.1")
nv.nvmlInit_v2.restype = ctypes.c_int
nv.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
nv.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
nv.nvmlDeviceGetPciInfo_v3.argtypes = [ctypes.c_void_p, ctypes.POINTER(PciInfo)]
nv.nvmlDeviceGetPciInfo_v3.restype = ctypes.c_int
out["nvml_init_rc"] = int(nv.nvmlInit_v2())
out["nvml_pci"] = []
for i in range(out["device_count"]):
    dev = ctypes.c_void_p()
    rc = nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(dev))
    if rc != 0:
        out["nvml_pci"].append(None)
        continue
    pci = PciInfo()
    rc = nv.nvmlDeviceGetPciInfo_v3(dev, ctypes.byref(pci))
    out["nvml_pci"].append(pci.busId.decode() if rc == 0 else None)

# cudart: the same bus id through the layer torch's properties use, so a
# mismatch between the two shims is a failure rather than a curiosity.
# CDLL(None) resolves in the global scope, i.e. the preloaded libcudart-vdev:
# a direct handle to libcudart.so.13 binds its own internal symbol and would
# measure the real runtime's one device, which is the trap this check exists for.
rt = ctypes.CDLL(None)
rt.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
rt.cudaDeviceGetPCIBusId.restype = ctypes.c_int
out["cudart_pci"] = []
for i in range(out["device_count"]):
    buf = ctypes.create_string_buffer(32)
    rc = rt.cudaDeviceGetPCIBusId(buf, 32, i)
    out["cudart_pci"].append(buf.value.decode() if rc == 0 else None)

out["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
with open("/proc/meminfo") as f:
    for line in f:
        if line.startswith("MemAvailable:"):
            out["mem_available_mib"] = int(line.split()[1]) // 1024
print("NODE " + json.dumps(out))
'''


def bus_number(bus_id):
    """'00000000:0B:00.0' or '0000:0B:00.0' -> 0x0b."""
    if not bus_id:
        return None
    try:
        return int(bus_id.split(":")[1], 16)
    except (IndexError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8)
    args = ap.parse_args()
    build()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(BUILD) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["LD_PRELOAD"] = str(BUILD / "libcudart-vdev.so.1")
    env["VDEV_NVDEV"] = "8"

    procs = []
    with tempfile.TemporaryDirectory(dir=str(REPO / ".lab" / "tmp")) as td:
        for n in range(args.nodes):
            e = dict(env)
            e["VDEV_NODE"] = str(n)
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", CHILD],
                    env=e,
                    cwd=td,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        outputs = []
        for n, p in enumerate(procs):
            out, err = p.communicate(timeout=300)
            line = [ln for ln in out.splitlines() if ln.startswith("NODE ")]
            if not line:
                print(f"node {n}: rc={p.returncode}\n{out[-1500:]}\n{err[-1500:]}")
                return 1
            outputs.append(json.loads(line[0][5:]))

    # D1 is the only source of the PCI base and the per-node stride; the
    # shims include the generated b300_spec.h from the same fields.
    spec = json.loads((REPO / "specs" / "b300_8x8.json").read_text())
    model = next(iter(spec["models"].values()))
    base = model["pci_bus_base"]
    stride = model["pci_bus_node_stride"]

    expected = []
    for n in range(args.nodes):
        for i in range(8):
            expected.append(base + n * stride + i)

    got_nvml = [bus_number(b) for o in outputs for b in o["nvml_pci"]]
    got_cudart = [bus_number(b) for o in outputs for b in o["cudart_pci"]]
    per_node = {
        o["node"]: {
            "count": o["device_count"],
            "names_ok": all("B300" in x for x in o["names"]),
            "nvml_buses": [bus_number(b) for b in o["nvml_pci"]],
            "cudart_buses": [bus_number(b) for b in o["cudart_pci"]],
            "first_bus": bus_number(o["nvml_pci"][0]) if o["nvml_pci"] else None,
            "peak_rss_mib": o["peak_rss_mib"],
            "mem_available_mib": o.get("mem_available_mib"),
        }
        for o in outputs
    }

    checks = {
        "eight_node_processes": len(outputs) == args.nodes,
        "every_node_sees_8": all(v["count"] == 8 for v in per_node.values()),
        "every_node_names_b300": all(v["names_ok"] for v in per_node.values()),
        "nvml_and_cudart_agree": got_nvml == got_cudart,
        "64_distinct_pci_bus_ids": len(set(got_nvml)) == 8 * args.nodes,
        "bus_set_is_d1_exact": sorted(got_nvml) == sorted(expected),
        "nodes_have_distinct_first_bus": len({v["first_bus"] for v in per_node.values()})
        == args.nodes,
        "node_n_first_bus": all(
            per_node[n]["first_bus"] == base + n * stride for n in range(args.nodes)
        ),
    }
    result = {
        "nodes": per_node,
        "expected_bus_ids": expected,
        "checks": checks,
        "total_peak_rss_mib": sum(v["peak_rss_mib"] for v in per_node.values()),
        "min_mem_available_mib": min(
            v["mem_available_mib"] for v in per_node.values()
        ),
    }
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        print("FAIL: " + ", ".join(k for k, v in checks.items() if not v))
        return 1
    print(
        "NODES OK: 8 processes x 8 B300, 64 distinct bus ids "
        f"({base:#04x}+n*{stride:#x}); total_peak_rss={result['total_peak_rss_mib']} MiB "
        f"min_mem_available={result['min_mem_available_mib']} MiB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
