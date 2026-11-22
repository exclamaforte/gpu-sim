#!/usr/bin/env python3
"""Increment (2): torch.distributed with world_size 8, one rank per node.

Eight processes start concurrently, VDEV_NODE=0..7, and each one is rank
0..7 of a gloo process group.  Each rank prints its node index and the eight
PCI bus ids it sees through the shims.  The group then does a real
``all_reduce`` (SUM of rank+1), which must produce 36.0 on every rank.  A
disagreement is a result, not a crash: the checker reports the per-rank values
verbatim and exits non-zero, and ``--inject-disagreement`` is the negative
control that proves the checker can see one.

The all_reduce here is the gloo (CPU) control plane.  It does NOT claim the
NCCL data path: that is 7q3w.3/7q3w.5.  The point of this increment is the
process group plus the per-node identity that the NCCL ranks will later use.

The same D1 file is the only source of the PCI base/stride; the expected bus
set is computed from specs/b300_8x8.json, never from a literal.

    python3 experiments/gpu-sim-7q3w.4/probe_distributed.py
    python3 experiments/gpu-sim-7q3w.4/probe_distributed.py --inject-disagreement
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_vdev import BUILD, REPO, build  # noqa: E402

NODES = 8
# One rank is torch import (543 MiB measured by probe_capacity.py) plus gloo
# threads/buffers; the margin is for allocator/stack/import jitter and the
# parent.  This is a floor, not a prediction: the measured RSS is reported.
PER_RANK_MIB = 600
MARGIN_MIB = 1024
REQUIRED_MIB = NODES * PER_RANK_MIB + MARGIN_MIB

CHILD = r'''
import ctypes, json, os, resource
import torch
import torch.distributed as dist

node = int(os.environ["VDEV_NODE"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
out = {"node": node, "rank": rank, "world_size": world_size, "backend": "gloo"}
out["device_count"] = torch.cuda.device_count()
out["names"] = [torch.cuda.get_device_name(i) for i in range(out["device_count"])]

# NVML: the per-node PciInfo.  nvmlPciInfo_t is the v3 layout from nvml.h.
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

# CDLL(None) resolves in the global scope, i.e. the preloaded libcudart-vdev;
# a direct handle to libcudart.so.13 binds its own internal symbol (the trap
# documented in findings.md sec. 7).
rt = ctypes.CDLL(None)
rt.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
rt.cudaDeviceGetPCIBusId.restype = ctypes.c_int
out["cudart_pci"] = []
for i in range(out["device_count"]):
    buf = ctypes.create_string_buffer(32)
    rc = rt.cudaDeviceGetPCIBusId(buf, 32, i)
    out["cudart_pci"].append(buf.value.decode() if rc == 0 else None)

# The process group.  env:// uses MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE set
# by the parent.  gloo is CPU; no NCCL transport claim is made here.
dist.init_process_group(backend="gloo", init_method="env://")
out["pg_initialized"] = dist.is_initialized()
out["pg_world_size"] = dist.get_world_size()
out["pg_rank"] = dist.get_rank()

# The agreement test: SUM(1..8) = 36.0 on every rank, or a disagreement.
t = torch.tensor([float(rank + 1)], dtype=torch.float64)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
value = float(t.item())
if os.environ.get("INJECT_DISAGREEMENT") == "1" and rank == 3:
    value += 1.0
out["all_reduce_value"] = value

# Every rank sees every rank's node/bus identity and value.  This is how a
# disagreement is printed, not hidden: the values travel with their owners.
world = [None for _ in range(world_size)]
dist.all_gather_object(
    world, {"node": node, "buses": out["nvml_pci"], "value": value}
)
out["world"] = world
dist.barrier()
dist.destroy_process_group()

out["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
with open("/proc/meminfo") as f:
    for line in f:
        if line.startswith("MemAvailable:"):
            out["mem_available_mib"] = int(line.split()[1]) // 1024
print("DIST " + json.dumps(out))
'''


def mem_available_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not in /proc/meminfo")


def bus_number(bus_id):
    """'00000000:0B:00.0' or '0000:0B:00.0' -> 0x0b."""
    if not bus_id:
        return None
    try:
        return int(bus_id.split(":")[1], 16)
    except (IndexError, ValueError):
        return None


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=NODES)
    ap.add_argument("--inject-disagreement", action="store_true")
    args = ap.parse_args()

    start_mem = mem_available_mib()
    if start_mem < REQUIRED_MIB:
        print(
            f"REFUSE: MemAvailable {start_mem} MiB < required {REQUIRED_MIB} MiB "
            f"({args.nodes}*{PER_RANK_MIB} + {MARGIN_MIB} margin); "
            f"no rank launched. measured_start={start_mem} MiB"
        )
        return 2

    build()
    port = free_port()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(BUILD) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["LD_PRELOAD"] = (str(BUILD / "libnvidia-ml.so.1") + ":"
                        + str(BUILD / "libcudart-vdev.so.1"))
    env["VDEV_NVDEV"] = "8"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    env["WORLD_SIZE"] = str(args.nodes)
    if args.inject_disagreement:
        env["INJECT_DISAGREEMENT"] = "1"

    procs = []
    with tempfile.TemporaryDirectory(dir=str(REPO / ".lab" / "tmp")) as td:
        for n in range(args.nodes):
            e = dict(env)
            e["VDEV_NODE"] = str(n)
            e["RANK"] = str(n)
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
        failures = []
        for n, p in enumerate(procs):
            out, err = p.communicate(timeout=300)
            line = [ln for ln in out.splitlines() if ln.startswith("DIST ")]
            if not line:
                failures.append(
                    f"rank {n}: rc={p.returncode}\n{out[-1200:]}\n{err[-1200:]}"
                )
                continue
            outputs.append(json.loads(line[0][5:]))

    end_mem = mem_available_mib()
    if failures or len(outputs) != args.nodes:
        print("\n".join(failures))
        print(
            f"FAIL: {len(outputs)}/{args.nodes} ranks reported; "
            f"mem_available_start={start_mem} end={end_mem}"
        )
        return 1

    # D1 is the only source of the PCI base and the per-node stride.
    spec = json.loads((REPO / "specs" / "b300_8x8.json").read_text())
    model = next(iter(spec["models"].values()))
    base = model["pci_bus_base"]
    stride = model["pci_bus_node_stride"]
    expected = [base + n * stride + i for n in range(args.nodes) for i in range(8)]

    per_rank = {}
    for o in outputs:
        buses_nvml = [bus_number(b) for b in o["nvml_pci"]]
        buses_cudart = [bus_number(b) for b in o["cudart_pci"]]
        per_rank[o["rank"]] = {
            "node": o["node"],
            "device_count": o["device_count"],
            "names": o["names"],
            "names_ok": all("B300" in x for x in o["names"]),
            "nvml_buses": buses_nvml,
            "cudart_buses": buses_cudart,
            "nvml_cudart_agree": buses_nvml == buses_cudart,
            "all_reduce_value": o["all_reduce_value"],
            "world": o["world"],
            "peak_rss_mib": o["peak_rss_mib"],
            "mem_available_mib": o.get("mem_available_mib"),
        }

    got_nvml = [b for r in per_rank.values() for b in r["nvml_buses"]]
    values = [r["all_reduce_value"] for r in per_rank.values()]
    all_values_equal = len(set(values)) == 1
    all_values_expected = all_values_equal and values[0] == 36.0
    world_nodes_ok = all(
        {w["node"] for w in r["world"]} == set(range(args.nodes))
        for r in per_rank.values()
    )

    checks = {
        "eight_ranks": len(per_rank) == args.nodes,
        "each_rank_is_its_node": all(
            r["node"] == rank for rank, r in per_rank.items()
        ),
        "process_group_world_size_8": all(
            o["pg_world_size"] == args.nodes and o["pg_rank"] == o["rank"]
            and o["pg_initialized"]
            for o in outputs
        ),
        "every_rank_sees_8": all(r["device_count"] == 8 for r in per_rank.values()),
        "every_rank_names_b300": all(r["names_ok"] for r in per_rank.values()),
        "nvml_and_cudart_agree": all(
            r["nvml_cudart_agree"] for r in per_rank.values()
        ),
        "64_distinct_pci_bus_ids": len(set(got_nvml)) == 8 * args.nodes,
        "bus_set_is_d1_exact": sorted(got_nvml) == sorted(expected),
        "all_gather_sees_every_node": world_nodes_ok,
        "all_reduce_agrees": all_values_expected,
    }

    result = {
        "per_rank": per_rank,
        "expected_bus_ids": expected,
        "all_reduce_values": values,
        "checks": checks,
        "rss_sum_mib": sum(r["peak_rss_mib"] for r in per_rank.values()),
        "min_child_mem_available_mib": min(
            r["mem_available_mib"] for r in per_rank.values()
        ),
        "mem_available_mib_start": start_mem,
        "mem_available_mib_end": end_mem,
        "preflight_required_mib": REQUIRED_MIB,
        "injected_disagreement": args.inject_disagreement,
    }
    print(json.dumps(result, indent=2))

    if args.inject_disagreement:
        # The control succeeds when the checker SEES the disagreement.
        detected = not checks["all_reduce_agrees"] and len(set(values)) > 1
        other_ok = all(v for k, v in checks.items() if k != "all_reduce_agrees")
        if detected and other_ok:
            print(
                "DISAGREEMENT DETECTED (control): per-rank all_reduce values "
                f"{values}; expected 36.0 on every rank. This is the valuable "
                "result the checker exists for."
            )
            return 0
        print(
            "FAIL(control): injected disagreement was not detected; "
            f"values={values} checks={checks}"
        )
        return 1

    if not all(checks.values()):
        print("FAIL: " + ", ".join(k for k, v in checks.items() if not v))
        print(f"DISAGREEMENT REPORT: per-rank all_reduce values {values}; "
              "the group did NOT agree.")
        return 1
    print(
        f"DIST OK: {args.nodes} ranks x 8 B300, "
        f"{len(set(got_nvml))} distinct bus ids "
        f"({base:#04x}+n*{stride:#x}), all_reduce={values[0]} on every rank, "
        f"rss_sum={result['rss_sum_mib']} MiB, "
        f"mem_available_start={start_mem} MiB end={end_mem} MiB "
        f"preflight_required={REQUIRED_MIB} MiB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
