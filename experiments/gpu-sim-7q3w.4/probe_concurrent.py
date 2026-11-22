#!/usr/bin/env python3
"""7q3w.4 deliverable (1): eight CONCURRENT node processes and the RAM preflight.

The design is 8 node processes, each hosting 8 ranks.  The capacity probe measured
the footprint that matters: torch + a 12.6M-parameter float32 transformer + AdamW,
one CPU step, ~1148 MiB peak RSS (probe_capacity.py, measured after correcting
the shape; the earlier 936 MiB was the 3.16M-parameter shape).  Eight of those
is 9184 MiB (8.97 GiB), so a launcher that starts without checking MemAvailable
is making a statement about the neighbours, not about the design.

This probe:
  1. reads MemAvailable and REFUSES to launch if it is below
     8 * 1148 MiB + a stated 1024 MiB margin = 10208 MiB;
  2. launches all eight children CONCURRENTLY, VDEV_NODE=0..7, each importing
     torch, building the same 12M transformer + AdamW and taking one CPU step,
     and reporting device_count, device name, its own RSS and MemAvailable;
  3. reads MemAvailable again after the last child exits and prints the sum.

    python3 experiments/gpu-sim-7q3w.4/probe_concurrent.py
    python3 experiments/gpu-sim-7q3w.4/probe_concurrent.py --required-mib 999999  # refusal arm

No GPU arithmetic is run here; the model is CPU-only so the RSS is the design's
real footprint.  Increment (2) (torch.distributed) starts only after this posts.
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
from probe_vdev import BUILD, REPO, build  # noqa: E402

# The stated budget.  PER_NODE_MIB is the measured capacity-probe figure for
# torch + the 12.6M-parameter transformer + AdamW after one CPU step: 1148 MiB
# measured 2026-11-22 (the earlier 936 MiB belonged to the 3.16M shape).  The
# margin is for allocator/stack/import jitter and the parent.
PER_NODE_MIB = 1148
MARGIN_MIB = 1024
NODES = 8
REQUIRED_MIB = NODES * PER_NODE_MIB + MARGIN_MIB

CHILD = r'''
import json, os, resource
import torch
import torch.nn as nn

node = int(os.environ["VDEV_NODE"])
torch.manual_seed(0)
# The same ~12M-parameter shape probe_capacity.py measured, CPU-only: this is a
# memory probe, not the training run (increment 3 owns steps/loss/checkpoint).
m = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                               batch_first=True), num_layers=4)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randn(8, 64, 512)
y = m(x).sum()
y.backward()
opt.step()
out = {
    "node": node,
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "params": sum(p.numel() for p in m.parameters()),
    "rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
}
with open("/proc/meminfo") as f:
    for line in f:
        if line.startswith("MemAvailable:"):
            out["mem_available_mib"] = int(line.split()[1]) // 1024
print("NODE " + json.dumps(out))
'''


def mem_available_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--required-mib", type=int, default=REQUIRED_MIB,
                    help=f"preflight floor (default {REQUIRED_MIB} = "
                         f"{NODES}*{PER_NODE_MIB} + {MARGIN_MIB} margin)")
    args = ap.parse_args()

    start = mem_available_mib()
    if start < args.required_mib:
        print(
            f"REFUSE: MemAvailable {start} MiB < required {args.required_mib} MiB "
            f"({NODES}*{PER_NODE_MIB} + {MARGIN_MIB} margin); no child launched. "
            f"measured_start={start} MiB"
        )
        return 2

    build()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(BUILD) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["LD_PRELOAD"] = (str(BUILD / "libnvidia-ml.so.1") + ":"
                        + str(BUILD / "libcudart-vdev.so.1"))
    env["VDEV_NVDEV"] = "8"

    procs = []
    with tempfile.TemporaryDirectory(dir=str(REPO / ".lab" / "tmp")) as td:
        for n in range(NODES):
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
        nodes = []
        for n, p in enumerate(procs):
            out, err = p.communicate(timeout=600)
            line = [ln for ln in out.splitlines() if ln.startswith("NODE ")]
            if not line:
                print(f"node {n}: rc={p.returncode}\n{out[-1500:]}\n{err[-1500:]}")
                return 1
            nodes.append(json.loads(line[0][5:]))
    end = mem_available_mib()

    d1 = next(iter(json.loads((REPO / "specs" / "b300_8x8.json").read_text())["models"].values()))
    rss_sum = sum(n["rss_mib"] for n in nodes)
    checks = {
        "eight_processes": len(nodes) == NODES,
        "every_process_sees_8": all(n["device_count"] == 8 for n in nodes),
        "every_process_names_D1": all(n["device_name"] == d1["name"] for n in nodes),
        "nodes_are_0_to_7": sorted(n["node"] for n in nodes) == list(range(NODES)),
        "per_node_rss_measured": all(n["rss_mib"] > 0 for n in nodes),
        "rss_sum_recorded": rss_sum > 0,
        "mem_available_recorded": start > 0 and end > 0,
        "preflight_passed": start >= args.required_mib,
    }
    result = {
        "nodes": nodes,
        "rss_per_node_mib": [n["rss_mib"] for n in nodes],
        "rss_sum_mib": rss_sum,
        "budget_mib": {"per_node": PER_NODE_MIB, "nodes": NODES,
                       "sum": NODES * PER_NODE_MIB, "margin": MARGIN_MIB,
                       "required": args.required_mib},
        "mem_available_mib_start": start,
        "mem_available_mib_end": end,
        "mem_available_delta_mib": end - start,
        "min_child_mem_available_mib": min(n.get("mem_available_mib", 0) for n in nodes),
        "checks": checks,
    }
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        print("FAIL: " + ", ".join(k for k, v in checks.items() if not v))
        return 1
    print(
        f"CONCURRENT OK: {NODES} processes device_count=8 name={d1['name']} "
        f"rss_sum={rss_sum} MiB (budget {NODES * PER_NODE_MIB} MiB) "
        f"mem_available_start={start} MiB end={end} MiB "
        f"preflight_required={args.required_mib} MiB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
