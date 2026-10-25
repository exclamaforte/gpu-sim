#!/usr/bin/env python3
"""Model-vs-harness byte reconciliation (gpu-sim-wrvw).

Reads a harness trace (JSONL), sums bytes per collective family, normalises by
steps and batch, and compares against comm_bytes() from epic_cluster.parallelism_search.
Prints model/measured ratios; flags any outside [0.95, 1.05].
"""
import argparse, json, os, sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from epic_cluster import parallelism_search as ps

FAMILIES = {
    "allreduce": "tp_allreduce",
    "allgather": "dp_allgather",
    "reducescatter": "dp_reducescatter",
    "send": "pp_sendrecv",
    "recv": "pp_sendrecv",
}


def read_trace(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--zero", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    rows = read_trace(args.trace)
    # Sum bytes per op family (rank 0 only to avoid double counting)
    meas = defaultdict(int)
    for r in rows:
        if r.get("rank", 0) != 0:
            continue
        op = r.get("op", "")
        meas[op] += int(r.get("bytes") or 0)

    # Normalise by steps (approx: divide by steps) and batch (model assumes 1 seq)
    # Measured trace covers `steps` steps * `batch` sequences.
    norm = args.steps * args.batch if args.steps * args.batch > 0 else 1

    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100"}
    m = ps.MODEL
    model = ps.comm_bytes(args.tp, args.pp, args.cp, args.dp, args.zero, hw, m)

    print(f"{'family':<20} {'model_bytes':>14} {'meas_bytes':>14} {'ratio':>8}")
    print("-" * 60)
    # Aggregate measured by family
    meas_family = defaultdict(int)
    for op, b in meas.items():
        fam = FAMILIES.get(op, op)
        meas_family[fam] += b

    for fam in ["tp_allreduce", "cp_exchange", "cp_grad_sync", "pp_sendrecv",
                "dp_allreduce", "dp_allgather", "dp_reducescatter"]:
        mb = model.get(fam, 0)
        raw = meas_family.get(fam, 0)
        meas_norm = raw // norm if norm > 0 else raw
        if mb == 0 and meas_norm == 0:
            print(f"{fam:<20} {mb:>14} {meas_norm:>14} {'n/a':>8}")
            continue
        ratio = (meas_norm / mb) if mb > 0 else float("inf")
        flag = "" if (0.95 <= ratio <= 1.05) else "  <-- OUT OF RANGE"
        print(f"{fam:<20} {mb:>14} {meas_norm:>14} {ratio:>8.3f}{flag}")

    print(f"\n{'total':<20} {model['total']:>14}")


if __name__ == "__main__":
    main()
