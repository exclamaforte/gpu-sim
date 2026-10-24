"""Validate that all ranks in a run issued equal collective counts per op.

Usage:
    python -m tools.torch_parallel.validate_trace --trace T --summary S
Exits 0 if all ranks have equal counts per op (within the measured window),
non-zero (INVALID) otherwise.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict, Counter

def counts_per_rank(trace_path, summary_path):
    rows = [json.loads(l) for l in open(summary_path) if l.strip()]
    start = {r["rank"]: r.get("measure_start_ns") for r in rows if r.get("measure_start_ns")}
    per = defaultdict(Counter)
    for line in open(trace_path):
        if not line.strip():
            continue
        e = json.loads(line)
        gr = e.get("grank", e.get("rank"))
        ns = start.get(gr)
        if ns is not None and e.get("ts_ns", 0) < ns:
            continue
        per[gr][e.get("op")] += 1
    return per

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()
    per = counts_per_rank(args.trace, args.summary)
    if not per:
        print("INVALID: no ranks found"); sys.exit(1)
    # compare counts across ranks
    ops = set()
    for c in per.values():
        ops.update(c.keys())
    ranks = sorted(per)
    ok = True
    for op in sorted(ops):
        vals = {r: per[r].get(op, 0) for r in ranks}
        if len(set(vals.values())) != 1:
            ok = False
            print(f"MISMATCH op={op}: {vals}")
    if not ok:
        print("INVALID: unequal collective counts across ranks")
        sys.exit(1)
    print(f"VALID: {len(ranks)} ranks, equal counts per op")
    sys.exit(0)

if __name__ == "__main__":
    main()
