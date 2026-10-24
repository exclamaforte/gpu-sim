"""Validate that ranks within the same PP stage have equal collective counts per op.

Usage:
    python -m tools.torch_parallel.validate_trace --trace T --summary S
Exits 0 if all ranks within each pp stage have equal counts per op,
non-zero (INVALID) otherwise. PP stages differ in role (send vs recv), so
comparison is per-stage, not global.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict, Counter

def counts_per_rank(trace_path, summary_path):
    rows = [json.loads(l) for l in open(summary_path) if l.strip()]
    start = {r["rank"]: r.get("measure_start_ns") for r in rows if r.get("measure_start_ns")}
    pp_of = {r["rank"]: r.get("pp_stage", r.get("pp_idx", 0)) for r in rows}
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
    return per, pp_of

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()
    per, pp_of = counts_per_rank(args.trace, args.summary)
    if not per:
        print("INVALID: no ranks found"); sys.exit(1)
    # group ranks by pp_stage
    stages = defaultdict(list)
    for r in per:
        stages[pp_of.get(r, 0)].append(r)
    ok = True
    for stage, ranks in stages.items():
        ops = set()
        for r in ranks:
            ops.update(per[r].keys())
        for op in sorted(ops):
            vals = {r: per[r].get(op, 0) for r in ranks}
            if len(set(vals.values())) != 1:
                ok = False
                print(f"MISMATCH stage={stage} op={op}: {vals}")
    if not ok:
        print("INVALID: unequal collective counts within a PP stage")
        sys.exit(1)
    print(f"VALID: {len(per)} ranks across {len(stages)} stages, equal counts per op within stage")
    sys.exit(0)

if __name__ == "__main__":
    main()
