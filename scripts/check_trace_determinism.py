#!/usr/bin/env python3
"""Is the harness DETERMINISTIC? Two identical runs must produce the same record stream (gpu-sim-det).

Every byte-reconciliation claim in this repo is a comparison between a model and a harness trace.
That comparison is only meaningful if the instrument is repeatable: if the same configuration
produced different collectives on a second run, a 2.5% agreement could be luck and a disagreement
could be noise. Nothing here had ever checked that.

What "the same" means with N concurrent processes appending to one file: the cross-rank ORDER of
records interleaves differently run to run (that is scheduling, not behaviour), but each rank's own
SEQUENCE must be identical. This script compares per rank, not line by line, for exactly that reason.

Usage:
  python3 scripts/check_trace_determinism.py run1.jsonl run2.jsonl
Exit 0 if per-rank sequences match, 1 otherwise. Output is a one-line verdict plus, on failure, the
first rank and index that differ.
"""
import collections
import json
import sys

FIELDS = ("op", "count", "bytes", "algorithm", "refused", "bottleneck_link", "regime")


def per_rank(path):
    out = collections.defaultdict(list)
    for line in open(path):
        if not line.strip():
            continue
        rec = json.loads(line)
        if "rank" not in rec:
            continue
        out[rec["rank"]].append(tuple(rec.get(f) for f in FIELDS))
    return {rank: tuple(recs) for rank, recs in out.items()}


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    # Guard: exactly one rank-less record per file, and it must be the header.
    for path in (argv[1], argv[2]):
        headerless = 0
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            if "rank" in rec:
                continue
            headerless += 1
            assert rec.get("type") == "header", f"rank-less record is not a header: {rec}"
        if headerless > 0:
            assert headerless == 1, f"{path}: expected at most one header record, got {headerless}"
    a, b = per_rank(argv[1]), per_rank(argv[2])
    if sorted(a) != sorted(b):
        print(f"DIFFERENT RANKS: {sorted(a)} vs {sorted(b)}")
        return 1
    for rank in sorted(a):
        if a[rank] == b[rank]:
            continue
        idx = next(i for i, (x, y) in enumerate(zip(a[rank], b[rank])) if x != y)
        print(f"NON-DETERMINISTIC at rank {rank}, record {idx}:")
        print(f"  run1: {a[rank][idx]}")
        print(f"  run2: {b[rank][idx]}")
        return 1
    counts = {r: len(v) for r, v in a.items()}
    print(f"DETERMINISTIC: {len(a)} ranks, identical per-rank sequences "
          f"({len(set(counts.values())) == 1 and list(counts.values())[0] or counts} records/rank)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
