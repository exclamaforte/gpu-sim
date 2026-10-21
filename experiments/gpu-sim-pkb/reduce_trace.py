#!/usr/bin/env python3
"""Reduce a committed fabric trace into the byte table.

Usage: reduce_trace.py <trace_path> [--json]
Byte factor = (sum of isend bytes per rank) / S. No phase split (digest is a
content hash, not a phase marker).
Completeness assertion (INVALID if any fails, exit 2):
  - metadata header present (S_BYTES, NRANKS, RUN_ID)
  - every isend line well-formed (rank, bytes, digest); malformed lines counted
  - per rank: total bytes is an exact multiple of S, with 1 <= total/S <= 4
"""
import argparse, json, re, sys
from pathlib import Path

def parse_trace(path):
    text = Path(path).read_text()
    S = nranks = run_id = None
    malformed = 0
    isends = []
    for line in text.splitlines():
        m = re.match(r"^#\s*S_BYTES=(\d+)\s+NRANKS=(\d+)\s+RUN_ID=(.+)$", line)
        if m:
            S = int(m.group(1)); nranks = int(m.group(2)); run_id = m.group(3).strip()
            continue
        if "isend" not in line:
            continue
        rm = re.search(r"rank=(\d+)", line)
        bm = re.search(r"bytes=(\d+)", line)
        dm = re.search(r"digest=([0-9a-fA-F]+)", line)
        if not (rm and bm and dm):
            malformed += 1
            continue
        isends.append({"rank": int(rm.group(1)), "bytes": int(bm.group(1)),
                       "digest": dm.group(1)})
    if S is None:
        raise ValueError(f"trace {path} missing metadata header (S_BYTES/NRANKS/RUN_ID)")
    return S, nranks, run_id, isends, malformed

def reduce(path):
    S, nranks, run_id, isends, malformed = parse_trace(path)
    if malformed > 0:
        raise AssertionError(f"INVALID: {malformed} malformed isend line(s)")
    if not isends:
        raise AssertionError("INVALID: no isend lines")
    by_rank = {}
    for isd in isends:
        by_rank.setdefault(isd["rank"], []).append(isd)
    per_rank = {}
    for rank, lst in by_rank.items():
        total_bytes = sum(isd["bytes"] for isd in lst)
        if total_bytes % S != 0:
            raise AssertionError(
                f"INVALID: rank {rank}: total_bytes {total_bytes} is not an exact "
                f"multiple of S {S} (factor {total_bytes/S:.4f})")
        factor = total_bytes // S
        if factor < 1 or factor > 4:
            raise AssertionError(
                f"INVALID: rank {rank}: byte factor {factor} outside [1,4]")
        per_rank[rank] = {
            "isends": len(lst),
            "total_bytes": total_bytes,
            "byte_factor": total_bytes / S,
            "distinct_digests": len({isd["digest"] for isd in lst}),
        }
    return {
        "trace": str(path),
        "S_bytes": S,
        "nranks": nranks,
        "run_id": run_id,
        "malformed_lines": malformed,
        "ranks": sorted(per_rank.keys()),
        "per_rank": per_rank,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        res = reduce(args.trace)
    except (AssertionError, ValueError, OSError) as e:
        print(f"INVALID: {e}")
        sys.exit(2)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        S = res["S_bytes"]
        for rank in res["ranks"]:
            pr = res["per_rank"][rank]
            print(f"S={S} rank={rank}: isends={pr['isends']} "
                  f"total_bytes={pr['total_bytes']} byte_factor={pr['byte_factor']:.2f} "
                  f"distinct_digests={pr['distinct_digests']}")

if __name__ == "__main__":
    main()
