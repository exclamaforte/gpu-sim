#!/usr/bin/env python3
"""Reduce a committed fabric trace into the byte table.

Usage: reduce_trace.py <trace_path> [--json]
Phase boundary rule (asserted in code): a new phase begins when an isend's
digest differs from the previous isend's digest for that rank.
Completeness assertion: per rank, total_bytes in [S, 4*S] and distinct digests == 2.
"""
import argparse, json, re, sys
from pathlib import Path

def parse_trace(path):
    text = Path(path).read_text()
    S = nranks = run_id = None
    for line in text.splitlines():
        m = re.match(r"^#\s*S_BYTES=(\d+)\s+NRANKS=(\d+)\s+RUN_ID=(.+)$", line)
        if m:
            S = int(m.group(1)); nranks = int(m.group(2)); run_id = m.group(3).strip()
            break
    if S is None:
        raise ValueError(f"trace {path} missing metadata header (S_BYTES/NRANKS/RUN_ID)")
    # parse isends
    isends = []  # list of dict: rank, bytes, digest
    for line in text.splitlines():
        if "isend" not in line:
            continue
        rm = re.search(r"rank=(\d+)", line)
        bm = re.search(r"bytes=(\d+)", line)
        dm = re.search(r"digest=([0-9a-fA-F]+)", line)
        if not (rm and bm and dm):
            continue
        isends.append({"rank": int(rm.group(1)), "bytes": int(bm.group(1)),
                       "digest": dm.group(1)})
    return S, nranks, run_id, isends

def reduce(path):
    S, nranks, run_id, isends = parse_trace(path)
    by_rank = {}
    for isd in isends:
        by_rank.setdefault(isd["rank"], []).append(isd)
    per_rank = {}
    for rank, lst in by_rank.items():
        # phase boundary = digest change
        phases = []  # list of list of isends
        cur = []
        prev_digest = None
        for isd in lst:
            if prev_digest is not None and isd["digest"] != prev_digest:
                phases.append(cur)
                cur = []
            cur.append(isd)
            prev_digest = isd["digest"]
        if cur:
            phases.append(cur)
        distinct_digests = len({isd["digest"] for isd in lst})
        total_bytes = sum(isd["bytes"] for isd in lst)
        # completeness assertion
        assert total_bytes >= S, f"rank {rank}: total_bytes {total_bytes} < S {S} (INVALID)"
        assert total_bytes <= 4 * S, f"rank {rank}: total_bytes {total_bytes} > 4S {4*S} (INVALID)"
        assert distinct_digests == 2, f"rank {rank}: distinct digests {distinct_digests} != 2 (INVALID)"
        per_phase = [{"phase": i, "pieces": len(ph), "bytes": sum(x["bytes"] for x in ph),
                      "digest": ph[0]["digest"]} for i, ph in enumerate(phases)]
        per_rank[rank] = {
            "isends": len(lst),
            "distinct_digests": distinct_digests,
            "phases": len(phases),
            "total_bytes": total_bytes,
            "byte_factor": total_bytes / S,
            "per_phase": per_phase,
        }
    return {
        "trace": str(path),
        "S_bytes": S,
        "nranks": nranks,
        "run_id": run_id,
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
    except AssertionError as e:
        print(f"INVALID: {e}")
        sys.exit(2)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        S = res["S_bytes"]
        for rank in res["ranks"]:
            pr = res["per_rank"][rank]
            print(f"S={S} rank={rank}: isends={pr['isends']} phases={pr['phases']} "
                  f"total_bytes={pr['total_bytes']} byte_factor={pr['byte_factor']:.2f} "
                  f"distinct_digests={pr['distinct_digests']}")
            for ph in pr["per_phase"]:
                print(f"   phase {ph['phase']}: pieces={ph['pieces']} bytes={ph['bytes']}")

if __name__ == "__main__":
    main()
