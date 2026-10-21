#!/usr/bin/env python3
"""Build the re-measured byte table from committed traces."""
import json, re, sys, glob, os
sys.path.insert(0, os.path.dirname(__file__))
from reduce_trace import parse_trace, reduce

# canonical trace per size (chosen for validity: both ranks dd==2 where possible)
canonical = {
    4096:   "traces/trace_S4096_n2_eut4-4096-1792598352.log",   # INVALID (rank1 dd=1)
    65536:  "traces/trace_S65536_n2_CLEAN.log",
    262144: "traces/trace_S262144_n2_CLEAN.log",
    524288: "traces/trace_S524288_n2_CLEAN.log",
    1048576:"traces/trace_S1048576_n2_CLEAN.log",
    2097152:"traces/trace_S2097152_n2_CLEAN.log",
}

rows = []
for S in sorted(canonical):
    path = canonical[S]
    try:
        res = reduce(path)
        valid = True
        validity_note = "valid"
    except AssertionError as e:
        res = None
        valid = False
        validity_note = str(e)
        # still parse for reporting
        S2, nranks, run_id, isends = parse_trace(path)
        by_rank = {}
        for isd in isends:
            by_rank.setdefault(isd["rank"], []).append(isd)
        rank_info = []
        for rank, lst in sorted(by_rank.items()):
            total = sum(x["bytes"] for x in lst)
            dd = len({x["digest"] for x in lst})
            rank_info.append({"rank": rank, "isends": len(lst), "total_bytes": total,
                              "byte_factor": total / S, "distinct_digests": dd,
                              "valid": (S <= total <= 4*S and dd == 2)})
        rows.append({"S_bytes": S, "trace": path, "valid": False,
                     "validity_note": validity_note, "ranks": rank_info})
        continue
    rank_info = []
    for rank in res["ranks"]:
        pr = res["per_rank"][rank]
        rank_info.append({"rank": rank, "isends": pr["isends"],
                          "total_bytes": pr["total_bytes"],
                          "byte_factor": pr["byte_factor"],
                          "distinct_digests": pr["distinct_digests"],
                          "valid": True})
    rows.append({"S_bytes": S, "trace": path, "valid": True,
                 "validity_note": "valid (both ranks dd==2)", "ranks": rank_info})

json.dump({"bead": "gpu-sim-eut", "canonical_traces": canonical, "rows": rows},
          open("table_results.json", "w"), indent=2)

# print
print(f"{'S':>8} {'rank0 factor':>13} {'rank1 factor':>13} {'valid':>6}")
for r in rows:
    f0 = r["ranks"][0]["byte_factor"] if len(r["ranks"])>0 else float('nan')
    f1 = r["ranks"][1]["byte_factor"] if len(r["ranks"])>1 else float('nan')
    print(f"{r['S_bytes']:>8} {f0:>13.2f} {f1:>13.2f} {str(r['valid']):>6}  {r['trace']}")
