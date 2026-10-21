#!/usr/bin/env python3
"""Build the re-measured byte table from committed traces (m1 runs)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from reduce_trace import reduce

# one representative run per size (m1); m2 is the regenerability check
canonical = {
    4096:   "traces/trace_S4096_n2_m1.log",
    65536:  "traces/trace_S65536_n2_m1.log",
    131072: "traces/trace_S131072_n2_m1.log",
    262144: "traces/trace_S262144_n2_m1.log",
    393216: "traces/trace_S393216_n2_m1.log",
    524288: "traces/trace_S524288_n2_m1.log",
    1048576:"traces/trace_S1048576_n2_m1.log",
    2097152:"traces/trace_S2097152_n2_m1.log",
}

rows = []
for S in sorted(canonical):
    path = canonical[S]
    try:
        res = reduce(path)
        valid = True
        note = "valid"
    except (AssertionError, ValueError) as e:
        res = None
        valid = False
        note = str(e)
        # parse raw for reporting
        import re
        text = open(path).read()
        S2, nranks, run_id, isends, mal = __import__("reduce_trace").parse_trace(path)
        by_rank = {}
        for isd in isends:
            by_rank.setdefault(isd["rank"], []).append(isd)
        rank_info = []
        for rank, lst in sorted(by_rank.items()):
            total = sum(x["bytes"] for x in lst)
            rank_info.append({"rank": rank, "isends": len(lst),
                              "total_bytes": total, "byte_factor": total / S,
                              "distinct_digests": len({x["digest"] for x in lst}),
                              "valid": (total % S == 0 and 1 <= total // S <= 4)})
        rows.append({"S_bytes": S, "trace": path, "valid": False,
                     "validity_note": note, "ranks": rank_info})
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
                 "validity_note": "valid", "ranks": rank_info})

json.dump({"bead": "gpu-sim-eut", "canonical_runs": "m1", "regenerability_runs": "m2",
           "rows": rows}, open("table_results.json", "w"), indent=2)
print("rebuilt table_results.json")
