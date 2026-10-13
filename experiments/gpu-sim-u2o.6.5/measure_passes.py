#!/usr/bin/env python3
"""gpu-sim-u2o.6.5 GPU half -- RUN ONLY INSIDE A ts JOB. Measures the 8
preflight-passing specs from preflight_results.json via
SafeHarness.measure_spec (preflight re-checked inside measure_spec, then
compile, then real-GPU run with 30 s timeout). Normal kernels only;
rejects are NEVER submitted (they have no spec row here by construction).

Writes measured.json: per-cid {trials_ms, n_trials, median_ms, success,
timeout, wall_time_sec}. Scoring/statistics happen on CPU outside the queue.
"""
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness  # noqa: E402


def main():
    pre = json.loads((HERE / "preflight_results.json").read_text())
    passes = [p for p in pre["passes"] if p["ok"]]
    assert len(passes) == 8, f"expected 8 preflighted passes, got {len(passes)}"
    h = SafeHarness(timeout_sec=30)
    rows = []
    for p in passes:
        t0 = time.time()
        try:
            trials, res = h.measure_spec(p["spec"])
            row = {"cid": p["cid"], "op": p["op"],
                   "problem_size": p["problem_size"],
                   "measured": True, "success": res.success,
                   "timeout": res.timeout, "n_trials": len(trials),
                   "trials_ms": trials,
                   "median_ms": (round(statistics.median(trials), 6)
                                 if trials else None),
                   "wall_time_sec": round(res.wall_time_sec, 3),
                   "job_sec": round(time.time() - t0, 1)}
        except RuntimeError as e:
            row = {"cid": p["cid"], "measured": False, "error": str(e)[:300]}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items()
                          if k != "trials_ms"}), flush=True)
    (HERE / "measured.json").write_text(json.dumps(rows, indent=2) + "\n")
    ok = sum(1 for r in rows if r.get("success") and r.get("n_trials"))
    print(f"measured_success={ok}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
