#!/usr/bin/env python3
"""gpu-sim-u2o.17.4 parity gate (CPU): real 17.5 fixture (+17.3 knee
supplement) vs scaling-law simulated trace. Criterion PRE-SET (PI
amendment): median |sim-measured|/measured <= 25% per op class AND
rank-order preserved (same class dominates by total time in both).
A FAIL is a recordable negative (11.6 protocol), not a hidden one.
"""
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/gpu-sim-u2o.17.3"))

import sim_trace_173 as S  # noqa: E402

EXP = REPO / "experiments/gpu-sim-u2o.17.4"
BOUND = 0.25


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    tmp = EXP / "_sim_tmp.json"
    knee = REPO / "experiments/gpu-sim-u2o.17.3/knee_cases.json"
    assert S.main(str(tmp), str(knee)) == 0
    sim = json.loads(tmp.read_text())
    tmp.unlink()
    per_class, rank = {}, {}
    for cls, r in sim["classes"].items():
        errs = sorted(abs(d["fit_rel_err"]) for d in r["held_out"])
        med = statistics.median(errs)
        tot_meas = sum(d["measured_ms"] for d in r["held_out"])
        tot_pred = sum(d["fit_pred_ms"] for d in r["held_out"])
        per_class[cls] = {"median_abs_rel_err": med,
                          "n_sizes": r["n_sizes"],
                          "passes_25pct": med <= BOUND}
        rank[cls] = (tot_meas, tot_pred)
    dom_meas = max(rank, key=lambda c: rank[c][0])
    dom_pred = max(rank, key=lambda c: rank[c][1])
    rank_ok = dom_meas == dom_pred
    verdict = ("PASS" if all(v["passes_25pct"] for v in per_class.values())
               and rank_ok else "FAIL")
    metrics = {"bead": "gpu-sim-u2o.17.4",
               "criterion": "median |sim-measured|/measured <= 25% per op "
                            "class + same dominant class by total time",
               "criterion_set": "PI amendment pre-run (no knee data existed)",
               "per_class": {k: {"median_abs_rel_err": round(v["median_abs_rel_err"], 4),
                                 "n_sizes": v["n_sizes"],
                                 "passes_25pct": v["passes_25pct"]}
                              for k, v in per_class.items()},
               "rank_order": {"dominant_measured": dom_meas,
                              "dominant_predicted": dom_pred,
                              "preserved": rank_ok},
               "verdict": verdict}
    (EXP / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for cls, v in metrics["per_class"].items():
        flag = "pass" if v["passes_25pct"] else "FAIL"
        print(f"  {cls}: median err {v['median_abs_rel_err']:.3f} [{flag}]")
    print(f"rank preserved: {rank_ok} ({dom_meas}); VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
