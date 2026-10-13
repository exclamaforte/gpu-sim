#!/usr/bin/env python3
"""gpu-sim-u2o.11.5 generator (CPU): emits the results-table block from
committed metrics files. The test pins every emitted line present verbatim
in results_writeup.md -- prose cannot drift from artifacts (6.11 direction).
Only MERGED artifacts are sourced; pending-review work is cited by branch,
never numerated."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def table() -> list[str]:
    m4 = json.loads((REPO / "experiments/gpu-sim-u2o.10.4"
                     / "metrics.json").read_text())
    m1 = json.loads((REPO / "experiments/gpu-sim-u2o.10.1"
                     / "metrics.json").read_text())
    m2 = json.loads((REPO / "experiments/gpu-sim-u2o.10.2"
                     / "metrics.json").read_text())
    c = m4["cells"]
    lines = [
        "| triple | roofline 9.4 | GBT 9.5 | transformer 9.1 |",
        f"| latency->latency | {c['latency->latency']['roofline_94']} "
        f"| {c['latency->latency']['gbt_95']} "
        f"| {c['latency->latency']['transformer_91']} |",
        f"| latency->memory | {c['latency->memory']['roofline_94']} "
        f"| {c['latency->memory']['gbt_95']} "
        f"| {c['latency->memory']['transformer_91']} |",
        f"Constant train-mean baseline (10.1, n=57 test): log-RMSE "
        f"{m1['log_space']['rmse']}, R2 {m1['log_space']['r2']}, raw-ms "
        f"MAPE {m1['raw_ms']['mape']}%, coverage_90 "
        f"{m1['log_space']['coverage_90']}.",
        f"Feature-17 single-feature ranker (10.2, n=57 test): spearman "
        f"{m2['feat17_ranker']['spearman']}, kendall "
        f"{m2['feat17_ranker']['kendall_tau']}.",
    ]
    return lines


if __name__ == "__main__":
    for line in table():
        print(line)
