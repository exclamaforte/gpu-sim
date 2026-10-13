#!/usr/bin/env python3
"""gpu-sim-u2o.10.4 JOIN (CPU): three-way table from COMMITTED artifacts.

Reads 9.4/9.5/9.1 metrics.json (log-MAE per triple on the family_holdout
test split), asserts the values behind the committed REPORT tables, and
writes the machine-readable verdict. No retraining, no recomputation of
any approach's numbers -- gaps (per-row predictions for ranking /
calibration; 10.3 suites) are recorded, not filled.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP9 = {
    "roofline": REPO / "experiments/gpu-sim-u2o.9.4/metrics.json",
    "gbt": REPO / "experiments/gpu-sim-u2o.9.5/metrics.json",
    "transformer": REPO / "experiments/gpu-sim-u2o.9.1/metrics.json",
}
# Committed REPORT-table values (9.1 REPORT three-way frame, sourced from
# 9.4 REPORT per-regime table + 9.5 holdout): the join asserts these.
REPORTED = {
    ("roofline", "latency->latency"): 0.5638,
    ("roofline", "latency->memory"): 0.6873,
    ("roofline", "memory->memory"): 1.1209,
    ("gbt", "latency->latency"): 0.107,
    ("gbt", "latency->memory"): 0.027,
    ("transformer", "latency->latency"): 0.144,
    ("transformer", "latency->memory"): 0.260,
}


def load():
    m4 = json.loads(EXP9["roofline"].read_text())
    m5 = json.loads(EXP9["gbt"].read_text())
    m1 = json.loads(EXP9["transformer"].read_text())
    return m4, m5, m1


def main(out=None) -> int:
    out = Path(out) if out else (REPO / "experiments" / "gpu-sim-u2o.10.4" / "metrics.json")
    m4, m5, m1 = load()
    t1 = m1["test_report"]["triple"]
    t5 = m5["stage2_holdout"]["test_report"]["triple"]
    # transformer + GBT: exact test log-MAE from committed metrics
    got = {
        ("transformer", "latency->latency"): t1["latency->latency"]["mean_abs_err"],
        ("transformer", "latency->memory"): t1["latency->memory"]["mean_abs_err"],
        ("gbt", "latency->latency"): t5["latency->latency"]["mean_abs_err"],
        ("gbt", "latency->memory"): t5["latency->memory"]["mean_abs_err"],
    }
    for k, rep in REPORTED.items():
        if k in got:
            assert abs(got[k] - rep) < 0.001, f"drift {k}: {got[k]} vs {rep}"
    # roofline: 9.4's artifact carries regime analysis, not a holdout MAE
    # table; its figures (0.5638/0.6873/1.1209) live in 9.4's REPORT table
    # and are quoted with that provenance (gap noted, not recomputed).
    assert "residual_A" in m4 and "per_regime_achieved" in m4
    table = {
        "bead": "gpu-sim-u2o.10.4",
        "split": "family_holdout_v1: test relu 57 / val mul 57 / train 342",
        "metric": "log-space mean abs err (committed aggregates)",
        "cells": {
            triple: {
                "roofline_94": REPORTED.get(("roofline", triple)),
                "gbt_95": round(got[("gbt", triple)], 4)
                if ("gbt", triple) in got else None,
                "transformer_91": round(got[("transformer", triple)], 4)
                if ("transformer", triple) in got else None,
            } for triple in ("latency->latency", "latency->memory",
                             "memory->memory")
        },
        "provenance": {
            "roofline_94": "experiments/gpu-sim-u2o.9.4/metrics.json "
                           "+ REPORT per-regime table (no holdout-MAE cell; "
                           "figures quoted, not recomputed)",
            "gbt_95": "experiments/gpu-sim-u2o.9.5/metrics.json "
                      ".stage2_holdout.test_report.triple",
            "transformer_91": "experiments/gpu-sim-u2o.9.1/metrics.json "
                              ".test_report.triple",
        },
        "verdicts": {
            "latency->latency": "GBT 0.107 < transformer 0.144 < roofline 0.564",
            "latency->memory": "GBT 0.027 < transformer 0.260 < roofline 0.687",
        },
        "gaps": [
            "ranking (Spearman/Kendall): no committed per-row predictions "
            "for any approach -- needs saved predictions (11.4 or follow-up)",
            "calibration: same -- no committed (mean, var) rows",
            "generalization suites: 10.3 frozen, no suite artifacts exist",
            "memory->memory test: absent from the holdout template for "
            "learned approaches (roofline 1.1209 from its own analysis, "
            "different basis -- not a head-to-head cell)",
        ],
    }
    out.write_text(json.dumps(table, indent=2))
    for t, v in table["verdicts"].items():
        print(f"  {t}: {v}")
    print("gaps recorded:", len(table["gaps"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
