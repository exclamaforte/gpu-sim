#!/usr/bin/env python3
"""gpu-sim-u2o.11.4 headline (CPU): derive the judged answer from 10.4's
verdict table. All ratios COMPUTED from the table (Sep-28 lesson: never
type a ratio); the test pins headline == source so prose cannot drift.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "experiments/gpu-sim-u2o.10.4/metrics.json"


def main(out=None) -> int:
    out = Path(out) if out else (REPO / "experiments/gpu-sim-u2o.11.4"
                                 / "metrics.json")
    t = json.loads(SRC.read_text())
    c = t["cells"]
    ll, lm = c["latency->latency"], c["latency->memory"]

    def ratio(a, b):
        return round(a / b, 2) if a and b else None

    headline = {
        "bead": "gpu-sim-u2o.11.4",
        "question": "does the learned model beat both baselines, by how "
                    "much, on each axis",
        "answer": "NO: transformer beats roofline, loses to GBT on both "
                  "head-to-head triples",
        "derived_from": "experiments/gpu-sim-u2o.10.4/metrics.json cells",
        "latency->latency": {
            "cells": ll,
            "roofline_over_transformer": ratio(ll["roofline_94"],
                                               ll["transformer_91"]),
            "transformer_over_gbt": ratio(ll["transformer_91"], ll["gbt_95"]),
            "reading": "transformer beats roofline ~4x, loses to GBT ~1.3x",
        },
        "latency->memory": {
            "cells": lm,
            "roofline_over_transformer": ratio(lm["roofline_94"],
                                               lm["transformer_91"]),
            "transformer_over_gbt": ratio(lm["transformer_91"], lm["gbt_95"]),
            "reading": "transformer beats roofline ~2.6x, loses to GBT ~9.5x",
        },
        "axes_without_headline": t["gaps"],
        "context_pending_review": "11.2 (branch lab/gpu-sim-u2o.11.2): "
            "transformer errors concentrate in measurable latency->memory; "
            "cited, not depended on -- files unmerged at write time",
    }
    out.write_text(json.dumps(headline, indent=2))
    print(headline["answer"])
    for k in ("latency->latency", "latency->memory"):
        print(f"  {k}: {headline[k]['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
