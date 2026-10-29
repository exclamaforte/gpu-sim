#!/usr/bin/env python3
"""Regenerate experiments/gpu-sim-b1r6/metrics.json precision band from model_facts.

Produced, not typed: derived from docs/model_facts.json -> fabric_precision,
which is itself derived from error_budget.ranked.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_docs_model_numbers as c

FACTS = ROOT / "docs" / "model_facts.json"
METRICS = ROOT / "experiments" / "gpu-sim-b1r6" / "metrics.json"

def main():
    facts = json.loads(FACTS.read_text())
    fp = facts.get("fabric_precision") or c._fabric_precision(facts)
    # build a human-readable band per cell
    band = {}
    for cell, d in fp.items():
        fd = d["term_scale_cp_flip_down_ratio"]
        fu = d["term_scale_cp_flip_up_ratio"]
        sides = d["term_scale_cp_measured_sides"]
        if fd is not None and fu is not None:
            desc = (f"two-sided: fabric may range {fd:.3f}x to {fu:.3f}x of fitted "
                    f"before {cell} verdict changes")
        elif fu is not None:
            desc = (f"one-sided (up): fabric must stay below {fu:.3f}x of fitted; "
                    f"no lower flip (down to 0.25x)")
        elif fd is not None:
            desc = (f"one-sided (down): fabric must stay above {fd:.3f}x of fitted")
        else:
            desc = "no flip within range [0.25, 4.0]: fabric not load-bearing for this cell"
        band[cell] = {
            "flip_down_ratio": fd,
            "flip_up_ratio": fu,
            "measured_sides": sides,
            "fraction_one_sided": d["term_scale_cp_fraction_one_sided"],
            "next_rank1_if_pinned": d["next_rank1_if_pinned"],
            "next_rank1_fraction_one_sided": d["next_rank1_fraction_one_sided"],
            "band_description": desc,
        }
    metrics = {
        "cells": list(fp.keys()),
        "fabric_precision_band": band,
        "fitted_term_scale_cp": 1.0,
        "range_tested": [0.25, 4.0],
        "method": "error_budget elasticity; term_scale_cp sweep over [0.25,4.0] per cell; flip ratios to fitted 1.0",
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print("wrote", METRICS)

if __name__ == "__main__":
    main()
