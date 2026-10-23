#!/usr/bin/env python3
import json, math, pathlib, argparse

HERE = pathlib.Path(__file__).resolve().parent
BK9 = HERE.parent.parent / "experiments" / "gpu-sim-bk9" / "metrics.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "metrics.json"))
    args = ap.parse_args()
    bk = json.loads(BK9.read_text())
    cells = bk["cells"]
    usable = [c for c in cells if c["target_usable"]]
    families = sorted(set(c["family"] for c in usable))
    sizes = sorted(set(c["size"] for c in usable))
    targets = [c["kernel_us"] for c in usable]
    log_targets = [math.log(t) for t in targets if t and t > 0]
    target_variance = (sum((x - sum(log_targets)/len(log_targets))**2 for x in log_targets) / len(log_targets)) if log_targets else 0.0

    # decision
    if len(usable) < 500 or len(families) < 3 or target_variance == 0:
        verdict = "UNDERPOWERED"
    else:
        verdict = "CONFIRMED"  # placeholder; real training not run in this bead
    # Actually no training performed; force UNDERPOWERED given 8 rows
    verdict = "UNDERPOWERED"

    n_rows_total = len(cells)
    n_excluded = n_rows_total - len(usable)

    data_requirement = {
        "usable_rows_now": len(usable),
        "usable_rows_needed": 500,
        "additional_usable_rows_needed": max(0, 500 - len(usable)),
        "families_now": len(families),
        "families_needed": 3,
        "estimated_gpu_hours": round((500 - len(usable)) * 2.0 / 60.0, 2),
        "note": "Each new (family,size) cell costs ~2 min of GPU (batched+single sweep). 8 of 25 current cells usable; need ~492 more usable cells.",
    }

    metrics = {
        "bead": "gpu-sim-tsy",
        "dataset": {
            "n_rows": n_rows_total,
            "n_usable_rows": len(usable),
            "n_excluded": n_excluded,
            "families": families,
            "sizes": sizes,
            "target_field": "kernel_us",
            "target_transform": "ln(kernel_us) (natural log), matching 10.4 log-space MAE",
            "target_variance": target_variance,
        },
        "cells": {},
        "gaps": [
            {"head": "latency->latency", "reason": "underpowered: no training run (8 usable rows)"},
            {"head": "latency->memory", "reason": "underpowered: no training run (8 usable rows)"},
            {"head": "memory->memory", "reason": "underpowered: no training run (8 usable rows)"},
        ],
        "verdict": verdict,
        "verdict_detail": {
            "underpowered_reasons": [
                f"usable rows {len(usable)} < 500",
            ],
            "data_requirement": data_requirement,
        },
        "headline": verdict,
    }
    pathlib.Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {args.out}: verdict={verdict}, usable={len(usable)}")

if __name__ == "__main__":
    main()
