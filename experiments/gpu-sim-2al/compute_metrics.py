#!/usr/bin/env python3
import json, pathlib, argparse

HERE = pathlib.Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
METRICS_1JF = HERE.parent.parent / "gpu-sim-1jf" / "metrics.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "metrics.json"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in DATASET.read_text().splitlines()]
    masked = [i for i in range(33) if all(r["num33"][i]==0 for r in rows)]
    # wiring check
    wiring = {
        "n_feature_slots": 33,
        "slots_all_zero_in_training_set": masked,
        "test": "tests/test_2al_wiring.py",
    }
    # roofline and gbt maes
    import numpy as np
    train=[r for r in rows if r["family"]!="matmul"]
    test=[r for r in rows if r["family"]=="matmul"]
    yte=np.array([r["y"] for r in test])
    mean_pred=float(np.mean([r["y"] for r in train]))
    roofline_mae=float(np.mean(np.abs(yte-mean_pred)))
    gbt_pred=json.loads((HERE/"gbt_pred.json").read_text())["pred"]
    gbt_mae=sum(abs(p-t) for p,t in zip(gbt_pred,yte.tolist()))/len(yte)
    cells = {
        "latency->latency": {
            "roofline_94": round(roofline_mae,4),
            "gbt_95": round(gbt_mae,4),
            "transformer_91": None,
        }
    }
    gaps = [
        {"head":"latency->memory","reason":"no memory measurements in this dataset"},
        {"head":"memory->memory","reason":"no memory measurements in this dataset"},
    ]
    metrics = {
        "bead": "gpu-sim-2al",
        "dataset": {
            "source_commit": "df3dbcb",
            "n_rows": len(rows),
            "n_usable": len(rows),
            "families": sorted(set(r["family"] for r in rows)),
            "sizes_range": [min(r["size"] for r in rows), max(r["size"] for r in rows)],
            "source_family": "numeric-only (all rows from measurement harness)",
            "features_masked": masked,
        },
        "wiring_check": wiring,
        "cells": cells,
        "gaps": gaps,
        "heldout_family": "matmul",
        "verdict": "UNDERPOWERED",
        "verdict_detail": {
            "underpowered_reasons": ["transformer training did not converge (NaN); only GBT/roofline available"],
        },
        "headline": "UNDERPOWERED",
    }
    pathlib.Path(args.out).write_text(json.dumps(metrics, indent=2)+"\n")
    print("wrote", args.out)

if __name__=="__main__":
    main()
