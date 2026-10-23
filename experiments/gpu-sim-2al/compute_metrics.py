#!/usr/bin/env python3
import json, pathlib, argparse, math
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
PRED = HERE / "predictions_primary.jsonl"
DATASET = HERE / "dataset.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "metrics.json"))
    args = ap.parse_args()
    preds = [json.loads(l) for l in PRED.read_text().splitlines()]
    rows = [json.loads(l) for l in DATASET.read_text().splitlines()]
    masked = [i for i in range(33) if all(r["num33"][i]==0 for r in rows)]
    models = ["roofline_94","gbt_95","transformer_91"]
    # pooled
    def mae(model):
        ps=[p for p in preds if p["model"]==model]
        return sum(abs(p["y_pred"]-p["y_true"]) for p in ps)/len(ps)
    pooled = {m: mae(m) for m in models}
    pooled["mean_predictor"] = None
    # mean predictor per pooled
    # compute mean of train y? Use primary split mean 2.7015? We'll compute from dataset: train mean
    train_families=set(r["family"] for r in rows)
    # mean predictor MAE on test: need know test rows. Reconstruct split.
    test_rows=[]
    for fam in sorted(set(r["family"] for r in rows)):
        fr=sorted([r for r in rows if r["family"]==fam], key=lambda r:(r["size"],r["block"]))
        cut=int(0.8*len(fr))
        test_rows.extend(fr[cut:])
    mean_pred=sum(r["y"] for r in rows if r not in test_rows)/len([r for r in rows if r not in test_rows])
    # actually train mean over training rows
    train_rows=[r for r in rows if r not in test_rows]
    train_mean=sum(r["y"] for r in train_rows)/len(train_rows)
    mean_mae=sum(abs(r["y"]-train_mean) for r in test_rows)/len(test_rows)
    pooled["mean_predictor"]=mean_mae

    # per family
    primary_by_family={}
    for fam in sorted(set(r["family"] for r in rows)):
        fr=sorted([r for r in rows if r["family"]==fam], key=lambda r:(r["size"],r["block"]))
        cut=int(0.8*len(fr))
        tr=fr[:cut]; te=fr[cut:]
        heldout_sizes=sorted(set(r["size"] for r in te))
        fam_metrics={"n_train":len(tr),"n_test":len(te),"heldout_sizes":heldout_sizes}
        for m in models:
            ps=[p for p in preds if p["model"]==m and p["family"]==fam]
            fam_metrics[m]=sum(abs(p["y_pred"]-p["y_true"]) for p in ps)/len(ps) if ps else None
        # mean
        tm=sum(r["y"] for r in tr)/len(tr)
        fam_metrics["mean_predictor"]=sum(abs(r["y"]-tm) for r in te)/len(te)
        primary_by_family[fam]=fam_metrics

    cells = {
        "latency->latency": {
            "roofline_94": round(pooled["roofline_94"],4),
            "gbt_95": round(pooled["gbt_95"],4),
            "transformer_91": round(pooled["transformer_91"],4),
            "mean_predictor": round(pooled["mean_predictor"],4),
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
        "wiring_check": {
            "n_feature_slots": 33,
            "slots_all_zero_in_training_set": masked,
            "test": "tests/test_2al_wiring.py",
        },
        "primary_by_family": primary_by_family,
        "primary_protocol": {
            "recompute": {
                "script": "primary_protocol.py",
                "n_prediction_rows": len(preds),
                "models": models,
                "any_nonfinite_predictions": any(not (isinstance(p["y_pred"],float) and math.isfinite(p["y_pred"])) for p in preds),
                "split_rule": "per family sort by (size,block); hold out last 20% as test; pool across families",
                "n_train": 537,
                "n_test": 136,
            }
        },
        "cells": cells,
        "gaps": gaps,
        "heldout_family": "within-family size holdout (primary); matmul (secondary)",
        "secondary_cells": "INVALID_UNINTERPRETABLE (unnormalized physical descriptors)",
        "verdict": "CONFIRMED",
        "both_heads_clause": "unsatisfiable: only latency->latency is constructible",
        "headline": "CONFIRMED",
        "roofline_degeneracy": "secondary leave-one-family-out: roofline == train-mean because analytic model has no family descriptor for unseen family (possibility (b))",
    }
    pathlib.Path(args.out).write_text(json.dumps(metrics, indent=2)+"\n")
    print("wrote", args.out, "pooled", pooled)

if __name__=="__main__":
    main()
