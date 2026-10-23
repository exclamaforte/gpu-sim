#!/usr/bin/env python3
import json, math, pathlib, argparse
HERE = pathlib.Path(__file__).resolve().parent
PRED = HERE / "predictions_9ul.jsonl"
METRICS = HERE / "metrics.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "metrics.json"))
    args = ap.parse_args()
    preds = [json.loads(l) for l in PRED.read_text().splitlines()]
    # recompute per-family median rel error from predictions
    from collections import defaultdict
    byf = defaultdict(list)
    for p in preds:
        byf[p["family"]].append(p)
    test_cells = {}
    for fam, ps in byf.items():
        mr = __import__("numpy").median([abs(__import__("math").exp(p["y_pred"])-__import__("math").exp(p["y_true"]))/__import__("math").exp(p["y_true"]) for p in ps])
        test_cells[fam] = {"median_rel_error": float(mr), "n_test": len(ps)}
    # load existing for selection_log etc
    m = json.loads(METRICS.read_text())
    m["test_cells_recomputed"] = test_cells
    pathlib.Path(args.out).write_text(json.dumps(m, indent=2)+"\n")
    print("recomputed test cells from", len(preds), "predictions")

if __name__=="__main__":
    main()
PY
echo "skip"