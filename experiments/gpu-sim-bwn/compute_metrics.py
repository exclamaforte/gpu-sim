#!/usr/bin/env python3
import json, pathlib, argparse

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "bwn_results.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "metrics.json"))
    args = ap.parse_args()
    r = json.loads(RES.read_text())
    g = r["gate_statistic_by_family"]
    metrics = {
        "bead": "gpu-sim-bwn",
        "gate_statistic_by_family": g,
        "baseline_from_nk0": r["baseline_from_nk0"],
        "pooled_median_rel_error": r["pooled_median_rel_error"],
        "verdict": r["verdict"],
        "informative_slots": r["informative_slots"],
        "bar": 0.25,
        "bar_criterion": "u2o.17: median |sim-measured|/measured <= 25% per op class",
        "gate_classes": ["elementwise","matmul","reduction"],
        "gate_pass": {k: r["gate_statistic_by_family"][k]["gbt_median_rel_error"]<=0.25 for k in ["elementwise","matmul","reduction"]},
        "feature_spec": r["feature_spec"],
        "size_only_control": {
            "status": "WITHDRAWN",
            "feature_list": ["log1p(size)"],
            "block_included": False,
            "standardization": "z-score (train mean/std, sd==0 -> 1)",
            "target_transform": "ln(median_runtime_ms); median relative error on exp(y)",
            "split": "per-family top 20% by size held out",
            "gbt_config": "HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, min_samples_leaf=5, random_state=0)",
            "reason_withdrawn": "PI reconciliation: bwn's stated definition reproduces 9ul's numbers, not bwn's committed ones",
        },
        "headline": r["verdict"],
    }
    pathlib.Path(args.out).write_text(json.dumps(metrics, indent=2)+"\n")
    print("wrote", args.out)

if __name__=="__main__":
    main()
