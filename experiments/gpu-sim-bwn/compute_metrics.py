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
        "gate_classes": ["elementwise","matmul","reduction"],
        "gate_pass": {k: r["gate_statistic_by_family"][k]["gbt_median_rel_error"]<=0.25 for k in ["elementwise","matmul","reduction"]},
        "feature_spec": r["feature_spec"],
        "headline": r["verdict"],
    }
    pathlib.Path(args.out).write_text(json.dumps(metrics, indent=2)+"\n")
    print("wrote", args.out)

if __name__=="__main__":
    main()
