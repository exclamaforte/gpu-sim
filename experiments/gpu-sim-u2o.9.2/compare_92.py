#!/usr/bin/env python3
"""gpu-sim-u2o.9.2 COMPARE (CPU): concat vs cross_attention holdout errors."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "arms"
    arms = {}
    for f in ("concat", "cross_attention"):
        arms[f] = json.loads((root / f / "run_metrics.json").read_text())
    cfgs = [{k: a[k] for k in ("epochs", "seed", "d_model", "n_heads",
                               "n_layers", "d_ff", "lr", "batch")}
            for a in arms.values()]
    assert all(c == cfgs[0] for c in cfgs), f"arms not matched: {cfgs}"
    assert arms["concat"]["split"]["sizes"] == \
        arms["cross_attention"]["split"]["sizes"], "split mismatch"
    metrics = {"bead": "gpu-sim-u2o.9.2", "matched": cfgs[0],
               "n_params": {f: arms[f]["n_params"] for f in arms},
               "per_arm_test": {
                   f: arms[f]["test_report"]["triple"] for f in arms},
               "per_arm_val": {
                   f: arms[f]["val_report"]["triple"] for f in arms}}
    lines = []
    for triple in ("latency->latency", "latency->memory"):
        c = metrics["per_arm_test"]["concat"][triple]["mean_abs_err"]
        x = metrics["per_arm_test"]["cross_attention"][triple]["mean_abs_err"]
        lines.append((triple, c, x))
    metrics["verdict"] = ("cross_attention" if all(x < c for _, c, x in lines)
                          else "concat" if all(c < x for _, c, x in lines)
                          else "split")
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for t, c, x in lines:
        print(f"  {t}: concat={c:.4f} cross_attn={x:.4f}")
    print("verdict:", metrics["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
