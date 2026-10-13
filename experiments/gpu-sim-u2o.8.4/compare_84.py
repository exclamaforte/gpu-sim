#!/usr/bin/env python3
"""gpu-sim-u2o.8.4 COMPARE (CPU): arms -> verdict vs the >5%/2% criteria.

Reads <run>/source|ptx|hybrid/arm_metrics.json. Criterion 1 (hybrid beats
best single-modality by >5% MAPE reduction, per suite, mean over suites):
verdict per suite + overall. Criterion 2 (a single modality within 2% of
hybrid): per suite + overall. Writes metrics.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "arms"
    arms = {}
    for arm in ("source", "ptx", "hybrid"):
        arms[arm] = json.loads((root / arm / "arm_metrics.json").read_text())
    cfgs = [{k: a[k] for k in ("n_params", "seed", "epochs", "batch", "lr",
                               "n_train")} for a in arms.values()]
    assert all(c == cfgs[0] for c in cfgs), f"arms not matched: {cfgs}"
    suites = sorted(arms["source"]["suites"])
    per_suite, c1_wins, c2_close = {}, 0, {"source": 0, "ptx": 0}
    for s in suites:
        m = {a: arms[a]["suites"][s]["mape"] for a in arms}
        best_single = min(m["source"], m["ptx"])
        red = (best_single - m["hybrid"]) / best_single if best_single else 0.0
        c1 = red > 0.05
        c1_wins += c1
        close = {a: (m[a] - m["hybrid"]) / m["hybrid"] < 0.02
                 if m["hybrid"] else False for a in ("source", "ptx")}
        for a in close:
            c2_close[a] += close[a]
        per_suite[s] = {"mape": m, "n": arms["source"]["suites"][s]["n"],
                        "hybrid_reduction": round(red, 4),
                        "criterion1_hybrid_wins": c1,
                        "criterion2_within_2pct": close}
    metrics = {"bead": "gpu-sim-u2o.8.4", "stage": "compare",
               "matched": cfgs[0],
               "train_mape": {a: arms[a]["train_mape"] for a in arms},
               "per_suite": per_suite,
               "criterion1_suites_won": f"{c1_wins}/{len(suites)}",
               "criterion2_suites_close": {a: f"{c}/{len(suites)}"
                                           for a, c in c2_close.items()},
               "verdict": ("hybrid" if c1_wins > len(suites) / 2
                           else "single-modality-sufficient"
                           if any(c > len(suites) / 2 for c in c2_close.values())
                           else "inconclusive")}
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "per_suite"}, indent=2))
    for s, p in per_suite.items():
        print(f"  {s}: n={p['n']} " +
              " ".join(f"{a}={p['mape'][a]:.4f}" for a in ("source", "ptx", "hybrid")) +
              f" hybrid_red={p['hybrid_reduction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
