"""Recompute the 8yb4 summary from committed rows; optionally require a match.

Split out of measure_k.py so the CPU-only run.sh can reproduce metrics.json without
touching nvcc or a GPU.
"""
import argparse, json, statistics, sys
from pathlib import Path


def summarize(rows):
    ratios = [r["ratio_k1_over_k40"] for r in rows if r.get("ratio_k1_over_k40")]
    fixed = [r["implied_fixed_cost_us"] for r in rows]
    med_ratio = round(statistics.median(ratios), 4)
    return {
        "bead": "gpu-sim-8yb4",
        "question": ("Is per-launch kernel time flat in K? The corpus measures K=1 "
                     "launch per event window; the gate prices K=40."),
        "k_test": 40,
        "configs": len(rows),
        "median_ratio_k1_over_k40": med_ratio,
        "median_implied_fixed_cost_us": round(statistics.median(fixed), 3),
        "finding": ("per-launch time falls as K grows" if med_ratio > 1.2
                    else "per-launch time is flat in K"),
        "rows_file": "rows.jsonl",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.rows).read_text().splitlines() if l.strip()]
    s = summarize(rows)
    Path(a.out).write_text(json.dumps(s, indent=2) + "\n")
    if a.compare:
        committed = json.loads(Path(a.compare).read_text())
        if committed != s:
            print("METRICS_MISMATCH", file=sys.stderr)
            print(json.dumps(committed, sort_keys=True), file=sys.stderr)
            print(json.dumps(s, sort_keys=True), file=sys.stderr)
            return 1
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
