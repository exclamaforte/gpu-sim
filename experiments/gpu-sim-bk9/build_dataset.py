#!/usr/bin/env python3
"""Build a supervised dataset from bk9 cells; refuse if any selected row is not target_usable."""
import argparse, json, sys, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="experiments/gpu-sim-bk9/metrics.json")
    ap.add_argument("--rows", default="all", help="comma-separated family:size ids, or 'all'")
    ap.add_argument("--out", default="experiments/gpu-sim-bk9/dataset.json")
    args = ap.parse_args()
    metrics = json.loads(pathlib.Path(args.metrics).read_text())
    cells = metrics["cells"]
    by_id = {f"{c['family']}:{c['size']}": c for c in cells}
    if args.rows == "all":
        selected = list(by_id.keys())
    else:
        selected = [r.strip() for r in args.rows.split(",") if r.strip()]
    unknown = [r for r in selected if r not in by_id]
    if unknown:
        print("REFUSED: unknown row ids: " + ", ".join(unknown), file=sys.stderr)
        sys.exit(1)
    offending = [r for r in selected if not by_id[r]["target_usable"]]
    if offending:
        print("REFUSED: target_usable=false rows: " + ", ".join(offending), file=sys.stderr)
        sys.exit(1)
    dataset = {"rows": [by_id[r] for r in selected]}
    pathlib.Path(args.out).write_text(json.dumps(dataset, indent=2) + "\n")
    print(f"built dataset with {len(selected)} rows -> {args.out}")

if __name__ == "__main__":
    main()
