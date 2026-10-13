#!/usr/bin/env python3
"""gpu-sim-u2o.13.4: build per-op-class latency lookup from MEASURED rows.

Source: committed stage-1 digest files (456 rows). Values are medians of
measured median_runtime_ms per op_type. Coverage note: stage-1 sizes are
launch-bound tiny kernels -- the table documents its range, it does not
claim general latency modeling. D_GPUSIM: simulation/CI timing ONLY,
never training ground truth (metadata-carried).
"""
import collections
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = ["experiments/gpu-sim-u2o.7.1/measured.jsonl",
       "experiments/gpu-sim-e5q/measured.jsonl"]


def main(out=None) -> int:
    out = Path(out) if out else (REPO / "experiments/gpu-sim-u2o.13.4"
                                 / "op_latency_table.json")
    rows = []
    for f in SRC:
        rows += [json.loads(l) for l in (REPO / f).read_text().splitlines()]
    byop = collections.defaultdict(list)
    for r in rows:
        byop[r["op_type"]].append(float(r["median_runtime_ms"]))
    table = {
        "_meta": {
            "bead": "gpu-sim-u2o.13.4",
            "sources": SRC,
            "n_rows": len(rows),
            "statistic": "median of per-row median_runtime_ms",
            "unit": "ms",
            "coverage": "stage-1 small sizes, launch-bound; NOT a general "
                        "latency model",
            "d_gpusim": "CI/behavioral-simulation timing ONLY -- never "
                        "training ground truth",
        },
        "ops": {op: {"median_ms": statistics.median(vs), "n": len(vs)}
                for op, vs in sorted(byop.items())},
    }
    out.write_text(json.dumps(table, indent=2))
    for op, v in table["ops"].items():
        print(f"  {op}: {v['median_ms']:.4f} ms (n={v['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
