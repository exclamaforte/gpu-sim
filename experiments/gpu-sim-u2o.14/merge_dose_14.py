#!/usr/bin/env python3
"""gpu-sim-u2o.14 DOSE MERGE (CPU): raw GPU rows -> paired artifact.

The queued job measured all 48 cells but died stamping the protocol
(dose_14 unregistered at the time). Measurements are intact in
build/dose_raw.jsonl; this asserts one row per manifest hash (the spec-id
-> trial-row pairing) and writes dose_measured.jsonl + dose_summary.json.
Re-runnable byte-identical (run.sh).
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o import corpus_protocol as cp  # noqa: E402


def main() -> int:
    man = {d["kernel_hash"]: d for d in
           json.loads((HERE / "dose_manifest.json").read_text())["specs"]}
    rows = [json.loads(l) for l in
            (HERE / "build" / "dose_raw.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["kernel_hash"] for r in rows}), "duplicate rows"
    assert {r["kernel_hash"] for r in rows} <= set(man), "row joins no spec"
    missing = [h for h in man if h not in {r["kernel_hash"] for r in rows}]
    assert cp.assert_single_protocol(rows, what="dose") == "dose_14"
    cp.write_rows(HERE / "dose_measured.jsonl", rows)
    stamps = sorted(r["measured_at"] for r in rows)
    summary = {"bead": "gpu-sim-u2o.14", "stage": "dose",
               "n_specs": len(man), "n_measured": len(rows),
               "n_shortfall": len(missing), "shortfall": missing,
               "n_launch_bound": sum(1 for r in rows
                                    if r["measured_regime"] == "launch_bound"),
               "n_measurable": sum(1 for r in rows
                                   if r["measured_regime"] == "measurable"),
               "median_ai_cv": statistics.median([r["ai_cv"] for r in rows]),
               "span_first": stamps[0], "span_last": stamps[-1],
               "protocol": "dose_14",
               "rows_path": "dose_measured.jsonl"}
    (HERE / "dose_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"MERGED {len(rows)}/{len(man)} rows "
          f"({summary['n_measurable']} measurable, "
          f"{summary['n_launch_bound']} launch_bound), "
          f"median_ai_cv={summary['median_ai_cv']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
