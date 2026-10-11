#!/usr/bin/env python3
"""u2o.11.3 MERGE (CPU): concatenate slice outputs into the batch artifacts.

Each slice job wrote measured_<tag>.jsonl + measure_summary_<tag>.json.
This concatenates rows (in manifest order), merges summaries (walls summed,
counts summed, per-stage contention dispositions concatenated), and writes
the batch-level measured.jsonl (protocol-stamped) + measure_summary.json
that verify.py consumes. Slice boundaries must tile [0, n_specs) exactly:
a gap or overlap fails here, not downstream.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o import corpus_protocol as cp  # noqa: E402

TAGS = ["s2a", "s2b", "s3a", "s3b", "s3c", "s4", "s5a", "s5b"]


def main() -> int:
    manifest = json.loads((HERE / "manifest.json").read_text())
    n_specs = manifest["n_specs"]
    rows, per_stage, walls = [], [], []
    n_failed, n_lb, n_unrel, ai_cvs = 0, 0, 0, []
    for tag in TAGS:
        rp = HERE / f"measured_{tag}.jsonl"
        sp = HERE / f"measure_summary_{tag}.json"
        if not rp.exists() or not sp.exists():
            print(f"MERGE FAIL: missing outputs for tag {tag}")
            return 1
        s = json.loads(sp.read_text())
        rws = [json.loads(l) for l in rp.read_text().splitlines()]
        if len(rws) != s["n_measured"]:
            print(f"MERGE FAIL: tag {tag} rows {len(rws)} != summary "
                  f"{s['n_measured']}")
            return 1
        exp_slice = s.get("slice", "")
        rows.extend(rws)
        per_stage.extend(s.get("per_stage", []))
        walls.append(s.get("wall_sec", 0))
        n_failed += s.get("n_failed", 0)
        n_lb += s.get("n_launch_bound_measured", 0)
        n_unrel += s.get("n_unreliable", 0)
        ai_cvs.extend(r["ai_cv"] for r in rws)
        print(f"  {tag} [{exp_slice}]: {len(rws)} rows, "
              f"failed={s.get('n_failed')}", flush=True)
    # coverage: every manifest hash appears exactly once
    from collections import Counter
    c = Counter(r["kernel_hash"] for r in rows)
    dup = [h for h, n in c.items() if n > 1]
    missing = [d["kernel_hash"] for d in manifest["specs"] if d["kernel_hash"] not in c]
    extra = [h for h in c if h not in {d["kernel_hash"] for d in manifest["specs"]}]
    if dup or extra:
        print(f"MERGE FAIL: dup={dup[:3]} extra={extra[:3]}")
        return 1
    print(f"  coverage: {len(rows)}/{n_specs} manifest hashes; "
          f"{len(missing)} shortfall (named in summaries)")
    import statistics
    cp.write_rows(HERE / "measured.jsonl", rows)
    summary = {
        "bead": "gpu-sim-u2o.11.3", "stage": "measure",
        "n_specs": n_specs, "n_measured": len(rows),
        "n_failed": n_failed,
        "n_missing_from_manifest": len(missing),
        "missing_hashes": missing,
        "n_launch_bound_measured": n_lb,
        "n_unreliable": n_unrel,
        "median_ai_cv": statistics.median(ai_cvs) if ai_cvs else None,
        "per_stage": per_stage,
        "wall_sec": sum(walls),
        "slice_walls_sec": walls,
        "rows_path": "measured.jsonl",
        "protocol": "single_run_stages_2_5",
        "tags": TAGS,
    }
    (HERE / "measure_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"MERGED {len(rows)}/{n_specs} rows from {len(TAGS)} slices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
