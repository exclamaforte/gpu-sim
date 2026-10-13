#!/usr/bin/env python3
"""u2o.11.3 MERGE (CPU): concatenate slice outputs into the batch artifacts.

Each slice job wrote measured_<tag>.jsonl + measure_summary_<tag>.json.
This concatenates rows (in manifest order), merges summaries (walls summed,
counts summed, per-stage contention dispositions concatenated), annotates
every row with the driver scalars it actually launched with (see below),
and writes the batch-level measured.jsonl (protocol-stamped) +
measure_summary.json that verify.py consumes.

HASH REALITY (found in the slices, recorded here): kernel_hash_for covers
(source, grid, block, problem_size) but NOT driver scalar values, so 2211
base occurrences collapse to 1711 unique cells; 500 are phantom duplicates
(same binary, consumed once). Binaries were built in manifest order with
later duplicates overwriting earlier ones, so the winner -- and the
scalars any row in the group launched with -- is the LAST manifest
occurrence. Deterministic and recomputable; each row carries it.

The mm-regime extension (111 appended specs, all with distinct hashes by
construction assert) adds unique cells with no phantoms.

Slice shortfall (18 unique cells, all with binaries present but 0 successful
invocations): segmented_read and strided_write templates index out of bounds
by construction (in[segment*n+offset], out[i*stride]), so those kernels
crash. Generator defect, named; filed as follow-up, not fixed here.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o import corpus_protocol as cp  # noqa: E402

TAGS = ["s2a", "s2b", "s3a", "s3b", "s3c", "s4r", "s5a", "s5b", "mmx"]


def int_metadata(d):
    return {k: v for k, v in d.get("metadata", {}).items() if type(v) is int}


def main() -> int:
    manifest = json.loads((HERE / "manifest.json").read_text())
    specs = manifest["specs"]
    # winner per hash: last manifest occurrence (compile_all built in order,
    # later duplicates overwrote earlier ones -- the binary that ran).
    winner, occurrences = {}, Counter()
    for d in specs:
        occurrences[d["kernel_hash"]] += 1
        winner[d["kernel_hash"]] = d
    n_phantom = sum(n - 1 for n in occurrences.values() if n > 1)
    print(f"manifest: {len(specs)} occurrences -> {len(winner)} unique cells "
          f"({n_phantom} phantom duplicates)")

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
        for r in rws:
            w = winner.get(r["kernel_hash"])
            if w is None:
                print(f"MERGE FAIL: row {r['kernel_hash']} joins no manifest cell")
                return 1
            r["driver_scalars"] = int_metadata(w)
            r["manifest_occurrences"] = occurrences[r["kernel_hash"]]
        rows.extend(rws)
        per_stage.extend(s.get("per_stage", []))
        walls.append(s.get("wall_sec", 0))
        n_failed += s.get("n_failed", 0)
        n_lb += s.get("n_launch_bound_measured", 0)
        n_unrel += s.get("n_unreliable", 0)
        ai_cvs.extend(r["ai_cv"] for r in rws)
        print(f"  {tag} [{s.get('slice', '')}]: {len(rws)} rows, "
              f"failed={s.get('n_failed')}", flush=True)
    seen = Counter(r["kernel_hash"] for r in rows)
    dup = [h for h, n in seen.items() if n > 1]
    if dup:
        print(f"MERGE FAIL: {len(dup)} hashes measured twice, e.g. {dup[:3]}")
        return 1
    missing = [h for h in winner if h not in seen]
    print(f"  coverage: {len(rows)}/{len(winner)} unique cells; "
          f"{len(missing)} never measured (shortfall, named in REPORT)")
    import statistics
    cp.write_rows(HERE / "measured.jsonl", rows)
    summary = {
        "bead": "gpu-sim-u2o.11.3", "stage": "measure",
        "n_manifest_occurrences": len(specs),
        "n_unique_cells": len(winner),
        "n_phantom_duplicates": n_phantom,
        "n_measured": len(rows),
        "n_failed_occurrences": n_failed,
        "n_missing_unique": len(missing),
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
    print(f"MERGED {len(rows)} unique rows from {len(TAGS)} slices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
