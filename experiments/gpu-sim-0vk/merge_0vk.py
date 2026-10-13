#!/usr/bin/env python3
"""gpu-sim-0vk MERGE (CPU): all raw slices into the shipped rows.

Consumes build/measured_{A,A2,C0..C7}_raw.jsonl (trial + MU files excluded:
trial rows are validation, MU shipped nothing). One row per hash asserted
(the skip chain prevents remeasure); driver_scalars joined from the manifest
(all hashes unique by construction assert, so manifest_occurrences is 1 and
the winner rule is trivial). Missing keep-hashes are SHORTFALL (named),
never rows -- item 3.
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o import corpus_protocol as cp  # noqa: E402


def int_metadata(d):
    return {k: v for k, v in d.get("metadata", {}).items() if type(v) is int}


def main() -> int:
    manifest = {d["kernel_hash"]: d for d in
                json.loads((HERE / "manifest.json").read_text())["specs"]}
    raws = sorted((HERE / "build").glob("measured_*_raw.jsonl"),
                  key=lambda p: p.name)
    raws = [p for p in raws if "_trial_" not in p.name and "_MU_" not in p.name
            and "trial" not in p.name]
    print("raws:", [p.name for p in raws])
    rows, seen = [], Counter()
    for rp in raws:
        for line in rp.read_text().splitlines():
            r = json.loads(line)
            if r["kernel_hash"] in seen:
                print(f"MERGE FAIL: {r['kernel_hash']} measured twice "
                      f"(also in {seen[r['kernel_hash']]})")
                return 1
            seen[r["kernel_hash"]] = rp.name
            w = manifest.get(r["kernel_hash"])
            if w is None:
                print(f"MERGE FAIL: row {r['kernel_hash']} joins no manifest cell")
                return 1
            r["driver_scalars"] = int_metadata(w)
            r["manifest_occurrences"] = 1
            r["predicted_class"] = w.get("predicted_class")
            r["predicted_ms"] = w.get("predicted_ms")
            rows.append(r)
    missing = [h for h in manifest if h not in seen]
    print(f"merged {len(rows)} unique rows; {len(missing)} keep-cells shortfall")
    with open(HERE / "shortfall.json", "w") as fh:
        json.dump([{"kernel_hash": h,
                    "op": manifest[h]["op_type"], "n": manifest[h]["problem_size"],
                    "grid": manifest[h]["grid"], "block": manifest[h]["block"],
                    "predicted_class": manifest[h].get("predicted_class")}
                   for h in missing], fh, indent=2)
    cp.write_rows(HERE / "measured_new.jsonl", rows)
    import statistics
    summary = {
        "bead": "gpu-sim-0vk", "stage": "merge",
        "n_manifest": len(manifest), "n_measured": len(rows),
        "n_shortfall": len(missing),
        "n_launch_bound": sum(1 for r in rows
                              if r["measured_regime"] == "launch_bound"),
        "n_measurable": sum(1 for r in rows
                            if r["measured_regime"] == "measurable"),
        "n_unreliable": sum(1 for r in rows if r.get("unreliable")),
        "median_ai_cv": statistics.median([r["ai_cv"] for r in rows]),
        "rows_path": "measured_new.jsonl",
        "protocol": "single_run_stages_2_5",
    }
    (HERE / "merge_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"MERGED {len(rows)} rows "
          f"({summary['n_measurable']} measurable, "
          f"{summary['n_unreliable']} unreliable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
