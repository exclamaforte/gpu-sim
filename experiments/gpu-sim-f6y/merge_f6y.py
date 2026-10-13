#!/usr/bin/env python3
"""gpu-sim-f6y MERGE (CPU): the 18 fixed-template rows.

One row per new kernel hash (18/18 groups measured exactly once; the other
62 occurrences are phantom-unlink events -- the binary their group already
consumed -- identical in kind to 11.3's 500). Winner per hash: the LAST
manifest occurrence (compile_all built in order, later duplicates overwrote
earlier ones -- the binary that ran). Each row carries that winner's driver
scalars and the group's occurrence count, same as merge_slices.py.
"""
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
    manifest = json.loads((HERE / "manifest.json").read_text())["specs"]
    winner, occurrences = {}, Counter()
    for d in manifest:
        occurrences[d["kernel_hash"]] += 1
        winner[d["kernel_hash"]] = d
    print(f"manifest: {len(manifest)} occurrences -> {len(winner)} unique cells "
          f"({sum(n - 1 for n in occurrences.values() if n > 1)} phantom duplicates)")
    assert len(winner) == 18, len(winner)

    rows = [json.loads(l) for l in
            (HERE / "measured_f6y_raw_stamped.jsonl").read_text().splitlines()]
    seen = Counter(r["kernel_hash"] for r in rows)
    dup = [h for h, n in seen.items() if n > 1]
    if dup:
        print(f"MERGE FAIL: {len(dup)} hashes measured twice, e.g. {dup[:3]}")
        return 1
    missing = [h for h in winner if h not in seen]
    if missing:
        print(f"MERGE FAIL: {len(missing)} fixed hashes with no row: {missing}")
        return 1
    for r in rows:
        w = winner.get(r["kernel_hash"])
        if w is None:
            print(f"MERGE FAIL: row {r['kernel_hash']} joins no manifest cell")
            return 1
        r["driver_scalars"] = int_metadata(w)
        r["manifest_occurrences"] = occurrences[r["kernel_hash"]]
    import statistics
    cp.write_rows(HERE / "measured_new.jsonl", rows)
    summary = json.loads((HERE / "measure_summary.json").read_text())
    summary["n_unique_cells"] = len(winner)
    summary["n_rows_shipped"] = len(rows)
    summary["rows_path"] = "measured_new.jsonl"
    (HERE / "measure_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"MERGED {len(rows)} unique rows "
          f"({sum(1 for r in rows if r['op_type'] == 'segmented_read')} seg + "
          f"{sum(1 for r in rows if r['op_type'] == 'strided_write')} strided)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
