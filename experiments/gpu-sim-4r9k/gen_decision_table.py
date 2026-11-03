#!/usr/bin/env python3
"""Regenerate decision_table.json from measured_gemm.jsonl and deep-compare
against the committed file. Exits nonzero on any mismatch, naming the path."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from scripts.grid_limits import grid_capped

MEASURED = HERE / "measured_gemm.jsonl"
COMMITTED = HERE / "decision_table.json"
SCRATCH = REPO / ".lab" / "tmp" / "decision_table.recomputed.json"


def rejection_reason(kernel_us, threshold, gc):
    if kernel_us >= threshold and (not gc) and kernel_us > 0:
        return "accepted"
    if kernel_us <= 0:
        return "nonpositive_kernel_us"
    if gc:
        return "grid_capped"
    if kernel_us < threshold:
        return "below_threshold"
    return "unknown"


def build():
    rows = []
    old_usable = 0
    new_usable = 0
    for line in MEASURED.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        kernel_us = d["kernel_us"]
        within = d["within_std_us"]
        residual = d["residual_us"]
        compute_bound = (d["family"] == "gemm_naive")
        old_threshold = d["threshold_us"]
        old_gc = bool(d["grid_capped"])
        old_usable_flag = bool(d["target_usable"])
        if old_usable_flag:
            old_usable += 1
        old_reason = "accepted" if old_usable_flag else (
            "grid_capped" if old_gc else (
                "nonpositive_kernel_us" if kernel_us <= 0 else (
                    "below_threshold" if kernel_us < old_threshold else "unknown")))
        # new gate
        if compute_bound and residual > 0:
            new_threshold = 5 * residual
        else:
            new_threshold = 5 * max(within, residual) if (within > 0 or residual > 0) else 0.0
        new_gc = bool(grid_capped(d["grid"]))
        new_usable_flag = (kernel_us >= new_threshold) and (not new_gc) and (kernel_us > 0)
        if new_usable_flag:
            new_usable += 1
        new_reason = rejection_reason(kernel_us, new_threshold, new_gc)
        rows.append({
            "build_idx": d["build_idx"],
            "size": d["size"],
            "block": d["block"],
            "grid": d["grid"],
            "kernel_us": kernel_us,
            "within_std_us": within,
            "residual_us": residual,
            "old_threshold_us": old_threshold,
            "old_grid_capped": old_gc,
            "old_usable": old_usable_flag,
            "old_rejection_reason": old_reason,
            "new_threshold_us": new_threshold,
            "new_grid_capped": new_gc,
            "new_usable": new_usable_flag,
            "new_rejection_reason": new_reason,
        })
    table = {
        "n_specs": len(rows),
        "old_usable": old_usable,
        "new_usable": new_usable,
        "rows": rows,
    }
    return table


def main():
    table = build()
    SCRATCH.parent.mkdir(parents=True, exist_ok=True)
    SCRATCH.write_text(json.dumps(table, indent=2) + "\n")
    committed = json.loads(COMMITTED.read_text())
    if table != committed:
        # find first mismatching path
        def find_diff(a, b, path=""):
            if type(a) != type(b):
                return path or "<root>", a, b
            if isinstance(a, dict):
                for k in a if isinstance(a, dict) else []:
                    if k not in b:
                        return f"{path}/{k}", a.get(k), None
                    r = find_diff(a[k], b[k], f"{path}/{k}")
                    if r:
                        return r
                for k in b:
                    if k not in a:
                        return f"{path}/{k}", None, b.get(k)
            elif isinstance(a, list):
                if len(a) != len(b):
                    return f"{path}[len]", len(a), len(b)
                for i, (x, y) in enumerate(zip(a, b)):
                    r = find_diff(x, y, f"{path}[{i}]")
                    if r:
                        return r
            else:
                if a != b:
                    return path or "<root>", a, b
            return None
        r = find_diff(table, committed)
        if r:
            path, av, bv = r
            print(f"MISMATCH at {path}: recomputed={av!r} committed={bv!r}")
            sys.exit(1)
    print(f"OK: decision table recomputes identically ({table['n_specs']} rows, {table['old_usable']}->{table['new_usable']} usable)")


if __name__ == "__main__":
    main()
