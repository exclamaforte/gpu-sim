#!/usr/bin/env python3
"""Regenerate AC3 numbers (gate re-run verdicts) from on-disk rows and
deep-compare against the committed ac3_table.json. Exits nonzero on mismatch."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.grid_limits import grid_capped

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-8zqr"))
from measure import nearest_residual

STAGEC = REPO / "experiments" / "gpu-sim-8zqr" / "measured_stageC.jsonl"
POWERED = REPO / "experiments" / "gpu-sim-8zqr" / "powered_rows.jsonl"
JF = REPO / "experiments" / "gpu-sim-1jf" / "metrics.json"
COMMITTED = HERE / "ac3_table.json"
SCRATCH = REPO / ".lab" / "tmp" / "ac3_table.recomputed.json"


def old_usable(row):
    kernel_us = row["kernel_us"]
    within = row["within_std_us"]
    residual = row["residual_us"]
    threshold = 5 * max(within, residual) if (within > 0 or residual > 0) else 0.0
    gc = bool(row["grid_capped"])
    return (kernel_us >= threshold) and (not gc) and (kernel_us > 0)


def new_usable(row, family):
    kernel_us = row["kernel_us"]
    within = row["within_std_us"]
    residual = row["residual_us"]
    compute_bound = (family in ("gemm_naive", "matmul"))
    if compute_bound and residual > 0:
        threshold = 5 * residual
    else:
        threshold = 5 * max(within, residual) if (within > 0 or residual > 0) else 0.0
    gc = bool(grid_capped(row["grid"])) if "grid" in row else bool(row["grid_capped"])
    return (kernel_us >= threshold) and (not gc) and (kernel_us > 0)


def build():
    # 8zqr 16kb strided_read: rows with size==16384 in measured_stageC
    stagec = [json.loads(l) for l in STAGEC.read_text().splitlines() if l.strip()]
    kb16 = [r for r in stagec if r.get("size") == 16384]
    kb16_old_rejected = sum(1 for r in kb16 if not old_usable(r))
    kb16_new_rejected = sum(1 for r in kb16 if not new_usable(r, r.get("family", "")))
    kb16_total = len(kb16)

    # 8zqr powered: usable unchanged count
    powered = [json.loads(l) for l in POWERED.read_text().splitlines() if l.strip()]
    unchanged = sum(1 for r in powered if old_usable(r) == new_usable(r, r.get("family", "")))
    powered_total = len(powered)
    powered_usable_old = sum(1 for r in powered if old_usable(r))
    powered_usable_new = sum(1 for r in powered if new_usable(r, r.get("family", "")))

    # 1jf corpus: 807 rows in experiments/gpu-sim-1jf/metrics.json
    jf_rows = json.loads(JF.read_text())["rows"]
    jf_old_usable = sum(1 for r in jf_rows if old_usable(r))
    jf_new_usable = sum(1 for r in jf_rows if new_usable(r, r.get("family", "")))
    jf_flips = sum(1 for r in jf_rows if old_usable(r) != new_usable(r, r.get("family", "")))
    jf_total = len(jf_rows)

    table = {
        "8zqr_16kb_strided_read": {
            "total": kb16_total,
            "old_rejected": kb16_old_rejected,
            "new_rejected": kb16_new_rejected,
            "unchanged_rejected": (kb16_old_rejected == kb16_new_rejected and kb16_old_rejected == kb16_total),
        },
        "8zqr_powered": {
            "total": powered_total,
            "usable_old": powered_usable_old,
            "usable_new": powered_usable_new,
            "flips": powered_total - unchanged,
        },
        "1jf_corpus": {
            "total": jf_total,
            "usable_old": jf_old_usable,
            "usable_new": jf_new_usable,
            "flips": jf_flips,
        },
    }
    return table


def main():
    table = build()
    SCRATCH.parent.mkdir(parents=True, exist_ok=True)
    SCRATCH.write_text(json.dumps(table, indent=2) + "\n")
    committed = json.loads(COMMITTED.read_text())
    if table != committed:
        # find first diff
        def find_diff(a, b, path=""):
            if type(a) != type(b):
                return path or "<root>", a, b
            if isinstance(a, dict):
                for k in a:
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
    print(f"OK: AC3 table recomputes identically")


if __name__ == "__main__":
    main()
