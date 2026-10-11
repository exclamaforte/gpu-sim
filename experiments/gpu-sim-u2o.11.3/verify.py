#!/usr/bin/env python3
"""u2o.11.3 VERIFY (CPU): join, census, and acceptance gates over measured rows.

Reads measured.jsonl + manifest.json (the COMMITTED copy at this directory,
so a fresh checkout recomputes) + measure/compile summaries; writes
metrics.json. Nonzero exit on any gate failure. No GPU.

HASH REALITY: the manifest holds occurrences over unique cells
(kernel_hash_for does not cover driver scalar values; phantom duplicates
documented in merge_slices.py). Every count below is over UNIQUE CELLS;
the join is per row against its cell. See merge_slices.py, which annotates
each row with the driver scalars it launched with.

Gates (acceptance in parentheses):
  join: every row's kernel_hash recomputed from its manifest cell
    (source/grid/block/n) matches; every row joins (1)
  stages: all four stages present with per-stage counts (2)
  census: 9.4's regime construction recomputed live per row via
    census.classify -- the SAME function predict_mm.py used for the
    extension keep-list, so prediction and verdict cannot drift (3,4)
  noise: ai_cv present on every row; unreliable marked in place, row count
    unchanged (5)
  contention: per-row queue_state carries own_job_id + n_other_running;
    per-slice contention dispositions present, all four stages covered,
    and reported (6)
  split: compile/measure walls + queue-at-submit recorded (7)
  anchor: path filter proves anchor_v1 untouched (8)
  item9: row count AND census both present (9)
  protocol: single protocol over all rows (corpus condition)
"""
import json
import statistics
import argparse
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from epic_u2o.generator_stage1 import kernel_hash_for  # noqa: E402
from epic_u2o.features import LaunchConfig, extract_features  # noqa: E402
from epic_u2o import corpus_protocol as cp  # noqa: E402
from census import census, require_census, classify, check_git_clean_of_anchors  # noqa: E402

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"GATE FAIL: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", default=None)
    args = ap.parse_args()
    metrics_path = Path(args.metrics_out) if args.metrics_out else HERE / "metrics.json"
    rows = [json.loads(l) for l in (HERE / "measured.jsonl").read_text().splitlines()]
    manifest = json.loads((HERE / "manifest.json").read_text())
    by_hash = {d["kernel_hash"]: d for d in manifest["specs"]}
    ms = json.loads((HERE / "measure_summary.json").read_text())
    cs = json.loads((HERE / "build" / "compile_summary.json").read_text())

    # ---- 1. join (unique cells) ----
    unjoined, mismatched = [], []
    for r in rows:
        d = by_hash.get(r["kernel_hash"])
        if d is None:
            unjoined.append(r["kernel_hash"])
            continue
        if kernel_hash_for(d["source"], d["grid"], d["block"],
                           d["problem_size"]) != r["kernel_hash"]:
            mismatched.append(r["kernel_hash"])
    if unjoined:
        fail(f"{len(unjoined)} rows join to no manifest cell, e.g. {unjoined[:3]}")
    if mismatched:
        fail(f"{len(mismatched)} rows fail hash re-computation, e.g. {mismatched[:3]}")
    if len({r["kernel_hash"] for r in rows}) != len(rows):
        fail("measured.jsonl contains duplicate hashes")

    # ---- 2. stages ----
    per_stage = Counter(r["stage"] for r in rows)
    for s in ("stage2", "stage3", "stage4", "stage5"):
        if per_stage.get(s, 0) == 0:
            fail(f"no rows from {s}: not a four-stage corpus")
    print(f"per-stage rows: {dict(sorted(per_stage.items()))}")

    # ---- protocol ----
    try:
        proto = cp.assert_single_protocol(rows, what="u2o.11.3 rows")
    except Exception as e:
        fail(f"protocol: {e}")
        proto = None
    if proto != "single_run_stages_2_5":
        fail(f"protocol is {proto!r}, not the registered scale-up protocol")

    # ---- 5. noise ----
    missing_cv = [r["kernel_hash"] for r in rows if r.get("ai_cv") is None]
    if missing_cv:
        fail(f"{len(missing_cv)} rows lack ai_cv, e.g. {missing_cv[:3]}")
    n_unrel = sum(1 for r in rows if r.get("unreliable"))
    print(f"ai_cv present on {len(rows) - len(missing_cv)}/{len(rows)}; "
          f"unreliable (in place): {n_unrel}")

    # ---- 6. queue_state + contention (per slice job, all stages covered) ----
    bad_q = [r["kernel_hash"] for r in rows
             if not r.get("queue_state", {}).get("own_job_id")
             or r["queue_state"].get("n_other_running") is None]
    if bad_q:
        fail(f"{len(bad_q)} rows lack queue_state fields, e.g. {bad_q[:3]}")
    slices = ms.get("per_stage", [])
    covered = sorted({s.get("stage") for s in slices})
    if covered != ["stage2", "stage3", "stage4", "stage5"]:
        fail(f"contention dispositions cover {covered}, not all four stages")
    if not all(isinstance(s.get("contention_disposition"), dict) for s in slices):
        fail("a slice lacks its contention disposition (rule not run)")
    fired = sorted({s["stage"] for s in slices
                    if (s.get("contention_disposition") or {}).get("fired")})
    print(f"contention rule ran in {len(slices)} slice jobs across {covered}; "
          f"fired on: {fired or 'none'}")

    # ---- 3/4. regime census (same classify() as the keep-list) ----
    counts, s4_rows = Counter(), 0
    for r in rows:
        d = by_hash[r["kernel_hash"]]
        g, b = r["launch_config"]["grid"], r["launch_config"]["block"]
        lc = LaunchConfig(grid_x=g[0], grid_y=g[1], grid_z=g[2],
                          block_x=b[0], block_y=b[1], block_z=b[2],
                          shared_mem_bytes=r["launch_config"]["shared_mem_bytes"])
        counts[classify(extract_features(d["source"], lc, r["problem_size"]))] += 1
        if r["stage"] == "stage4":
            s4_rows += 1
    cen = census(dict(counts))
    print(f"regime census: {json.dumps(cen)}")
    if cen["verdict"] != "PASS":
        fail(f"census verdict FAIL: {cen['reasons']}")
    if s4_rows < 64:
        fail(f"stage-4 population {s4_rows} < 64 rows")
    print(f"stage-4 measured rows: {s4_rows}")

    # ---- 7. split discipline ----
    if not cs.get("wall_sec") or not ms.get("wall_sec"):
        fail("compile/measure durations missing")
    qat = HERE / "queue_at_submit.txt"
    if not qat.exists() or not qat.read_text().strip():
        fail("queue state at submit not recorded")
    print(f"compile wall {cs.get('wall_sec'):.0f}s (CPU, unqueued) / "
          f"measure wall {ms.get('wall_sec'):.0f}s (queued slices)")

    # ---- 8. anchor_v1 path filter ----
    anchor_touched = check_git_clean_of_anchors()
    if anchor_touched:
        fail(f"anchor paths touched: {anchor_touched}")
    print("anchor filter: 0 paths under anchor paths")

    metrics = {
        "bead": "gpu-sim-u2o.11.3",
        "total_rows": len(rows),
        "manifest_occurrences": ms.get("n_manifest_occurrences"),
        "unique_cells": ms.get("n_unique_cells"),
        "phantom_duplicates": ms.get("n_phantom_duplicates"),
        "join": {"unjoined": len(unjoined), "mismatched": len(mismatched)},
        "per_stage_rows": dict(sorted(per_stage.items())),
        "protocol": proto,
        "regime_census": cen,
        "stage4_rows": s4_rows,
        "ai_cv": {"missing": len(missing_cv),
                  "median": statistics.median([r["ai_cv"] for r in rows]),
                  "unreliable_in_place": n_unrel},
        "queue_state": {"rows_missing_fields": len(bad_q),
                        "contention_fired_stages": fired},
        "split": {"compile_wall_sec": cs.get("wall_sec"),
                  "measure_wall_sec": ms.get("wall_sec"),
                  "queue_at_submit_recorded": qat.exists()},
        "anchor_v1_untouched": not anchor_touched,
        "shortfall": {"preflight": manifest.get("preflight_failures", []),
                      "compile": cs.get("n_failed"),
                      "measure_missing_unique": ms.get("n_missing_unique"),
                      "missing_hashes": ms.get("missing_hashes")},
    }
    # ---- 9. the trap: total AND census, or fail here ----
    try:
        require_census(metrics)
    except ValueError as e:
        fail(f"item 9: {e}")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nrows={len(rows)} (need >= 1544 additional)")
    if len(rows) < 1544:
        fail(f"only {len(rows)} rows (< 1544 additional)")
    if FAILURES:
        print(f"\nVERIFY FAILED: {len(FAILURES)} gates")
        return 1
    print("VERIFY PASS: all gates green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
