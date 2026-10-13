#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.3 rework: reshape the committed artifact, no re-measurement.

Review verdict (three items, explicitly no GPU/re-run): the criterion names
fields the artifact does not use, per-invocation rows carry no provenance,
and the discard rule's floor lives apart from the rule -- plus the PI's later
rulings (drop the clocks.sm clause, regime-weight the thresholds, drop the
always-true 'discriminates' boolean, state the partial-tenant limitation).

Every number here comes from the committed metrics.json produced by ts job
20260928t072008-b34ddec9. Nothing is re-measured; the shaping functions are
imported from measure_contention.py so the artifact and any future run are
produced by the same code. Deterministic: re-running must byte-reproduce
metrics.json (modulo the stamp block, which names its own commit).

CPU only. Never through ts (it imports the measurement module for its pure
functions and never calls measure() or run_kernel).
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from measure_contention import (  # noqa: E402
    CONDS,
    analyse_detection,
    build_discard_rule,
    headline_number,
    shape_cells_for_criterion,
)


def commit_of(path: Path) -> str:
    r = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h",
                        "--", str(path)], capture_output=True, text=True)
    return r.stdout.strip()


def main() -> int:
    src = HERE / "metrics.json"
    old = json.loads(src.read_text())

    cells = old["cells"]
    tel = old["telemetry_delta"]

    # Item 2: every per-invocation row carries its condition's queue_state.
    for name, cell in cells.items():
        for cond in CONDS:
            blk = cell.get(cond) or {}
            q = (tel.get(cond) or {}).get("queue_state") or {}
            prov = {
                "own_job_id": q.get("own_job_id"),
                "n_other_active": q.get("n_other_active"),
                "overlapping_job_ids": q.get("other_active_jobs", []),
                "condition": cond,
            }
            for row in blk.get("per_invocation", []):
                row["queue_state"] = dict(prov)

    # Item 1: criterion-named flat keys (nested blocks stay authoritative).
    shaped = shape_cells_for_criterion(cells)

    # Detection block regenerated from the same telemetry by the same code
    # (no 'discriminates' boolean; ranking by relative_spread instead).
    detection = analyse_detection(tel)
    for s, v in old["detection"]["signals"].items():
        assert v["relative_spread"] == detection["signals"][s]["relative_spread"], (
            f"signal {s}: committed spread {v['relative_spread']} != "
            f"recomputed {detection['signals'][s]['relative_spread']}")

    # Headline recomputed from the same numbers; ratios must be identical.
    new_headline = headline_number(
        {k: v for k, v in shaped.items()}, {n: None for n in shaped})
    for new_row, old_row in zip(new_headline["rows"], old["headline"]["rows"]):
        assert new_row["cell"] == old_row["cell"], "headline row order moved"
        assert abs(new_row["ratio"] - old_row["ratio"]) < 1e-12, (
            f"headline ratio moved for {new_row['cell']}")

    new = dict(old)
    new["cells"] = shaped
    new["detection"] = detection
    new["headline"] = new_headline
    if "superseded_20260928_rule" not in old:
        new["superseded_20260928_rule"] = old["discard_rule"]
    new["discard_rule"] = build_discard_rule()
    new["rework_provenance"] = {
        "ts_job": "20260928t072008-b34ddec9 (no re-measurement; all numbers "
                  "are the committed ones, reshaped)",
        "committed_metrics_commit": commit_of(
            Path("experiments/gpu-sim-u2o.6.6.3/metrics.json")),
        "rewrite_script": "experiments/gpu-sim-u2o.6.6.3/rewrite_artifact.py",
        "rewrite_commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "deterministic": ("re-running this script byte-reproduces metrics.json "
                          "except this stamp block"),
    }

    src.write_text(json.dumps(new, indent=2) + "\n")
    print(f"cells: {sorted(shaped)}")
    print(f"headline statement: {new_headline['statement']}")
    print("wrote metrics.json (reshaped, not re-measured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
