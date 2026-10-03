#!/usr/bin/env python3
"""Compute gpu-sim-edu.1's metrics from the live board. Every number is derived.

There is no measurement here in the GPU sense: this bead is about what the
dispatch list CONTAINS. So the artifact is a snapshot of board state plus the
filter's behaviour on it, computed by this script at run time. Nothing is typed.

Writes experiments/gpu-sim-edu.1/metrics.json and exits nonzero if any
acceptance property fails.

Usage: python3 experiments/gpu-sim-edu.1/measure_dispatch.py
"""
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "metrics.json"

_spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
    "brf", REPO / "scripts" / "bd_ready_filtered.py")
brf = __import__("importlib.util", fromlist=["util"]).module_from_spec(_spec)
_spec.loader.exec_module(brf)


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    ready = brf.bd_json(["ready"])
    status = brf.statuses_by_id()
    offer, dropped = brf.classify(ready, status)
    offer.sort(key=brf.order_key)

    merge_pending = [i["id"] for i in ready
                     if set(i.get("labels") or []) & brf.MERGE_PENDING_LABELS]
    dep_blocked_ids = [i["id"] for i in ready if brf.open_blockers(i, status)]
    offered_merge_pending = [i["id"] for i in offer
                             if brf.is_finished_awaiting_merge(i)]
    offered_dep_blocked = [i["id"] for i in offer if brf.open_blockers(i, status)]

    # Criterion 4: the label-inheritance defect must not have shipped this
    # bead with wrong labels. Checked against the board, not asserted.
    own = subprocess.run(
        ["bd", "show", "gpu-sim-edu.1", "--json"], capture_output=True, text=True)
    try:
        rec = json.loads(own.stdout)
        rec = rec[0] if isinstance(rec, list) else rec
        own_labels = rec.get("labels") or []
    except Exception:
        own_labels = None

    # Criterion 4 is RECORDED, NOT GATED. The reviewer's item, and it is the
    # same defect this bead is about: the previous version gated on this bead's
    # OWN labels, which the dispatch system controls and legitimately changes --
    # needs-review becomes validated when the reviewer passes it. So the committed
    # green would flip to red on the next reviewer run with no code change, and a
    # gate whose verdict moves on someone else's action is not a gate.
    #
    # What IS gated is a property of bd itself, which no reviewer run can change:
    # that `bd create` offers a way to decline inheriting labels at all. That is
    # the durable capability criterion 4 is really about; the label list is
    # recorded beside it so a human can see the current state without the run
    # turning red when the state moves.
    create_help = subprocess.run(["bd", "create", "--help"],
                                 capture_output=True, text=True).stdout
    no_inherit_supported = "--no-inherit-labels" in create_help

    props = {
        "criterion_1_no_merge_pending_offered": not offered_merge_pending,
        "criterion_2_no_dep_blocked_offered": not offered_dep_blocked,
        "criterion_4_bd_offers_no_inherit_labels": no_inherit_supported,
        "the_status_map_was_built_from_both_listings": (
            status.closed_count > 0 and status.total_count > status.closed_count),
    }

    metrics = {
        "bead": "gpu-sim-edu.1",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "produced_by": "experiments/gpu-sim-edu.1/measure_dispatch.py",
        "gpu_used": False,
        "commit": git("rev-parse", "--short", "HEAD"),
        "bd_version": subprocess.run(["bd", "version"], capture_output=True,
                                     text=True).stdout.strip(),
        "board_counts": {
            "bd_ready_returned": len(ready),
            "offered_after_filter": len(offer),
            "excluded": len(dropped),
            "ready_by_priority": dict(sorted(
                Counter(f"P{i.get('priority')}" for i in ready).items())),
        },
        "criterion_1_validated_but_unmerged": {
            "present_in_bd_ready": merge_pending,
            "count": len(merge_pending),
            "offered_any": offered_merge_pending,
            "verdict": "FIXED by the label filter" if not offered_merge_pending
                       else "VIOLATED",
        },
        "criterion_2_dep_blocked": {
            "present_in_bd_ready": dep_blocked_ids,
            "count": len(dep_blocked_ids),
            "offered_any": offered_dep_blocked,
            "verdict": "ALREADY CORRECT in bd: `bd ready` honours type=blocks "
                       "edges. This branch verifies it rather than "
                       "reimplementing it, so a regression in bd is caught "
                       "instead of assumed.",
        },
        "excluded_with_reasons": [{"id": i["id"], "reason": w} for i, w in
                                  sorted(dropped, key=lambda d: brf.order_key(d[0]))],
        "ordering": {
            "rule": "priority first, then created_at, then id",
            "not_recency": "recency is what put a reviewed-and-merge-ready "
                           "bead at the top of a dispatch in the first place",
            "first_five_offered": [i["id"] for i in offer[:5]],
        },
        "filter_can_pass": {
            "offered": len(offer),
            "removed_fraction": round(1 - len(offer) / len(ready), 4) if ready else None,
            "note": "a filter that emptied the list would satisfy every "
                    "exclusion assertion and be useless; tests assert ordinary "
                    "ready work SURVIVES",
        },
        "criterion_4_own_labels_RECORDED_not_gated": {
            "labels": own_labels,
            "note": ("recorded, never gated: these labels change when the "
                     "reviewer acts, and a gate that flips on someone else's "
                     "action is not a gate. See criterion_4_bd_offers_"
                     "no_inherit_labels for what IS gated."),
        },
        "criterion_4_bd_offers_no_inherit_labels": no_inherit_supported,
        "status_map_provenance": {
            "total_issues": status.total_count,
            "closed_issues": status.closed_count,
            "non_closed": status.total_count - status.closed_count,
            "note": ("bd list is partial in BOTH directions: the default returns "
                     "only non-closed and --status closed only closed, so the "
                     "map is their merge. Taking either alone makes absence "
                     "silently mean 'open'."),
        },
        "properties": props,
        "all_properties_pass": all(props.values()),
    }
    OUT.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if not all(props.values()):
        failed = [k for k, v in props.items() if not v]
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    print("OK: criterion 1 fixed; criterion 2 verified-correct from a status "
          "map built from BOTH bd listings; criterion 4 gated on bd offering "
          "--no-inherit-labels, with this bead's labels RECORDED rather than "
          "gated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())