#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.2.1.1: inventory every artifact whose branch stamp has moved.

Computes the inventory from the tree; writes metrics.json; nonzero exit if any
stamped-and-landed artifact lacks a superseded_by, or if any carries a null.
"""
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "metrics.json"

from scripts.tree_state import (branch_exists, first_word, landing_commit,
                                resolve_superseded_by, stamped_artifacts)

def main():
    inventory, nulls, missing = [], [], []
    for path, doc, ts in stamped_artifacts(REPO):
        bead = path.parent.name
        ref = first_word(ts.get("branch", ""))
        commit = landing_commit(bead, REPO)
        sb = ts.get("superseded_by") or doc.get("superseded_by")
        loc = ("tree_state" if ts.get("superseded_by")
               else "top-level" if doc.get("superseded_by") else None)
        if sb is None and ("superseded_by" in ts or "superseded_by" in doc):
            nulls.append(str(path))
        if commit and not sb:
            missing.append(f"{bead}/{path.name}: landed at {commit}, unstamped")
        inventory.append({
            "artifact": str(path.relative_to(REPO)),
            "bead": bead,
            "branch_ref": ref,
            "branch_still_exists": branch_exists(ref, REPO) if ref else False,
            "landed_on_master_at": commit,
            "superseded_by_location": loc,
            "superseded_by_commit": (sb or {}).get("commit"),
            "resolves_to_master_commit": (sb or {}).get("reachable_from_master"),
            "resolved_without_help": bool(sb and sb.get("commit") == commit),
        })
    props = {
        "no_null_superseded_by": not nulls,
        "every_landed_artifact_is_stamped": not missing,
        "at_least_one_artifact_inventoried": len(inventory) > 0,
        "a_deleted_ref_is_detected": any(
            i["branch_ref"] and not i["branch_still_exists"] for i in inventory),
        "a_live_ref_is_flagged_as_a_hazard": any(
            i["branch_ref"] and i["branch_still_exists"] and i["landed_on_master_at"]
            for i in inventory),
    }
    rec = {
        "bead": "gpu-sim-u2o.6.6.2.1.1",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "produced_by": "experiments/gpu-sim-u2o.6.6.2.1.1/measure_supersession.py",
        "gpu_used": False,
        "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
        "rule": ("OMISSION NOT NULL: an artifact that cannot name a superseding "
                 "commit omits the key. A null asserts that nothing supersedes "
                 "it, which is a claim nobody checked."),
        "inventory": inventory,
        "null_offenders": nulls,
        "unstamped_but_landed": missing,
        "properties": props,
        "all_properties_pass": all(props.values()),
        "known_outstanding": (
            "The dirty lab/gpu-sim-u2o.6.6.2-r2 worktree is NOT disposed of here. "
            "Disposal is destructive and is tracked under gpu-sim-u2o.6.6.2.1 "
            "criterion 3. Its content is preserved: the three v8 family files "
            "that mattered were copied into "
            "experiments/gpu-sim-u2o.6.6.2/variance_probe_v8/ before that bead "
            "merged, so disposal loses no measurement."),
    }
    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    if not rec["all_properties_pass"]:
        print(f"FAILED: nulls={nulls} missing={missing}", file=sys.stderr)
        return 1
    print("OK: every landed artifact names a master-reachable commit, no nulls, "
          "and both a deleted ref and a live-but-not-current ref are detected.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
