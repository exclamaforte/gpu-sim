#!/usr/bin/env python3
"""Guard: close_audit.json is live bead-DB state, not code.

experiments/gpu-sim-u2o.15/close_audit.json is regenerated on `main` as its own
close-audit commit (see experiments/gpu-sim-u2o.15/README.md). A branch that
modifies it (or deletes it) is carrying branch noise into review: two branches
from the same base shipped different values (365/348 vs 370/355) and every
rebase produced a spurious diff.

So a branch may NOT change this path relative to the `main`/`master` tip.
Regenerate it on main only.

Exit 0 if unchanged, 1 otherwise (including when no ref can be resolved).
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATH = "experiments/gpu-sim-u2o.15/close_audit.json"

def run(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                         capture_output=True, text=True)

def main() -> int:
    # Resolve the integration branch tip (not merge-base): a stale base must
    # never produce a pass (gpu-sim-xv3c).
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from resolve_trunk import resolve_trunk
    main_ref = resolve_trunk(str(REPO))  # single source of truth (gpu-sim-e7yw)
    if main_ref is None:
        print("INVALID: check_close_audit_unchanged: no main/master ref "
              "could be resolved; cannot verify close_audit.json is unchanged.")
        print("Convention: close_audit.json is regenerated on main as its own commit.")
        print("See experiments/gpu-sim-u2o.15/README.md.")
        return 1

    # Blob on main (the authoritative version).
    main_blob = run("show", f"{main_ref}:{PATH}")
    main_exists = main_blob.returncode == 0

    # Compare the COMMITTED blob (HEAD), never the working tree.
    head_blob = run("show", f"HEAD:{PATH}")
    head_exists = head_blob.returncode == 0

    # Absence on a branch that postdates main's copy is a deletion, not a pass.
    if not main_exists:
        # main doesn't have it; cannot verify against authoritative state.
        print(f"UNVERIFIABLE: {PATH} absent on main; cannot verify unchanged.")
        return 1

    # main has the file: HEAD must have the same blob.
    if not head_exists:
        print(f"INVALID: {PATH} deleted on this branch; restore it unchanged.")
        return 1
    if main_blob.stdout != head_blob.stdout:
        print(f"INVALID: {PATH} differs from its base blob.")
        return 1
    print("check_close_audit_unchanged: OK (unchanged vs main)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
