#!/usr/bin/env python3
"""Guard: close_audit.json is live bead-DB state, not code.

experiments/gpu-sim-u2o.15/close_audit.json is regenerated on `main` as its own
close-audit commit (see experiments/gpu-sim-u2o.15/README.md). A branch that
modifies it (or deletes it) is carrying branch noise into review: two branches
from the same base shipped different values (365/348 vs 370/355) and every
rebase produced a spurious diff.

So a branch may NOT change this path relative to its merge-base with `main`/`master`.
Regenerate it on main only.

Exit 0 if unchanged, 1 otherwise (including when no base can be resolved).
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
    # Resolve the integration base.
    base = None
    for ref in ("master", "main"):
        mb = run("merge-base", "HEAD", ref)
        if mb.returncode == 0 and mb.stdout.strip():
            base = mb.stdout.strip()
            break
    if base is None:
        print("INVALID: check_close_audit_unchanged: no merge-base with master/main "
              "could be resolved; cannot verify close_audit.json is unchanged. "
              "Run inside a checkout that has the integration branch (master/main).")
        print("Convention: close_audit.json is regenerated on main as its own commit.")
        print("See experiments/gpu-sim-u2o.15/README.md.")
        return 1

    # Blob at base (may be absent).
    base_blob = run("show", f"{base}:{PATH}")
    base_exists = base_blob.returncode == 0

    # Compare the COMMITTED blob (HEAD), never the working tree.
    head_blob = run("show", f"HEAD:{PATH}")
    head_exists = head_blob.returncode == 0

    if base_exists == head_exists:
        if base_exists:
            if base_blob.stdout == head_blob.stdout:
                print("check_close_audit_unchanged: OK (unchanged vs base)")
                return 0
            print(f"INVALID: {PATH} differs from its base blob ({base[:7]}).")
        else:
            print("check_close_audit_unchanged: OK (absent on both sides)")
            return 0
    # one side missing
    if base_exists and not head_exists:
        print(f"INVALID: {PATH} deleted on this branch; restore it unchanged.")
    elif (not base_exists) and head_exists:
        print(f"INVALID: {PATH} added on this branch; it is main-only state.")
    print("Convention: close_audit.json is regenerated on main as its own commit.")
    print("See experiments/gpu-sim-u2o.15/README.md.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
