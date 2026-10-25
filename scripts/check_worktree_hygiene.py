#!/usr/bin/env python3
"""Report worktree sprawl -- advisory, never fails the gate (gpu-sim-gtree).

Context: this repo accumulated 88 git worktrees (81 under .lab/wt/, six in /tmp). That is not
just disk: every worktree carries its own copy of scripts/gate.sh, so a background session
running an old copy reports a RED verdict with no way to tell which tree it came from. Two
phantom REDs in one day came from that. It also filled the 16 GB /tmp tmpfs from worktrees
living outside the repo.

This is advisory because a large number is sometimes CORRECT (a fan-out of parallel
experiments), and removing worktrees is never something a gate should do on its own. What it
does instead is classify them so pruning is mechanical:

  contained_clean  branch fully in master, tree clean  -> holds nothing master lacks, removable
  contained_dirty  branch in master, uncommitted edits -> someone's work in progress, KEEP
  diverged_*       commits not in master               -> KEEP

Exit status is always 0 unless the repo itself is unreadable.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARN_AT = 40          # main tree + a healthy fan-out fits comfortably under this


def git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def main() -> int:
    out = git("worktree", "list", "--porcelain").stdout
    blocks = [b for b in out.strip().split("\n\n") if b.strip()]
    here = os.path.realpath(REPO)
    cats = {}
    removable, outside = [], []
    for b in blocks:
        fields = {}
        for line in b.splitlines():
            key, _, value = line.partition(" ")
            fields[key] = value
        path = fields.get("worktree", "")
        branch = (fields.get("branch") or "").replace("refs/heads/", "")
        if os.path.realpath(path) == here:
            continue
        if not os.path.isdir(path):
            cats["missing"] = cats.get("missing", 0) + 1
            continue
        if path.startswith("/tmp"):
            outside.append(path)
        dirty = bool(git("-C", path, "status", "--porcelain").stdout.strip())
        contained = bool(branch) and git("merge-base", "--is-ancestor", branch, "master",
                                        cwd=path).returncode == 0
        key = ("contained_" if contained else "diverged_") + ("dirty" if dirty else "clean")
        cats[key] = cats.get(key, 0) + 1
        if key == "contained_clean":
            removable.append(path)

    total = len(blocks) - 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))
    print(f"worktrees: {total} (main tree excluded) | {summary or 'none'}")
    if removable:
        print(f"  removable (branch in master + clean): {len(removable)}")
    if outside:
        print(f"  OUTSIDE the repo tree (eats /tmp quota): {len(outside)}")
    if total >= WARN_AT:
        print(f"  WARNING: {total} worktrees >= {WARN_AT} -- each carries a stale copy of "
              f"scripts/gate.sh, and stale copies report RED verdicts that name no tree. "
              f"Prune the removable ones (see this script's docstring) and keep /tmp clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
