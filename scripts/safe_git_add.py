#!/usr/bin/env python3
"""gpu-sim-k9w: refuse a bulk git add that would take files master already owns.

THE HAZARD, OBSERVED. This repo runs ~30 worktrees under .lab/wt/ plus the main
tree. Several worktrees hold UNTRACKED files that are ALREADY TRACKED ON
MASTER: stale copies left behind by a branch that forked from an old base. A
plain `git add -A` in such a worktree stages all of them, and committing adds
dozens of master-owned files to a branch that has no business owning them.

This happened to me on gpu-sim-u2o.6.10: 49 files and 7,966 insertions
committed where 2 were mine. It was caught only by reading the commit stat.

WHY THIS IS NOT ALREADY PREVENTED. gpu-sim-edu.2 guards the DELETE direction --
`git clean -fd` would remove the whole lab runtime. gpu-sim-u2o.6.6.2.1 TRIAGES
this one, listing the collisions in an artifact. Both document. Neither enforces,
and the ADD direction had no guard at all. A stale worktree's untracked files
look exactly like work in progress, which is what `git add -A` is for.

Usage:
    python3 scripts/safe_git_add.py            # check the current index/worktree
    python3 scripts/safe_git_add.py --explain  # list what a bulk add would take
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Where this SCRIPT lives, which is not necessarily where the caller is.
SCRIPT_REPO = Path(__file__).resolve().parents[1]


def current_repo() -> Path:
    """The worktree the CALLER is standing in, not the one holding this script.

    The reviewer's point, and it is the difference between a guard that works
    and one that appears to. Resolving the repo from __file__ means running the
    script from inside a stale worktree checks the wrong tree and reports
    "nothing untracked" -- which is exactly the false all-clear that makes this
    class of mistake invisible. `git rev-parse --show-toplevel` asks git where
    the CWD actually is.
    """
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return Path(r.stdout.strip())
    return SCRIPT_REPO
# Files that are SUPPOSED to be shared rather than owned by a worktree.
SHARED_OK = (".lab/",)


def git(*args: str, cwd: Path = None) -> str:
    cwd = cwd or current_repo()
    p = subprocess.run(["git", "-C", str(cwd), *args],
                       capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else ""


def untracked_paths(repo: Path = None) -> list[str]:
    out = git("ls-files", "--others", "--exclude-standard", cwd=repo)
    return [p for p in out.splitlines() if p.strip()]


def master_tracked(repo: Path = None) -> set[str]:
    return {p for p in git("ls-tree", "-r", "--name-only", "master", cwd=repo).splitlines()
            if p.strip()}


def collisions(repo: Path = None) -> list[tuple[str, str]]:
    """(path, status) for every untracked path master ALREADY tracks.

    Status is 'identical' when the worktree copy matches master byte for byte --
    those are the stale duplicates, indistinguishable from a deliberate copy -- and
    'DIFFERS' when it does not, which is worse: the branch would overwrite a
    master file with different content.
    """
    master = master_tracked(repo)
    out = []
    for path in untracked_paths(repo):
        if path not in master or path.startswith(SHARED_OK):
            continue
        head = subprocess.run(["git", "-C", str(repo), "show", f"master:{path}"],
                             capture_output=True)
        disk = (repo / path).read_bytes() if (repo / path).exists() else b""
        status = "identical" if head.stdout == disk else "DIFFERS"
        out.append((path, status))
    return sorted(out)


def check(repo: Path = None) -> int:
    repo = repo or current_repo()
    untracked = untracked_paths(repo)
    hits = collisions(repo)

    if not untracked:
        print("nothing untracked; a bulk add would take nothing.")
        return 0

    print(f"untracked paths here: {len(untracked)}")
    if not hits:
        print("bulk add is safe: none of them are tracked on master.")
        return 0

    print(f"\nREFUSING a bulk add. {len(hits)} of these are ALREADY TRACKED ON "
          f"MASTER, so `git add -A` would add master-owned files to this "
          f"branch:")
    for path, status in hits[:25]:
        print(f"  [{status}] {path}")
    if len(hits) > 25:
        print(f"  ... and {len(hits) - 25} more")
    identical = sum(1 for _, s in hits if s == "identical")
    differs = len(hits) - identical
    print(f"\n  {identical} identical to master (stale duplicates)")
    print(f"  {differs} DIFFER from master (an add would overwrite these)")
    print("\nStage them explicitly if they are genuinely yours:")
    print("  git add <path> <path> ...     # and check the commit stat afterwards")
    print("Or, if these are stale copies of master, delete them from the worktree")
    print("rather than committing them onto a branch that does not own them.")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="With no --repo, the tree checked is the one you are standing "
               "in (`git rev-parse --show-toplevel`), so running this inside a "
               "stale worktree checks THAT worktree.")
    ap.add_argument("--repo", default=None,
                    help="tree to check (default: the current worktree)")
    ap.add_argument("--explain", action="store_true",
                    help="print what a bulk add would take")
    a = ap.parse_args()
    repo = Path(a.repo).resolve() if a.repo else current_repo()
    return check(repo)


if __name__ == "__main__":
    sys.exit(main())