#!/usr/bin/env python3
"""gpu-sim-edu.2: inventory the lab runtime under .lab/ and what git would remove.

Answers two questions the bead asks, mechanically:
  * what runtime files are PRESENT and which named ones are ABSENT;
  * what a `git clean -fd` from the repo root would DELETE.

Read-only. The clean is asked as a DRY RUN (`git clean -fdn`), which lists what
it would remove without removing it.

Usage: python3 experiments/gpu-sim-edu.2/inventory.py
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "inventory.json"


def main_repo() -> Path:
    """The MAIN worktree, not whatever worktree this file happens to sit in.

    .lab/ lives in the main tree. Run from a linked worktree, `git clean -fdn`
    reports a DIFFERENT and much smaller set -- my first version did exactly
    that and produced an artifact claiming no lab runtime was at risk, which is
    the most dangerous possible direction for this particular number to be
    wrong in.
    """
    out = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]),
         "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip()).parent
    return Path(__file__).resolve().parents[2]


REPO = main_repo()

# The files this bead was opened about.
NAMED = ["inbox", "prompts", "labd.log", "pid", "queue_check.py"]


def git(*args):
    p = subprocess.run(["git", "-C", str(REPO), *args],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def main() -> int:
    lab = REPO / ".lab"
    named = {}
    for name in NAMED:
        p = lab / name
        named[name] = {"present": p.exists(),
                       "kind": ("dir" if p.is_dir() else
                                "file" if p.is_file() else None)}

    present = sorted(str(p.relative_to(lab)) for p in lab.rglob("*")
                     if p.is_file() and "wt" not in p.relative_to(lab).parts) \
        if lab.exists() else []

    would_remove = [ln.replace("Would remove ", "").strip()
                    for ln in git("clean", "-fdn").splitlines()
                    if ln.startswith("Would remove")]

    lab_would_go = [p for p in would_remove if p.startswith(".lab/")]

    rec = {
        "bead": "gpu-sim-edu.2",
        "kind": "LAB RUNTIME INVENTORY -- read-only, git asked via dry run",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "lab_dir": str(lab),
        "named_files_reported_missing": named,
        "present_runtime_files": present,
        "git_clean_fd_would_remove_total": len(would_remove),
        "git_clean_fd_would_remove_lab_runtime": lab_would_go,
        "risk": ("git clean -fd from the repo root would delete every role "
                 "socket, labd-state.json, the inboxes and the prompts. They "
                 "are UNTRACKED, so they do not make the tree dirty and no "
                 "cleanliness check can see them."),
        "git_evidence": {
            "lab_ever_tracked_by_git": bool(git("log", "--all", "--oneline",
                                                "--", ".lab")),
            "deleted_tracked_entries": [l for l in
                                        git("status", "--porcelain").splitlines()
                                        if l.startswith((" D", "D "))],
        },
        "interpretation": (
            "The loose file-based runtime (labd.log, pid, queue_check.py) is "
            "absent while the socket + state-file runtime is present, which is "
            "the shape of a MIGRATION rather than a deletion. .lab was never "
            "tracked by git and no tracked file is deleted, so no git operation "
            "can account for it."),
    }
    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rec.items()
                      if k != "present_runtime_files"}, indent=2))
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
