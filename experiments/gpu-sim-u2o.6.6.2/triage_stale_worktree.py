#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.2.1: triage the 6.6.2-r2 worktree so nobody inherits it blind.

Acceptance items 2 and 3. The worktree that hosted the v8 run has 5 modified
tracked files and 47 untracked paths. Both are recorded here as a DECISION with
evidence, not left for the next session to guess at.

This script only INSPECTS and REPORTS. It never edits the worktree: deleting 34
master-owned copies out from under a ten-commit-stale fork is exactly the class
of move that destroyed the v8 launch artifact in the first place, and this bead
exists because someone did it by accident.

Usage: python3 experiments/gpu-sim-u2o.6.6.2/triage_stale_worktree.py
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def _main_repo() -> Path:
    """The MAIN worktree. `.lab/wt/...` lives there, not under this worktree.

    Resolving the repo root as parents[2] of this file points at the LINKED
    worktree, which has no `.lab/wt/` of its own -- so the stale worktree reads
    as absent and the triage silently reports nothing. That is the THIRD time
    in this session a repo-root resolution has been wrong in the direction that
    makes a check report NOTHING rather than fail; it is resolved from git's own
    common dir instead.
    """
    out = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]),
         "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip()).parent
    return Path(__file__).resolve().parents[2]


REPO = _main_repo()
STALE = REPO / ".lab/wt/gpu-sim-u2o.6.6.2-r2"
OUT = Path(__file__).resolve().parent / "worktree_triage.json"


def git(*args, cwd=REPO):
    p = subprocess.run(["git", "-C", str(cwd), *args],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def main() -> int:
    if not STALE.exists():
        print(f"{STALE} does not exist; nothing to triage", flush=True)
        return 0

    porcelain = git("status", "--porcelain", cwd=STALE).splitlines()
    modified, untracked = [], []
    for line in porcelain:
        if line.startswith("??"):
            untracked.append(line[3:].strip())
        elif line[:2].strip():
            modified.append({"path": line[3:].strip(), "status": line[:2].strip()})

    master_tracked = set(git("ls-tree", "-r", "--name-only", "master").splitlines())

    master_owned, genuinely_new, identical, differing = [], [], [], []
    for rel in untracked:
        if rel not in master_tracked:
            genuinely_new.append(rel)
            continue
        master_owned.append(rel)
        head = git("show", f"master:{rel}")
        disk = (STALE / rel).read_text(errors="replace") \
            if (STALE / rel).exists() else ""
        (identical if head == disk else differing).append(rel)

    # Acceptance item 3's assertion: after triage, no untracked path in the
    # worktree may also be a tracked path in master. Reported, not asserted by
    # deletion -- the count is the evidence a human acts on.
    collisions = sorted(set(untracked) & master_tracked)

    rec = {
        "bead": "gpu-sim-u2o.6.6.2.1",
        "kind": "STALE WORKTREE TRIAGE -- inspects and reports, never edits",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "worktree": str(STALE),
        "item_2_modified_tracked_files": {
            "count": len(modified),
            "files": modified,
            "disposition": ("UNRESOLVED BY THIS SCRIPT, ON PURPOSE. The PI ruled "
                            "on gpu-sim-u2o.6.6.2 that the v8 family files stay "
                            "exactly as found and are dispositioned under this "
                            "bead. Committing them requires regenerating "
                            "metrics.json from a MIXED set (v8 memory/reduce/"
                            "gemm beside v6 launch), which is the very "
                            "mixed-provenance artifact this bead exists to "
                            "prevent. Resetting them discards three finished GPU "
                            "jobs. Both need the PI's word, so neither is done "
                            "here."),
        },
        "item_3_untracked_paths": {
            "total": len(untracked),
            "master_owned_copies": len(master_owned),
            "genuinely_new_to_this_branch": len(genuinely_new),
            "genuinely_new_paths": sorted(genuinely_new),
            "identical_to_master": len(identical),
            "differing_from_master": sorted(differing),
            "collisions_with_master_tracked": collisions,
            "CORRECTION": (
                "The bead's design -- and my own earlier report on "
                "gpu-sim-u2o.6.6.2 -- says TWO of the master-owned copies "
                "differ (harness.md, preflight_validator.py). MEASURED, it is "
                f"{len(differing)} of {len(master_owned)}, all under epic_u2o/. "
                "Those two files really are in the set; they are not the only "
                "ones. I had confirmed two examples and reported the number of "
                "examples as the total, which understates the hazard by an "
                "order of magnitude -- twenty stale copies of master-owned code "
                "is a different problem from two."),
            "assertion": ("no untracked path in the worktree may also be tracked "
                          "in master; the collisions list above is the evidence, "
                          "and it is currently NON-EMPTY"),
            "why_it_matters": ("a bulk `git add` in that worktree would add 34 "
                               "files master already owns, and any branch switch "
                               "there would overwrite them -- the rebase earlier "
                               "refused for exactly this reason."),
        },
        "item_4_mechanism": {
            "problem": ("queue jobs wrote artifacts into a live worktree while "
                        "git operations were available in the same tree; a "
                        "tracked-path restore there replaced a finished job's "
                        "output 20 seconds after it was written"),
            "what_this_bead_adds": ("measure_family.py stamps every artifact with "
                                    "its writing job id and a content hash, and "
                                    "verify_artifact_provenance.py compares "
                                    "on-disk provenance against the queue's own "
                                    "record -- so the restore is detectable "
                                    "rather than silent. That is detection, not "
                                    "prevention; prevention is item 4's open "
                                    "question of where jobs should write."),
            "open": ("item 4 also asks that a worktree be QUIESCED before a git "
                     "operation that can touch tracked paths. That is enforced "
                     "for destructive git by scripts/guard_destructive_git.py "
                     "(gpu-sim-edu.1.1 / edu.2), which now also refuses anything "
                     "that would delete the lab runtime. It is NOT yet enforced "
                     "for non-destructive tracked-path writes, and this script "
                     "does not pretend otherwise."),
        },
    }
    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({
        "modified_tracked": len(modified),
        "untracked_total": len(untracked),
        "master_owned_copies": len(master_owned),
        "identical_to_master": len(identical),
        "differing_from_master": len(differing),
        "genuinely_new": len(genuinely_new),
        "collisions_with_master": len(collisions),
    }, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())