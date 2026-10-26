"""Closed is not merged: find closed beads whose branch never reached the integration branch (gpu-sim-wzzx).

On 2026-10-26 gpu-sim-hz3u was reviewed, gate-verified and CLOSED while its branch sat 3 commits ahead of
master. Nothing surfaced it: `bd list`, the gate and the bead comments all looked correct, because closing and
merging are separate acts and only one of them is tracked. The three follow-on traps found the same day are the
same class -- state that exists in git but not in beads:

  * a closed bead whose branch is unmerged (this script),
  * a bead claim that expires while work sits uncommitted in a worktree,
  * `lab review-request` resolving origin/HEAD to a branch that is not the integration branch.

Branch names carry the bead id (`lab/gpu-sim-<id>`), so this is mechanical.

Usage:
  python3 scripts/check_branch_ancestry.py            # report; exit 0 even when it finds gaps (advisory)
  python3 scripts/check_branch_ancestry.py --strict   # exit 1 when a closed bead's branch is unmerged

Wire into the gate only after the branch you are landing does not itself trip it.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

BRANCH_RE = re.compile(r"^lab/gpu-sim-([0-9a-z]+)$")


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True, check=False).stdout


def parse_closed_ids(raw: str) -> set[str]:
    """bd emits FULL ids ("gpu-sim-hz3u"); branch names carry only the SUFFIX ("hz3u").

    Both bugs this function has had were normalisation bugs, and both were invisible because the checker
    reported "OK":
      1. comparing "gpu-sim-hz3u" against "hz3u" -- never matched, zero offenders, real gap missed;
      2. stripping the prefix from a capture group that had ALREADY excluded it -- every id became "", the set
         collapsed to one element, same silent OK.
    Keeping the parse pure makes it testable against a real bd payload, which is the only version of this test
    that could have caught either.
    """
    return set(re.findall(r'"id"\s*:\s*"gpu-sim-([0-9a-z]+)"', raw))


def closed_beads() -> set[str]:
    """Bead ids in a terminal state. Absent bd -> empty set (advisory tool, must not explode)."""
    raw = subprocess.run(("bd", "list", "--status", "closed", "--json"),
                         capture_output=True, text=True, check=False).stdout
    return parse_closed_ids(raw)


def lab_branches() -> dict[str, str]:
    """branch -> bead id, for branches named after a bead."""
    out = {}
    for line in git("for-each-ref", "--format=%(refname:short)", "refs/heads/lab").splitlines():
        m = BRANCH_RE.match(line.strip())
        if m:
            out[line.strip()] = m.group(1)
    return out


def is_contained(branch: str, base: str = "master") -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", branch, base),
                          capture_output=True).returncode == 0


def offenders(branches: dict[str, str], closed: set[str], contained) -> list[tuple[str, str]]:
    """Pure core (testable without git or bd): closed bead + branch not contained in base."""
    return sorted((b, bid) for b, bid in branches.items() if bid in closed and not contained(b))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 on any closed-but-unmerged branch")
    ap.add_argument("--base", default="master")
    a = ap.parse_args(argv)

    bad = offenders(lab_branches(), closed_beads(), lambda b: is_contained(b, a.base))
    if not bad:
        print(f"branch ancestry OK: every closed bead's lab/ branch is contained in {a.base}")
        return 0
    print(f"CLOSED BUT NOT MERGED into {a.base}: {len(bad)}")
    for branch, bid in bad:
        ahead = git("rev-list", "--count", f"{a.base}..{branch}").strip()
        print(f"  {branch}  ({bid}, {ahead} commits ahead)")
    print("\nA closed bead is not a merged bead. Merge it, or comment on the bead saying why not.")
    return 1 if a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
