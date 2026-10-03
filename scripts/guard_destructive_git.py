#!/usr/bin/env python3
"""gpu-sim-edu.1.1: guard destructive git against an inherited target.

INCIDENT (2026-10-03, recovered byte-exactly). `git checkout -- .` was run in
/home/gabe/Documents/gpu-sim intending to clean a WORKTREE
(.lab/wt/gpu-sim-u2o.6.6.2.1). It reverted long-standing uncommitted
modifications to AGENTS.md and CLAUDE.md in the MAIN tree instead. Recovery came
from `git fsck --unreachable`, which surfaced stash-shaped commit f73d442
("WIP on master: 2b40b19") holding exactly those two files.

THE DEFECT IS NOT RECOVERY, IT IS TARGET SELECTION. A destructive git command
that takes its target from the CWD cannot tell the intended tree from any other
tree on the machine. This repo runs ~30 worktrees plus the main tree, all one
repository, all reachable by `cd`, all with a plausible CLAUDE.md and AGENTS.md
at the root. There is no visual difference between them and no error when the
wrong one is chosen.

I then MADE THIS MISTAKE MYSELF while building this guard. I created the
intended worktree path with the file-writing tool rather than `git worktree
add`, so the directory existed but was not a worktree; every `git` command run
there walked up to the MAIN repository. The commit never happened, so no damage
was done -- but the bead exists because the failure is real and I reproduced it
inside the session that wrote the guard for it.

This is the fourth time this session that an operation was correct but its
TARGET was inherited from ambient context -- bd create --parent inheriting
labels, the dispatch offering validated-but-unmerged beads, a join key assumed
injective, and this.

TWO GUARDS PLUS A FREE SAFETY NET, all cheap:

  (a) NO CWD-SCOPED DESTRUCTIVE GIT in the main tree. `checkout -- .`,
      `restore .`, `reset --hard`, `clean -fd` are refused unless the command
      names explicit paths or targets a worktree with `git -C`.

  (b) A PRE-DESTRUCTIVE-OP GUARD that FAILS CLOSED on a dirty main tree. If the
      main tree has modified tracked files and no explicit pathspec was given,
      it refuses. A guard that cannot fail is not a guard.

  (c) A RECOVERY POINT. `git stash create` does NOT touch the working tree
      (unlike `git stash push`); it snapshots the dirty state as an unreachable
      commit and prints the hash. That is precisely what saved this incident, so
      it is taken automatically rather than left to memory.

Exit codes: 0 proceed, 2 refused (the command did NOT run).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DESTRUCTIVE = {"checkout", "restore", "reset", "clean", "stash"}

# Options that consume the NEXT token as their value. Without skipping these,
# `git -C .lab/wt/<bead> checkout -- .` parses the worktree PATH as the
# subcommand, and the guard then reports "not a destructive subcommand" for the
# exact command form the rules REQUIRE. A guard blind to the safe spelling of
# the dangerous one teaches that `git -C` is not worth writing.
VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace",
              "--exec-path", "--config-env"}

# `--opt=value` spellings that name a target just as explicitly as `-C <p>`.
EQUALS_TARGET_OPTS = ("--git-dir=", "--work-tree=", "--namespace=")

# A pathspec is only EXPLICIT if it names a path. These are CWD-scoped, which is
# the entire hazard: my first version accepted `git checkout -- .` as "explicit"
# precisely because it has a pathspec, so the incident command sailed straight
# through. It looks named. It is not.
CWD_SCOPED_PATHSPECS = {".", "./", "*", "**", "..."}


def parse(argv: list[str]):
    """Split a git command into (subcommand, flags, pathspecs).

    Deliberately a guard, not a git parser, and it says so. It raises rather
    than guessing: a guard that silently misparses is worse than none.
    """
    if not argv or argv[0] != "git":
        raise ValueError("command must start with 'git'")
    rest = argv[1:]
    sub, tail = None, []
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok in VALUE_OPTS:
            i += 2
            continue
        if not tok.startswith("-"):
            sub, tail = tok, rest[i + 1:]
            break
        i += 1
    if sub is None:
        raise ValueError(f"no subcommand found in {shlex.join(argv)}")
    flags = [t for t in tail if t.startswith("-")]
    if "--" in tail:
        paths = tail[tail.index("--") + 1:]
    elif sub == "clean":
        paths = [t for t in tail if not t.startswith("-")]
    else:
        paths = []
    return sub, flags, paths


def is_scoped(args: list[str]) -> bool:
    """Does the command name its target explicitly? Ambient CWD is not a target."""
    try:
        sub, _flags, paths = parse(args)
    except ValueError:
        return False                      # unparseable: err toward refusing
    if "-C" in args[1:]:
        return True
    if any(a.startswith(EQUALS_TARGET_OPTS) for a in args[1:]):
        return True
    if sub not in ("checkout", "restore") or not paths:
        return False
    return all(p not in CWD_SCOPED_PATHSPECS and not p.startswith("*")
               for p in paths)


def main_tree_dirty(repo: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                          "--untracked-files=no"],
                         capture_output=True, text=True)
    modified = []
    for line in out.stdout.splitlines():
        if line.startswith("??"):
            continue
        if line[:2].strip():
            modified.append(line[3:].strip())
    return modified


def create_recovery_point(repo: Path) -> str | None:
    """`git stash create` -- does NOT touch the working tree.

    This is what made the 2026-10-03 incident recoverable, so it is taken
    automatically before a destructive command rather than remembered.
    """
    p = subprocess.run(["git", "-C", str(repo), "stash", "create"],
                       capture_output=True, text=True)
    return p.stdout.strip() or None


def guard(argv: list[str], repo: Path, run: bool = True) -> dict:
    sub, _flags, paths = parse(argv)
    scoped = is_scoped(argv)
    dirty = main_tree_dirty(repo)
    result = {"command": shlex.join(argv), "subcommand": sub,
              "destructive": sub in DESTRUCTIVE, "explicitly_scoped": scoped,
              "main_tree_modified_tracked": dirty, "paths": paths,
              "recovery_commit": None, "ran": False, "refused": False,
              "reason": None}

    if sub not in DESTRUCTIVE:
        result["reason"] = "not a destructive git subcommand; nothing to guard"
        if run:
            result["ran"] = True
            result["rc"] = subprocess.run(argv, cwd=str(repo)).returncode
        return result

    if scoped:
        result["reason"] = "target named explicitly (git -C, --git-dir=, or a real pathspec)"
    elif dirty:
        result["refused"] = True
        result["reason"] = (
            f"REFUSED. The main tree has {len(dirty)} modified tracked "
            f"file(s) ({', '.join(dirty[:5])}) and this command takes its target "
            f"from the CWD. Ambient CWD is not a target: name the files, or use "
            f"`git -C <worktree>`. If this discard is genuinely intended, "
            f"re-run with an explicit pathspec.")
        return result
    else:
        result["reason"] = "main tree is clean; a CWD-scoped discard is safe"

    if dirty or not scoped:
        result["recovery_commit"] = create_recovery_point(repo)

    if run:
        result["ran"] = True
        result["rc"] = subprocess.run(argv, cwd=str(repo)).returncode
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--check-only", action="store_true",
                    help="report the verdict without running the command")
    ap.add_argument("git_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    argv = list(a.git_args)
    # argparse.REMAINDER keeps the `--` separator in some shapes, and git
    # rejects a bare `--` as a git option. Drop LEADING separators only; the one
    # that separates pathspecs from the subcommand is load-bearing.
    while argv and argv[0] == "--":
        argv.pop(0)
    # Accept the full command or just the arguments after `git`.
    if argv and argv[0] != "git":
        argv = ["git", *argv]
    if len(argv) < 2:
        ap.error("give a git command, e.g. -- checkout -- .")
    try:
        res = guard(argv, Path(a.repo), run=not a.check_only)
    except ValueError as exc:
        print(f"GUARD CANNOT PARSE THE COMMAND ({exc}); refusing to run it. A "
              f"guard that cannot read what it is guarding is not a guard.",
              file=sys.stderr)
        return 2
    print(res["reason"], file=sys.stderr)
    if res["refused"]:
        return 2
    if res["recovery_commit"]:
        print(f"recovery point: {res['recovery_commit']} (unreachable; "
              f"`git stash create` does not touch the tree)", file=sys.stderr)
    if res.get("ran"):
        return res.get("rc", 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())