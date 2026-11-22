#!/usr/bin/env python3
"""Reap leaked pytest/gate scratch under a tree's .lab/tmp (gpu-sim-c3hd).

Why this exists: on 2026-11-22 every atomic write on this box failed

    ENOSPC: no space left on device, rename ...

while `df -h /home` reported 70 GiB free -- btrfs METADATA exhaustion, driven by
9.1 GiB of leaked scratch under .lab/tmp (gpu-sim-cywp). scripts/check_fs_headroom.py
now fails the gate before that point, and this is the other half: reclaim the
pile instead of only diagnosing it.

DELETING A LIVE TEST'S SCRATCH CORRUPTS THAT TEST, so the classifier is the
whole tool. A directory is reaped only when ALL of these hold:

  1. it is a real directory under --root (symlinks are never followed), and its
     basename matches one of --pattern (default: lab-suite-*, lab-gate-*,
     pytest-*, tmp* -- the names actually measured in .lab/tmp/implementer:
     1.4 GiB of lab-suite-*, 1.8 GiB of tmp*, 385 lab-gate-*, 366 r9j-scope-*);
  2. its age (mtime) is at least --older-than-hours (default 48), UNLESS it is
     registered and its registrant is provably gone;
  3. NO live process has it (or anything below it) as its working directory, and
     no live process names it on its command line;
  4. it is not inside a checked-out tree: any directory containing .git is
     pruned, because .lab/tmp also holds reviewers' disposable clones.

Condition 3 is not theoretical: while this was being written, seven live
processes on this box had a cwd under a .lab/tmp/implementer directory (two other
repositories' trees). A reaper keyed on age alone would have deleted them.

Dry run is the default: `--apply` is required to delete anything. The gate calls
it with --apply, because the alternative is a box that cannot rename.

Registry: `record <dir> --pid <pid>` appends {path, pid, starttime} to
<root>/scratch-labs.jsonl under an exclusive lock, so a harness that creates a
scratch tree can register its owner. Liveness uses pid + /proc/<pid>/stat
starttime, not pid alone: pids are reused, and a stale registry entry that
looks alive would keep a leak forever (scripts/appendonly_worktrees.py uses the
same rule for worktrees).
"""
import argparse
import errno
import fnmatch
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / ".lab" / "tmp"
DEFAULT_PATTERNS = ("lab-suite-*", "lab-gate-*", "pytest-*", "tmp*", "r9j-scope-*")
REGISTRY_NAME = "scratch-labs.jsonl"
GiB = 1024 ** 3


# ----------------------------------------------------------------- registry ---
def registry_path(root):
    return Path(root) / REGISTRY_NAME


def starttime_of(pid):
    """Field 22 of /proc/<pid>/stat -- identical to appendonly_worktrees.py."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    r = data.rfind(")")
    if r == -1:
        return None
    try:
        return int(data[r + 1:].split()[19])
    except (ValueError, IndexError):
        return None


def read_records(root):
    p = registry_path(root)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def append_record(root, path, pid):
    p = registry_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"path": str(path), "pid": pid, "starttime": starttime_of(pid),
           "at": time.time()}
    with open(p, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return rec


# --------------------------------------------------------------- liveness ---
def live_owners(root):
    """One pass over /proc: cwds, and the cmdlines that mention `root` at all.

    Both halves matter. The cwd set catches a process RUNNING inside the tree;
    the cmdline set catches `pytest --basetemp <dir>` and friends, where the
    directory is an argument rather than a cwd. The cmdline half is prefiltered
    by the root string in this single pass: doing it per candidate cost 3744
    candidates x ~1000 processes of /proc reads (34 s) in the first version of
    this file, which is too slow for a gate preflight.
    """
    cwds, cmdlines = set(), []
    needle = str(root).encode()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            # a zombie's cwd link reads as "... (deleted)" or fails; both are
            # fine -- a zombie owns nothing.
            cwds.add(os.readlink(f"/proc/{entry}/cwd"))
        except OSError:
            pass
        try:
            cmd = Path(f"/proc/{entry}/cmdline").read_bytes()
        except OSError:
            continue
        if needle in cmd:
            cmdlines.append(cmd)
    return cwds, cmdlines


def owned_by_live_process(path, owners):
    cwds, cmdlines = owners
    s = str(path)
    prefix = s.rstrip("/") + "/"
    for cwd in cwds:
        if cwd == s or cwd.startswith(prefix):
            return True
    needle = s.encode()
    return any(needle in cmd for cmd in cmdlines)


def registered_live(records, path):
    """True when a record for this path names a live pid with a matching start."""
    s = str(path)
    for rec in records:
        if rec.get("path") != s:
            continue
        pid = rec.get("pid")
        if not isinstance(pid, int):
            continue
        cur = starttime_of(pid)
        if cur is None:
            continue
        if rec.get("starttime") is None or rec.get("starttime") == cur:
            return True
    return False


def registered_but_gone(records, path):
    s = str(path)
    for rec in records:
        if rec.get("path") != s:
            continue
        pid = rec.get("pid")
        if not isinstance(pid, int):
            continue
        cur = starttime_of(pid)
        if cur is None or (rec.get("starttime") is not None
                           and rec.get("starttime") != cur):
            return True
    return False


# ------------------------------------------------------------------ reaper ---
def dir_bytes(path):
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def _walk_matching(parent, patterns):
    """Yield dirs whose BASENAME matches, at any depth.

    Recursion is the point: the pile that triggered this bead lived at
    .lab/tmp/implementer/lab-suite-*, not at .lab/tmp/lab-suite-*. A matching dir
    is yielded and NOT descended into -- either it is deleted whole or it belongs
    to a live owner, and in both cases its contents are not separately decidable.
    Symlinks are never followed, and are never yielded.
    """
    try:
        entries = sorted(os.scandir(parent), key=lambda e: e.name)
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        path = Path(entry.path)
        # A checked-out tree is never scratch, however it is named: .lab/tmp
        # holds reviewers' disposable CLONES (e.g. .../vmi5v/experiments/...),
        # and deleting inside one would destroy someone's measurement mid-review.
        if (path / ".git").exists():
            continue
        if any(fnmatch.fnmatch(entry.name, pat) for pat in patterns):
            yield path
            continue
        yield from _walk_matching(path, patterns)


def plan(root, patterns, older_than_hours):
    root = Path(root).resolve()
    records = read_records(root)
    owners = live_owners(root)
    now = time.time()
    floor = older_than_hours * 3600.0
    candidates, kept = [], []
    for path in _walk_matching(root, patterns):
        try:
            age = now - path.lstat().st_mtime
        except OSError:
            continue
        if registered_live(records, path):
            kept.append((path, "registered owner is alive"))
            continue
        if owned_by_live_process(path, owners):
            kept.append((path, "a live process has it open (cwd or cmdline)"))
            continue
        if registered_but_gone(records, path):
            candidates.append((path, "owner died (pid/starttime)"))
            continue
        if age < floor:
            kept.append((path, f"younger than {older_than_hours}h ({age/3600:.1f}h)"))
            continue
        candidates.append((path, f"unowned, {age/3600:.1f}h old"))
    return candidates, kept


def main(argv=None):
    ap = argparse.ArgumentParser(description="reap leaked .lab/tmp scratch")
    ap.add_argument("action", nargs="?", default="reap", choices=("reap", "record"))
    ap.add_argument("path", nargs="?")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--pattern", action="append", default=None,
                    help=f"basename glob (repeatable; default: {' '.join(DEFAULT_PATTERNS)})")
    ap.add_argument("--older-than-hours", type=float, default=48.0)
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--pid", type=int, default=None, help="with 'record'")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"scratch reaper: no such root {root}", file=sys.stderr)
        return 2

    if args.action == "record":
        if not args.path or not args.pid:
            print("record needs <dir> --pid <pid>", file=sys.stderr)
            return 2
        rec = append_record(root, Path(args.path).resolve(), args.pid)
        print(f"recorded {rec['path']} pid={rec['pid']} starttime={rec['starttime']}")
        return 0

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    candidates, kept = plan(root, patterns, args.older_than_hours)
    freed = sum(dir_bytes(p) for p, _ in candidates)

    deleted, failed = [], []
    if args.apply:
        root_s = str(root).rstrip("/") + "/"
        for path, why in candidates:
            # belt and braces: never delete outside the root, never a symlink
            if not (str(path).startswith(root_s) and path.is_dir()
                    and not path.is_symlink()):
                failed.append((path, "refused: not a real dir under root"))
                continue
            try:
                shutil.rmtree(path)
                deleted.append((path, why))
            except OSError as e:
                failed.append((path, f"{e.strerror or e} (errno {e.errno})"))

    if args.json:
        print(json.dumps({
            "root": str(root),
            "patterns": list(patterns),
            "older_than_hours": args.older_than_hours,
            "apply": args.apply,
            "candidates": [{"path": str(p), "why": w} for p, w in candidates],
            "kept": [{"path": str(p), "why": w} for p, w in kept],
            "candidate_bytes": freed,
            "deleted": [{"path": str(p), "why": w} for p, w in deleted],
            "failed": [{"path": str(p), "why": w} for p, w in failed],
        }, indent=2))
    else:
        verb = "deleted" if args.apply else "would delete"
        for p, why in (deleted if args.apply else candidates):
            print(f"  {verb} {p}  [{why}]")
        for p, why in kept:
            print(f"  keep    {p}  [{why}]")
        print(f"scratch-reap {root}: {len(candidates)} candidate(s), "
              f"{freed/GiB:.2f} GiB {'reclaimed' if args.apply else 'reclaimable'}, "
              f"{len(kept)} kept, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
