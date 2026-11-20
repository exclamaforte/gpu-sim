#!/usr/bin/env python3
"""Reap leaked disposable append-only worktrees recorded by name."""
import argparse, json, os, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()
    repo = Path(args.repo) if args.repo else Path.cwd()
    record = repo / ".lab" / "tmp" / "appendonly-worktrees.json"
    if not record.is_file():
        print("no recorded append-only worktrees")
        return 0
    try:
        data = json.loads(record.read_text())
    except Exception:
        data = []
    if not isinstance(data, list):
        data = []
    remaining = []
    reaped = 0
    for rec in data:
        path = rec.get("path")
        pid = rec.get("pid")
        if not path:
            continue
        # live process check
        alive = False
        if pid is not None:
            try:
                os.kill(int(pid), 0)
                alive = True
            except (OSError, ValueError):
                alive = False
        if alive:
            remaining.append(rec)
            continue
        # reap by name
        p = Path(path)
        if p.exists():
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", path],
                           capture_output=True, text=True)
            try:
                import shutil
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass
        subprocess.run(["git", "-C", str(repo), "worktree", "prune"],
                       capture_output=True, text=True)
        print(f"reaped: {path}")
        reaped += 1
    record.write_text(json.dumps(remaining))
    if reaped == 0:
        print("no stale append-only worktrees to reap")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
