#!/usr/bin/env python3
"""Pre-land: refuse pushed history carrying blobs over the remote limit."""
import subprocess, sys

REMOTE_BLOB_LIMIT_BYTES = 100 * 1024 * 1024  # 100 MiB


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def main():
    # objects reachable from HEAD (history), not just the tip tree
    rev = git("rev-list", "--objects", "HEAD")
    if rev.returncode != 0:
        print("blob-size: git rev-list failed: " + rev.stderr.strip())
        return 1
    seen = {}
    for line in rev.stdout.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        sha, path = parts
        if not path:
            continue
        seen.setdefault(sha, path)
    offenders = []
    for sha, path in seen.items():
        sz = git("cat-file", "-s", sha)
        if sz.returncode != 0:
            continue
        try:
            size = int(sz.stdout.strip())
        except ValueError:
            continue
        if size > REMOTE_BLOB_LIMIT_BYTES:
            # commit that introduced this path
            log = git("log", "--diff-filter=A", "--format=%H", "-1", "HEAD", "--", path)
            commit = log.stdout.strip().splitlines()[0] if log.stdout.strip() else "unknown"
            offenders.append((path, size, sha, commit))
    if offenders:
        print("BLOB-SIZE: refusing: history carries blobs over the remote limit "
              f"({REMOTE_BLOB_LIMIT_BYTES} bytes):")
        for path, size, sha, commit in offenders:
            print(f"  {path}: {size} bytes (blob {sha}), introduced by {commit}")
        return 1
    print("blob-size OK: no blobs over the remote limit in reachable history")
    return 0


if __name__ == "__main__":
    sys.exit(main())
