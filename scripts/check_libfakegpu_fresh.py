"""Guard: libfakegpu.so must be at least as new as every src/*.c it is built from.

Usage:
    python3 scripts/check_libfakegpu_fresh.py          # exits 0 if fresh, 1 if stale
    python3 scripts/check_libfakegpu_fresh.py --so PATH --src GLOB
"""
import argparse
import glob
import os
import sys


def freshness(so_path, src_glob="src/*.c"):
    """Return (fresh: bool, message: str)."""
    srcs = sorted(glob.glob(src_glob))
    if not srcs:
        return True, "no sources found"
    if not os.path.exists(so_path):
        return False, "%s is missing (run `make`)" % so_path
    so_mtime = os.path.getmtime(so_path)
    stale = [s for s in srcs if os.path.getmtime(s) > so_mtime]
    if stale:
        return False, "%s is stale: newer sources %s (run `make`)" % (so_path, ", ".join(stale))
    return True, "fresh"


def assert_fresh(so_path="libfakegpu.so", src_glob="src/*.c"):
    fresh, msg = freshness(so_path, src_glob)
    if not fresh:
        raise RuntimeError("libfakegpu.so freshness guard: " + msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default="libfakegpu.so")
    ap.add_argument("--src", default="src/*.c")
    args = ap.parse_args()
    fresh, msg = freshness(args.so, args.src)
    print(msg)
    return 0 if fresh else 1


if __name__ == "__main__":
    sys.exit(main())
