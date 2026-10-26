"""Guard: libfakegpu.so must be at least as new as every source it is built from.

The source list comes from the Makefile's SRCS variable -- NOT from a `src/*.c` glob. The glob version was wrong
in the direction that matters: it compared the interposer against every C file in src/, including
src/fakegpu_fabric.c, which SRCS does not compile into libfakegpu.so (it is only linked into the fabric test
binaries). The moment that file was touched the gate failed with "libfakegpu.so is stale (run `make`)" and `make`
correctly did nothing, so the check could never be satisfied -- a hard gate that fails for a reason no action can
fix teaches people to ignore it.

Single source of truth: ask the Makefile what it compiles.

Usage:
    python3 scripts/check_libfakegpu_fresh.py          # exits 0 if fresh, 1 if stale
    python3 scripts/check_libfakegpu_fresh.py --so PATH --src GLOB   (explicit source list override)
"""
import argparse
import glob
import os
import re
import sys


def srcs_from_makefile(path="Makefile"):
    """Parse the Makefile's SRCS assignment (with backslash continuations) into a list of source paths.

    Returns [] when the Makefile or the variable is absent, so callers can fall back rather than explode.
    """
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    m = re.search(r"^SRCS\s*[:+]?=\s*((?:.*\\\n)*.*)$", text, re.MULTILINE)
    if not m:
        return []
    body = m.group(1).replace("\\\n", " ")
    body = body.split("#", 1)[0]
    return [t for t in body.split() if t.endswith((".c", ".cc", ".cpp"))]


def freshness(so_path, srcs):
    """Return (fresh: bool, message: str).

    `srcs` is an explicit list of paths, or a glob string for the legacy callers (tests/test_libfakegpu_freshness
    and conftest used to pass "src/*.c"). Accepting both keeps the strict Makefile-SRCS list as the default
    without breaking a caller that was written against the glob API.
    """
    if isinstance(srcs, str):
        srcs = glob.glob(srcs)
    srcs = sorted(srcs)
    if not srcs:
        return True, "no sources found"
    if not os.path.exists(so_path):
        return False, "%s is missing (run `make`)" % so_path
    so_mtime = os.path.getmtime(so_path)
    stale = [s for s in srcs if os.path.getmtime(s) > so_mtime]
    if stale:
        return False, "%s is stale: newer sources %s (run `make`)" % (so_path, ", ".join(stale))
    return True, "fresh"


def assert_fresh(so_path="libfakegpu.so", srcs=None):
    srcs = srcs_from_makefile() if srcs is None else srcs
    fresh, msg = freshness(so_path, srcs)
    if not fresh:
        raise RuntimeError("libfakegpu.so freshness guard: " + msg)


def make_says_stale(so_path="libfakegpu.so", target=None):
    """Ask make itself: `make -q <target>` exits 1 when the target is out of date.

    Make is the authority on staleness -- it decides from the same dependency graph it builds from. The mtime
    comparison below only approximates that, and on 2026-10-26 the two disagreed: the guard said stale, `make`
    said nothing to do, and the hard gate was unsatisfiable until the tree was rebuilt by hand. Prefer make;
    fall back to mtimes only when make is unavailable.
    """
    import subprocess
    try:
        rc = subprocess.run(("make", "-q", target or so_path), capture_output=True).returncode
    except OSError:
        return None
    return rc == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default="libfakegpu.so")
    ap.add_argument("--src", default=None, help="explicit glob override (default: the Makefile's SRCS)")
    ap.add_argument("--makefile", default="Makefile")
    a = ap.parse_args()
    srcs = sorted(glob.glob(a.src)) if a.src else srcs_from_makefile(a.makefile)
    if a.src is None:
        stale = make_says_stale(a.so)
        if stale is True:
            print("%s is stale: `make -q %s` reports it out of date (run `make`)" % (a.so, a.so))
            return 1
    fresh, msg = freshness(a.so, srcs)
    print(msg if srcs or a.src else "no sources found (Makefile SRCS not parsed)")
    return 0 if fresh else 1


if __name__ == "__main__":
    sys.exit(main())
