"""The libfakegpu freshness guard must compare the .so against what the Makefile actually compiles.

Regression this pins (2026-10-26): the guard globbed `src/*.c`, which includes src/fakegpu_fabric.c -- a file
SRCS does not link into libfakegpu.so (test binaries only). Touching it made the HARD gate fail with "run `make`"
while `make` correctly did nothing, so no action could satisfy the check. A gate that cannot be satisfied is
worse than no gate: it trains people to ignore hard failures.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_libfakegpu_fresh import freshness, srcs_from_makefile  # noqa: E402

MAKEFILE = """LIB = libfakegpu.so
SRCS = src/a.c src/b.c \\
       src/c.c
OBJS = $(SRCS:.c=.o)
all: $(LIB)
"""


def _mk(path: Path, when: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("/* x */")
    os.utime(path, (when, when))


def test_srcs_come_from_the_makefile_not_a_glob(tmp_path):
    mk = tmp_path / "Makefile"
    mk.write_text(MAKEFILE)
    got = srcs_from_makefile(str(mk))
    assert got == ["src/a.c", "src/b.c", "src/c.c"], got


def test_a_source_outside_srcs_does_not_make_the_so_stale(tmp_path):
    """THE REGRESSION: src/fakegpu_fabric.c is newer than the .so and that is FINE."""
    now = time.time()
    _mk(tmp_path / "libfakegpu.so", now - 100)
    _mk(tmp_path / "src" / "a.c", now - 200)
    _mk(tmp_path / "src" / "fakegpu_fabric.c", now)          # newer, but not in SRCS
    srcs = [str(tmp_path / "src" / "a.c")]                   # i.e. what the Makefile compiles
    fresh, msg = freshness(str(tmp_path / "libfakegpu.so"), srcs)
    assert fresh, msg


def test_a_real_source_newer_than_the_so_is_still_flagged(tmp_path):
    """Negative control: the guard must still catch a genuinely stale interposer."""
    now = time.time()
    _mk(tmp_path / "libfakegpu.so", now - 100)
    _mk(tmp_path / "src" / "a.c", now)                       # in SRCS, newer -> stale
    fresh, msg = freshness(str(tmp_path / "libfakegpu.so"), [str(tmp_path / "src" / "a.c")])
    assert not fresh and "stale" in msg and "a.c" in msg, msg


def test_repo_makefile_parses_and_excludes_the_test_only_source():
    root = Path(__file__).resolve().parents[1]
    srcs = srcs_from_makefile(str(root / "Makefile"))
    assert srcs, "SRCS did not parse: the guard would silently check nothing"
    assert all(Path(s).exists() for s in srcs), "SRCS names a source that does not exist"
    assert not any("fakegpu_fabric" in s for s in srcs), (
        "fakegpu_fabric.c is linked into the fabric tests, not into libfakegpu.so")


def test_make_q_is_the_authority_on_staleness():
    """mtime heuristics and make disagreed on 2026-10-26; make wins, because it owns the dependency graph."""
    from check_libfakegpu_fresh import make_says_stale
    # A target that certainly exists is asked for by name; the return must be a bool or None, never an exception.
    assert make_says_stale("libfakegpu.so") in (True, False)
