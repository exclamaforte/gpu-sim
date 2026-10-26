"""gpu-sim-x1v: libfakegpu.so freshness guard tests."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.check_libfakegpu_fresh import freshness


def test_freshness_logic_stale():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "a.c"
        so = d / "lib.so"
        src.write_text("x")
        so.write_text("y")
        # make .so older than src
        old = time.time() - 100
        os.utime(so, (old, old))
        fresh, msg = freshness(str(so), str(src))
        assert not fresh
        assert "stale" in msg


def test_freshness_logic_fresh():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "a.c"
        so = d / "lib.so"
        src.write_text("x")
        time.sleep(0.05)
        so.write_text("y")
        fresh, msg = freshness(str(so), str(src))
        assert fresh


def test_freshness_logic_missing_so():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "a.c"
        src.write_text("x")
        fresh, msg = freshness(str(d / "missing.so"), str(src))
        assert not fresh
        assert "missing" in msg


def test_guard_fires_on_stale_binary():
    """The conftest guard must RAISE when the binary is stale -- tested without touching the real tree.

    REWRITTEN (2026-10-26). The original version bumped the mtime of the REAL src/fakegpu_config.c, ran a nested
    pytest to watch conftest refuse to start, then restored the mtime in a finally block. That is poisonous in
    three ways, and all three fired:
      * a run killed inside the try left the source bumped, and the next run captured the BUMPED mtime as
        "old", so the restore ratcheted the tree permanently stale;
      * because conftest asserts freshness at STARTUP, a poisoned tree makes every later pytest run die with
        INTERNALERROR before a single test executes -- the suite cannot repair itself;
      * the restore called `make`, which does not rebuild when only a source mtime moved and the objects are
        already newer, so "restore then rebuild" did not restore.
    A guard is tested by calling it, not by sabotaging the repository it protects.
    """
    from scripts.check_libfakegpu_fresh import assert_fresh
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src, so = d / "a.c", d / "lib.so"
        src.write_text("x")
        so.write_text("y")
        old = time.time() - 100
        os.utime(so, (old, old))
        try:
            assert_fresh(str(so), [str(src)])
        except RuntimeError as e:
            assert "freshness guard" in str(e)
        else:
            raise AssertionError("guard did not fire on a stale binary")


def test_repo_binary_is_fresh_without_mutating_it():
    """Read-only check of the real tree: the guard must pass and must leave every mtime alone."""
    from scripts.check_libfakegpu_fresh import srcs_from_makefile
    so = REPO / "libfakegpu.so"
    before = {p: p.stat().st_mtime for p in [so, REPO / "src" / "fakegpu_config.c"]}
    # The guard's source list is the Makefile's SRCS, NOT a src/*.c glob: the glob includes
    # src/fakegpu_fabric.c, which SRCS does not compile into the .so (test binaries link it
    # directly). Asserting the glob against the real tree fails in any worktree whose .so
    # predates a fabric.c checkout, even though the gate is green -- the test must check what
    # the gate checks. (gpu-sim-4qcx review found this: worktree .so 12:39 vs fabric.c 13:12.)
    srcs = [str(REPO / x) for x in srcs_from_makefile(str(REPO / "Makefile"))]
    assert srcs, "SRCS did not parse: the guard would silently check nothing"
    fresh, msg = freshness(str(so), srcs)
    assert fresh, msg + " (run `make`)"
    after = {p: p.stat().st_mtime for p in before}
    assert before == after, "the freshness test modified a real file's mtime"
