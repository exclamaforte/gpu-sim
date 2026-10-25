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
    """Touch a real source so the .so becomes stale, expect pytest to fail."""
    src = REPO / "src" / "fakegpu_config.c"
    so = REPO / "libfakegpu.so"
    # make .so older
    old_mtime = src.stat().st_mtime
    # gpu-sim-x1v rework (PI): the staleness condition is "source newer than the
    # .so", so base the bumped mtime on the .so, not on the source. src+10s is
    # only stale when the last build happened <10s after the checkout and stops
    # the guard from firing on a normally-built tree (red on master 02954b0).
    new_mtime = so.stat().st_mtime + 60
    os.utime(src, (new_mtime, new_mtime))
    fresh, msg = freshness(str(so), str(REPO / "src" / "*.c"))
    assert not fresh, "test setup did not make the binary stale: " + msg
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_model_fault.py", "-q"],
            cwd=str(REPO), capture_output=True, text=True, timeout=120)
        assert r.returncode != 0, "guard did not fire"
        assert "freshness guard" in (r.stdout + r.stderr), r.stdout + r.stderr
    finally:
        # restore source mtime and rebuild
        os.utime(src, (old_mtime, old_mtime))
        subprocess.run(["make"], cwd=str(REPO), capture_output=True, timeout=300)
        # verify fresh again
        r2 = subprocess.run(
            [sys.executable, "scripts/check_libfakegpu_fresh.py"],
            cwd=str(REPO), capture_output=True, text=True)
        assert r2.returncode == 0, r2.stdout + r2.stderr
