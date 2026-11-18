"""2fo4: --fast must name the stage it skips (behavioural, via gate_coverage)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GC = ROOT / "scripts" / "gate_coverage.py"


def _run(*args):
    r = subprocess.run([sys.executable, str(GC), *args],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_fast_mode_prints_skipped_suite():
    rc, out = _run("--fast", "--pass", "12", "--fail", "0", "--advisory", "0")
    assert rc == 0, out
    assert "SKIPPED: suite (pytest) -- fast mode" in out, out


def test_fast_summary_names_skipped():
    rc, out = _run("--fast", "--pass", "12", "--fail", "0", "--advisory", "0")
    assert rc == 0, out
    assert "1 skipped (suite)" in out, out


def test_full_mode_does_not_print_skipped():
    rc, out = _run("--pass", "12", "--fail", "0", "--advisory", "0")
    assert rc == 0, out
    assert "SKIPPED: suite (pytest) -- fast mode" not in out, out
