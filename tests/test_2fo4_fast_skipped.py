"""2fo4: --fast must name the stage it skips (behavioural)."""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "gate.sh"


def _run_fast(timeout=30):
    proc = subprocess.Popen(
        ["bash", str(GATE), "--fast"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    out = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out.append(proc.stdout.read())
            break
        try:
            line = proc.stdout.readline()
            if line:
                out.append(line)
                if "SKIPPED: suite (pytest) -- fast mode" in line:
                    break
        except Exception:
            break
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    proc.wait()
    return "".join(out)


def test_fast_mode_prints_skipped_suite():
    out = _run_fast()
    assert "SKIPPED: suite (pytest) -- fast mode" in out, out


def test_fast_summary_names_skipped():
    out = _run_fast()
    assert "1 skipped (suite)" in out, out


def test_full_mode_does_not_print_skipped():
    proc = subprocess.Popen(
        ["bash", str(GATE)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    out = []
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            out.append(proc.stdout.read())
            break
        try:
            line = proc.stdout.readline()
            if line:
                out.append(line)
        except Exception:
            break
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    proc.wait()
    text = "".join(out)
    assert "SKIPPED: suite (pytest) -- fast mode" not in text, text
