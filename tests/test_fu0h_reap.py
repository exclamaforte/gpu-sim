"""Guard: fu0h reaper removes dead-PID build dirs and keeps live/foreign ones."""
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNSH = ROOT / "experiments" / "gpu-sim-7q3w.1" / "run.sh"
TMP = ROOT / ".lab" / "tmp"


def test_reaper_removes_dead_pid_dir():
    TMP.mkdir(parents=True, exist_ok=True)
    dead = TMP / "7q3w-build-999999"
    dead.mkdir(exist_ok=True)
    (dead / "x.txt").write_text("dead")
    live_proc = subprocess.Popen(["sleep", "60"])
    live = TMP / f"7q3w-build-{live_proc.pid}"
    live.mkdir(exist_ok=True)
    (live / "x.txt").write_text("live")
    foreign = TMP / "7q3w-build-foreign"
    foreign.mkdir(exist_ok=True)
    (foreign / "x.txt").write_text("foreign")
    try:
        r = subprocess.run(["bash", str(RUNSH)], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert not dead.exists(), "dead-PID build dir was not reaped"
        assert live.exists(), "live-PID build dir was reaped"
        assert foreign.exists(), "foreign (non-numeric) build dir was reaped"
    finally:
        live_proc.terminate()
        live_proc.wait()
        for d in (dead, live, foreign):
            import shutil
            shutil.rmtree(d, ignore_errors=True)


def test_reaper_keeps_live_pid_dir():
    # covered by test_reaper_removes_dead_pid_dir; kept as named alias for the ledger.
    test_reaper_removes_dead_pid_dir()
