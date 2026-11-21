"""Guard: fu0h reaper removes dead-PID build dirs (gpu-sim-fu0h).

This row binds run.sh only; the cu12 twin (run_cu12.sh) carries the same reaper
block and is covered by the same property manually. The corrupt-registry-no-rewrite
property does not apply here: the fu0h shell reaper has no registry, so it
cannot rewrite one; that property lives in
tests/test_k3gb_reap.py::test_corrupt_registry_fails.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNSH = ROOT / "experiments" / "gpu-sim-7q3w.1" / "run.sh"


def test_reaper_removes_dead_pid_dir():
    scratch = Path(tempfile.mkdtemp(prefix="fu0h-root-", dir=ROOT / ".lab" / "tmp"))
    (scratch / ".lab" / "tmp").mkdir(parents=True, exist_ok=True)
    tmp = scratch / ".lab" / "tmp"
    # a guaranteed-dead PID: spawn, terminate, reuse its PID
    proc = subprocess.Popen(["sleep", "60"])
    proc.terminate()
    proc.wait()
    dead = tmp / f"7q3w-build-{proc.pid}"
    dead.mkdir(exist_ok=True)
    (dead / "x.txt").write_text("dead")
    # a live PID (this test process)
    live = tmp / f"7q3w-build-{os.getpid()}"
    live.mkdir(exist_ok=True)
    (live / "x.txt").write_text("live")
    # a foreign (non-numeric) name that must survive
    foreign = tmp / "7q3w-build-foreign"
    foreign.mkdir(exist_ok=True)
    (foreign / "x.txt").write_text("foreign")
    env = dict(os.environ)
    env["FU0H_TMP_ROOT"] = str(scratch)
    try:
        r = subprocess.run(["bash", str(RUNSH)], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=300, env=env)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert not dead.exists(), "dead-PID build dir was not reaped"
        assert live.exists(), "live-PID build dir was reaped"
        assert foreign.exists(), "foreign (non-numeric) build dir was reaped"
    finally:
        for d in (dead, live, foreign):
            shutil.rmtree(d, ignore_errors=True)
        # clean any build dirs created by run.sh under scratch
        for d in tmp.glob("7q3w-build-*"):
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
