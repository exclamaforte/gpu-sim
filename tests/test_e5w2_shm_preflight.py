"""Guard: /dev/shm preflight (gpu-sim-e5w2). Behavioural."""
import os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run_preflight(shm_dir, threshold):
    return subprocess.run(
        [sys.executable, "scripts/shm_preflight.py", "--dir", str(shm_dir),
         "--threshold", str(threshold)],
        cwd=str(REPO), capture_output=True, text=True, timeout=60)


def test_dead_pid_segment_removed():
    with tempfile.TemporaryDirectory() as td:
        seg = Path(td) / "gpusim-corpus-999999999-1048576-deadbeef"
        seg.mkdir()
        (seg / "x").write_text("x")
        r = _run_preflight(td, 100)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert not seg.exists(), "dead-pid segment not removed"


def test_live_pid_segment_survives():
    with tempfile.TemporaryDirectory() as td:
        seg = Path(td) / f"gpusim-corpus-{os.getpid()}-1048576-livebeef"
        seg.mkdir()
        (seg / "x").write_text("x")
        r = _run_preflight(td, 100)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert seg.exists(), "live-pid segment was removed"


def test_above_threshold_fails_with_message():
    with tempfile.TemporaryDirectory() as td:
        # fabricate utilisation by... can't easily fill tmpfs; instead set threshold 0
        env = dict(os.environ)
        r = subprocess.run(
            [sys.executable, "scripts/shm_preflight.py", "--dir", td, "--threshold", "0"],
            cwd=str(REPO), capture_output=True, text=True, timeout=60, env=env)
        assert r.returncode != 0
        assert "PREFLIGHT FAIL" in (r.stdout + r.stderr), (r.stdout, r.stderr)


def test_gate_wiring_registered():
    """The wiring (one grep) is the artifact."""
    gate = (REPO / "scripts" / "gate.sh").read_text()
    assert "scripts/shm_preflight.py" in gate
    assert "SHM_THRESHOLD_PCT" in gate
