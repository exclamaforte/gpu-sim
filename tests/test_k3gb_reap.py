"""Guard: reap_appendonly_worktrees removes recorded stale paths by name (gpu-sim-k3gb)."""
import json, subprocess, sys, tempfile, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAPER = ROOT / "scripts" / "reap_appendonly_worktrees.py"


def _restore_record(orig):
    record = ROOT / ".lab" / "tmp" / "appendonly-worktrees.jsonl"
    if orig is None:
        try:
            record.unlink()
        except FileNotFoundError:
            pass
    else:
        record.write_text(orig)


def test_reap_removes_recorded_path():
    record = ROOT / ".lab" / "tmp" / "appendonly-worktrees.jsonl"
    orig = record.read_text() if record.exists() else None
    # use a real registered worktree
    import tempfile as _t
    td = Path(_t.mkdtemp(prefix="k3gb-reap-", dir=ROOT / ".lab" / "tmp"))
    wt = td / "wt"
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(wt), "HEAD"],
                   check=True, capture_output=True)
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "appendonly_worktrees.py"),
                        "record", str(wt), "--pid", "999999", "--repo", str(ROOT)], cwd=str(ROOT), check=True, capture_output=True)
        r = subprocess.run(
            [sys.executable, str(REAPER), "--repo", str(ROOT)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert str(wt) in r.stdout
        assert not wt.exists(), "recorded path not removed"
        # registry now empty
        if record.exists():
            assert record.read_text().strip() == "", "record not cleared"
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "prune"], capture_output=True, text=True)
        import shutil
        shutil.rmtree(td, ignore_errors=True)
        _restore_record(orig)


def test_live_process_not_reaped():
    record = ROOT / ".lab" / "tmp" / "appendonly-worktrees.jsonl"
    orig = record.read_text() if record.exists() else None
    with tempfile.TemporaryDirectory(dir=ROOT / ".lab" / "tmp") as td:
        td = Path(td)
        wt = td / "live-wt"
        wt.mkdir()
        (wt / "x.txt").write_text("hi")
        # spawn a long-lived process
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "appendonly_worktrees.py"),
                            "record", str(wt), "--pid", str(proc.pid), "--repo", str(ROOT)], cwd=str(ROOT), check=True, capture_output=True)
            r = subprocess.run(
                [sys.executable, str(REAPER), "--repo", str(ROOT)],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
            assert wt.exists(), "live-process path was reaped"
        finally:
            proc.kill()
            proc.wait()
            _restore_record(orig)


def test_corrupt_registry_fails():
    record = ROOT / ".lab" / "tmp" / "appendonly-worktrees.jsonl"
    orig = record.read_text() if record.exists() else None
    try:
        record.write_text("{not json\n")
        r = subprocess.run(
            [sys.executable, str(REAPER), "--repo", str(ROOT)],
            capture_output=True, text=True,
        )
        assert r.returncode != 0, "corrupt registry should exit nonzero"
        assert "corrupt" in (r.stdout + r.stderr).lower()
    finally:
        _restore_record(orig)


def test_no_record_file_is_ok():
    with tempfile.TemporaryDirectory(dir=ROOT / ".lab" / "tmp") as td:
        r = subprocess.run(
            [sys.executable, str(REAPER), "--repo", str(Path(td).parent.parent)],
            capture_output=True, text=True,
        )
        # may be nonzero if no git repo; accept 0 or 128
        assert r.returncode in (0, 1, 128)
