"""Guard: reap_appendonly_worktrees removes recorded stale paths by name (gpu-sim-k3gb)."""
import json, subprocess, sys, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAPER = ROOT / "scripts" / "reap_appendonly_worktrees.py"


def test_reap_removes_recorded_path():
    with tempfile.TemporaryDirectory(dir=ROOT / ".lab" / "tmp") as td:
        td = Path(td)
        # create a fake recorded worktree dir
        wt = td / "fake-wt"
        wt.mkdir()
        (wt / "x.txt").write_text("hi")
        record = ROOT / ".lab" / "tmp" / "appendonly-worktrees.json"
        # save original
        orig = record.read_text() if record.exists() else None
        try:
            record.write_text(json.dumps([{"path": str(wt), "pid": 999999}]))
            r = subprocess.run(
                [sys.executable, str(REAPER), "--repo", str(ROOT)],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
            assert "reaped:" in r.stdout or str(wt) in r.stdout
            assert not wt.exists(), "recorded path not removed"
            data = json.loads(record.read_text())
            assert all(r.get("path") != str(wt) for r in data), "record not cleared"
        finally:
            if orig is None:
                try:
                    record.unlink()
                except FileNotFoundError:
                    pass
            else:
                record.write_text(orig)


def test_no_record_file_is_ok():
    # point at a repo with no record file
    with tempfile.TemporaryDirectory(dir=ROOT / ".lab" / "tmp") as td:
        r = subprocess.run(
            [sys.executable, str(REAPER), "--repo", str(Path(td))],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
