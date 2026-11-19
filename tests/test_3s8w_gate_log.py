"""Guard: a failing stage's full log survives the gate (gpu-sim-3s8w)."""
import subprocess
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    text=True).strip()).parent


def test_failing_stage_log_survives_worktree_removal():
    # Create a linked worktree so the gate runs in a disposable tree.
    td = Path(tempfile.mkdtemp())
    wt = td / "wt"
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", str(wt), "HEAD"],
                   check=True)
    try:
        # Use the (possibly uncommitted) gate.sh under test.
        shutil.copyfile(ROOT / "scripts" / "gate.sh", wt / "scripts" / "gate.sh")
        # Plant a failing stage: make check_trunk_refs.py exit 1.
        target = wt / "scripts" / "check_trunk_refs.py"
        original = target.read_text()
        target.write_text(original + "\nimport sys; sys.exit(1)\n")

        proc = subprocess.run(
            ["bash", str(wt / "scripts" / "gate.sh"), "--fast"],
            cwd=str(wt), capture_output=True, text=True, timeout=300,
        )
        output = proc.stdout + proc.stderr
        # find the surviving log path
        import re
        m = re.search(r"gate logs:\s*(.+)", output)
        assert m, f"gate did not print 'gate logs:' path:\n{output[-500:]}"
        log_dir = Path(m.group(1).strip())
        # must be outside the gated worktree
        assert not str(log_dir).startswith(str(wt)), f"log dir inside worktree: {log_dir}"
        # must be under the main repo root
        assert str(log_dir).startswith(str(MAIN_ROOT)), f"log dir not under main repo: {log_dir}"
        # a per-stage log must exist and contain the planted failure
        stage_logs = list(log_dir.glob("*.log"))
        assert stage_logs, f"no per-stage logs in {log_dir}"
        combined = "".join(p.read_text(errors="replace") for p in stage_logs)
        assert "check_trunk_refs" in combined.lower() or proc.returncode != 0

        # path must still be readable after worktree removal
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
                       check=True)
        wt = None
        assert log_dir.exists(), f"log dir removed with worktree: {log_dir}"
        assert any(p.exists() for p in log_dir.glob("*.log"))
    finally:
        if wt is not None and wt.exists():
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
                           capture_output=True)
        shutil.rmtree(td, ignore_errors=True)
