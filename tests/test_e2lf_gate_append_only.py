"""e2lf: the landing gate runs append-only in a disposable worktree (gpu-sim-e2lf)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gate.sh"
ROW = ROOT / "tests" / "fs2y_ledger.d" / "ypdk_tally.json"


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                         capture_output=True, text=True, check=True)


def test_append_only_stage_fails_on_induced_violation_and_cleans_up():
    """A merged tip whose ledger row carries mutation 'n/a' must make the
    append-only stage FAIL inside the gate, and the gate must leave no
    disposable 'appendonly' worktree behind."""
    original_branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    original_head = _git("rev-parse", "HEAD").stdout.strip()
    backup = ROW.read_text()
    temp_branch = "e2lf-test-violation"
    try:
        _git("checkout", "-b", temp_branch)
        d = json.loads(ROW.read_text())
        d["mutation"] = "n/a"
        ROW.write_text(json.dumps(d, indent=2) + "\n")
        _git("add", str(ROW.relative_to(ROOT)))
        _git("commit", "-m", "e2lf induced violation (test)")
        r = subprocess.run(["bash", str(SCRIPT), "--fast"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        assert r.returncode != 0, f"expected gate failure, got rc={r.returncode}"
        assert "append-only records" in out and "FAIL" in out, out
        # cleanup: no leftover appendonly worktree
        wl = subprocess.run(["git", "-C", str(ROOT), "worktree", "list"],
                            capture_output=True, text=True).stdout
        assert "appendonly" not in wl, f"leftover worktree: {wl}"
    finally:
        # restore branch/commit
        subprocess.run(["git", "-C", str(ROOT), "checkout", original_branch],
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", str(ROOT), "branch", "-D", temp_branch],
                       capture_output=True, text=True)
        ROW.write_text(backup)
        # ensure clean
        subprocess.run(["git", "-C", str(ROOT), "checkout", "--", str(ROW.relative_to(ROOT))],
                       capture_output=True, text=True)
