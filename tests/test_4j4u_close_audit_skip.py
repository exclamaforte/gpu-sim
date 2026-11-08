"""Guard: close audit reports UNVERIFIABLE (rc=1) in a checkout with no beads DB (gpu-sim-4j4u rework)."""
import subprocess, os, tempfile, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_close_audit_skips_without_db():
    # True DB-less clone (real git clone, .beads removed): uses real bd.
    dst = Path("/var/tmp/realclone-4j4u")
    if dst.exists():
        shutil.rmtree(dst)
    subprocess.run(["git", "clone", "/home/gabe/Documents/gpu-sim", str(dst)], check=True, capture_output=True)
    beads = dst / ".beads"
    if beads.exists():
        shutil.rmtree(beads)
    # Copy the modified script under test
    (dst / "scripts").mkdir(parents=True, exist_ok=True)
    import shutil as _sh
    _sh.copy(str(REPO / "scripts" / "close_requires_artifact.py"),
             str(dst / "scripts" / "close_requires_artifact.py"))
    try:
        r = subprocess.run(["python3", "scripts/close_requires_artifact.py"],
                           cwd=str(dst), capture_output=True, text=True)
        assert r.returncode != 0, f"expected UNVERIFIABLE rc!=0, got {r.returncode}: {r.stdout}\n{r.stderr}"
        assert "UNVERIFIABLE" in (r.stdout + r.stderr), f"expected UNVERIFIABLE message: {r.stdout}\n{r.stderr}"
    finally:
        shutil.rmtree(dst, ignore_errors=True)

def test_close_audit_audits_with_db():
    # In a worktree (DB available via parent), the audit runs and rc=0
    r = subprocess.run(["python3", "scripts/close_requires_artifact.py"],
                       cwd=str(REPO), capture_output=True, text=True)
    assert r.returncode == 0, f"expected audit rc=0, got {r.returncode}: {r.stdout}\n{r.stderr}"
