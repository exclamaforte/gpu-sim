"""Guard: close audit skips (rc=0) when no beads DB is present (gpu-sim-4j4u)."""
import subprocess, os, tempfile, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_close_audit_skips_without_db():
    # Simulate a DB-less checkout by copying repo to temp and removing .beads/embeddeddolt
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "clone"
        subprocess.run(["cp", "-r", str(REPO), str(dst)], check=True)
        db = dst / ".beads" / "embeddeddolt"
        if db.exists():
            shutil.rmtree(db)
        r = subprocess.run(["python3", "scripts/close_requires_artifact.py"],
                           cwd=str(dst), capture_output=True, text=True)
        assert r.returncode == 0, f"expected skip rc=0, got {r.returncode}: {r.stdout}\n{r.stderr}"
