"""Guard: pre-land blob-size check refuses blobs over limit in history (gpu-sim-enjy)."""
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_blob_size.py"


def _setup_repo(td):
    root = Path(td) / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(root), check=True)
    big = root / "big.bin"
    big.write_bytes(b"\0" * (120 * 1024 * 1024))
    subprocess.run(["git", "add", "big.bin"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-qm", "add big"], cwd=str(root), check=True)
    big.unlink()
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-qm", "delete big"], cwd=str(root), check=True)
    # copy checker
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "check_blob_size.py").write_text(SCRIPT.read_text())
    return root


def test_blob_over_limit_refused():
    with tempfile.TemporaryDirectory() as td:
        root = _setup_repo(td)
        r = subprocess.run(["python3", "scripts/check_blob_size.py"], cwd=str(root),
                           capture_output=True, text=True)
        assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
        assert "big.bin" in r.stdout + r.stderr, (r.stdout, r.stderr)


def test_blob_under_limit_ok():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=str(root), check=True)
        subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=str(root), check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=str(root), check=True)
        (root / "small.txt").write_text("hello\n")
        subprocess.run(["git", "add", "small.txt"], cwd=str(root), check=True)
        subprocess.run(["git", "commit", "-qm", "small"], cwd=str(root), check=True)
        (root / "scripts").mkdir(exist_ok=True)
        (root / "scripts" / "check_blob_size.py").write_text(SCRIPT.read_text())
        r = subprocess.run(["python3", "scripts/check_blob_size.py"], cwd=str(root),
                           capture_output=True, text=True)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
