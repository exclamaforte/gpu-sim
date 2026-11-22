"""Guard: pre-land blob-size check refuses over-limit blobs in history (gpu-sim-enjy)."""
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_blob_size.py"
spec = importlib.util.spec_from_file_location("check_blob_size", SCRIPT)
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)
LIMIT = cb.REMOTE_BLOB_LIMIT_BYTES


def _git(root, *args):
    subprocess.run(["git", *args], cwd=str(root), check=True)


def _init_repo(root):
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@e.com")
    _git(root, "config", "user.name", "t")
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "check_blob_size.py").write_text(SCRIPT.read_text())


def _rev(root, ref):
    return subprocess.run(["git", "-C", str(root), "rev-parse", ref],
                          capture_output=True, text=True, check=True).stdout.strip()


def _run_checker(root):
    return subprocess.run(["python3", "scripts/check_blob_size.py"], cwd=str(root),
                          capture_output=True, text=True)


def test_blob_over_limit_refused():
    scratch = Path(tempfile.mkdtemp(prefix="enjy-", dir=ROOT / ".lab" / "tmp"))
    try:
        root = scratch / "repo"
        _init_repo(root)
        big = root / "big.bin"
        big.write_bytes(b"\0" * (LIMIT + 1))
        _git(root, "add", "big.bin")
        blob_sha = _rev(root, ":big.bin")
        _git(root, "commit", "-qm", "add big")
        add_commit = _rev(root, "HEAD")
        big.unlink()
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "delete big")
        r = _run_checker(root)
        assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
        out = r.stdout + r.stderr
        assert "big.bin" in out
        assert str(LIMIT + 1) in out
        assert str(LIMIT) in out
        # the ACTUAL blob sha and the commit that introduced it, not any 40-hex
        assert blob_sha in out, f"blob sha {blob_sha} not in refusal: {out}"
        assert add_commit in out, f"introducing commit {add_commit} not in refusal: {out}"
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def test_blob_exact_limit_ok():
    scratch = Path(tempfile.mkdtemp(prefix="enjy-", dir=ROOT / ".lab" / "tmp"))
    try:
        root = scratch / "repo"
        _init_repo(root)
        big = root / "big.bin"
        big.write_bytes(b"\0" * LIMIT)
        _git(root, "add", "big.bin")
        _git(root, "commit", "-qm", "add big")
        big.unlink()
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "delete big")
        r = _run_checker(root)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def test_merged_side_branch_names_introducer():
    scratch = Path(tempfile.mkdtemp(prefix="enjy-", dir=ROOT / ".lab" / "tmp"))
    try:
        root = scratch / "repo"
        _init_repo(root)
        # initial commit on main
        (root / "README").write_text("x\n")
        _git(root, "add", "README")
        _git(root, "commit", "-qm", "init")
        # side branch adds big then deletes
        _git(root, "checkout", "-q", "-b", "side")
        big = root / "big.bin"
        big.write_bytes(b"\0" * (LIMIT + 1))
        _git(root, "add", "big.bin")
        _git(root, "commit", "-qm", "side add big")
        side_add_commit = _rev(root, "HEAD")
        big.unlink()
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "side delete big")
        _git(root, "checkout", "-q", "main")
        _git(root, "merge", "--no-ff", "-qm", "merge side", "side")
        r = _run_checker(root)
        assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
        out = r.stdout + r.stderr
        assert "big.bin" in out
        assert side_add_commit in out, (
            f"introducing side-branch commit {side_add_commit} not named in refusal: {out}")
        assert "unknown" not in out
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def test_blob_under_limit_ok():
    scratch = Path(tempfile.mkdtemp(prefix="enjy-", dir=ROOT / ".lab" / "tmp"))
    try:
        root = scratch / "repo"
        _init_repo(root)
        (root / "small.txt").write_text("hello\n")
        _git(root, "add", "small.txt")
        _git(root, "commit", "-qm", "small")
        r = _run_checker(root)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
