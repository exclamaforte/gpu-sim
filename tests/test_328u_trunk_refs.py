"""Guard: check_trunk_refs fires on drift (gpu-sim-328u)."""
import importlib.util, subprocess, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_trunk_refs.py"


def _make_repo():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name) / "repo"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True)
    (root / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "c1"], check=True)
    (root / "g.txt").write_text("y")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "c2"], check=True)
    return td, root


def _load(root):
    spec = importlib.util.spec_from_file_location("ctr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = Path(root)
    return mod


def test_drift_fires():
    td, root = _make_repo()
    try:
        # master one commit behind main
        main_sha = subprocess.run(["git", "-C", str(root), "rev-parse", "main"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        behind = subprocess.run(["git", "-C", str(root), "rev-parse", "main~1"],
                                capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(["git", "-C", str(root), "update-ref", "refs/heads/master", behind], check=True)
        mod = _load(root)
        # capture stdout by monkeypatching? main() prints and returns code
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
        out = buf.getvalue()
        assert rc == 1, (rc, out)
        assert "master=" in out and "main=" in out, out
        assert "divergence" in out, out
    finally:
        td.cleanup()


def test_agree_passes():
    td, root = _make_repo()
    try:
        main_sha = subprocess.run(["git", "-C", str(root), "rev-parse", "main"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(["git", "-C", str(root), "update-ref", "refs/heads/master", main_sha], check=True)
        mod = _load(root)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
        out = buf.getvalue()
        assert rc == 0, (rc, out)
        assert "agree" in out, out
    finally:
        td.cleanup()


def test_no_main_passes():
    td, root = _make_repo()
    try:
        # delete main ref to simulate legacy clone (no main)
        subprocess.run(["git", "-C", str(root), "update-ref", "-d", "refs/heads/main"], check=True)
        mod = _load(root)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
        out = buf.getvalue()
        assert rc == 0, (rc, out)
        assert "skipped" in out, out
    finally:
        td.cleanup()
