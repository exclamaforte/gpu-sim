"""Guard: stamp_agreement_errors reads reachable_from_trunk (gpu-sim-0fwg)."""
import json, subprocess, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _make_repo():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name) / "repo"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True)
    (root / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "c"], check=True)
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    (root / "experiments" / "x").mkdir(parents=True)
    return td, root, sha


def test_new_key_checked():
    td, root, sha = _make_repo()
    try:
        art = root / "experiments" / "x" / "a.json"
        # landing commit = sha (file f.txt landed at sha); but artifact untracked -> no landing
        # Use a real landing: commit the artifact too
        art.write_text(json.dumps({"tree_state": {"branch": "lab/x",
            "superseded_by": {"commit": sha[:7], "commit_kind": "oldest_add",
                              "reachable_from_trunk": False}}}) + "\n")
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "art"], check=True)
        sha2 = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        # rewrite with sha2 and untracked? Simpler: keep committed; landing exists
        art.write_text(json.dumps({"tree_state": {"branch": "lab/x",
            "superseded_by": {"commit": sha[:7], "commit_kind": "oldest_add",
                              "reachable_from_trunk": False}}}) + "\n")
        from scripts.tree_state import stamp_agreement_errors
        # need importable; add scripts to path
        import sys
        sys.path.insert(0, str(ROOT))
        errs = stamp_agreement_errors("experiments/x/a.json", root)
        assert len(errs) >= 1
        assert any("not reachable from trunk" in e for e in errs), errs
    finally:
        td.cleanup()


def test_old_key_shape_checked():
    td, root, sha = _make_repo()
    try:
        art = root / "experiments" / "x" / "a.json"
        art.write_text(json.dumps({"tree_state": {"branch": "lab/x",
            "superseded_by": {"commit": sha[:7], "commit_kind": "oldest_add",
                              "reachable_from_master": False}}}) + "\n")
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "art"], check=True)
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts.tree_state import stamp_agreement_errors
        errs = stamp_agreement_errors("experiments/x/a.json", root)
        assert len(errs) >= 1
        assert any("not reachable from trunk" in e for e in errs), errs
    finally:
        td.cleanup()
