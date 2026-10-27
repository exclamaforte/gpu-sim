#!/usr/bin/env python3
"""Guard: close_audit.json must not change on a branch (gpu-sim-lvta)."""
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_close_audit_unchanged",
    REPO / "scripts" / "check_close_audit_unchanged.py")
mod = importlib.util.module_from_spec(_spec)
sys.modules["check_close_audit_unchanged"] = mod
_spec.loader.exec_module(mod)

REL = "experiments/gpu-sim-u2o.15/close_audit.json"


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


class BaseRepo(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.repo = self.td / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "master")
        git(self.repo, "config", "user.email", "t@example.com")
        git(self.repo, "config", "user.name", "t")
        (self.repo / "experiments" / "gpu-sim-u2o.15").mkdir(parents=True)
        (self.repo / REL).write_text('{"primary_rows": 365}\n')
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "base")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def run_guard(self):
        mod.REPO = self.repo
        mod.PATH = REL
        # capture stdout
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
        return rc, buf.getvalue()


class TestUnchangedPasses(BaseRepo):
    def test_base_branch_unchanged(self):
        rc, out = self.run_guard()
        self.assertEqual(rc, 0)


class TestModifiedFails(BaseRepo):
    def test_modified_value_fails(self):
        p = self.repo / REL
        p.write_text('{"primary_rows": 999}\n')
        git(self.repo, "checkout", "-b", "feature")
        git(self.repo, "add", REL)
        git(self.repo, "commit", "-qm", "modify")
        rc, out = self.run_guard()
        self.assertEqual(rc, 1)
        self.assertIn("differs from its base blob", out)

    def test_deleted_fails(self):
        (self.repo / REL).unlink()
        git(self.repo, "checkout", "-b", "feature")
        git(self.repo, "rm", "-q", REL)
        git(self.repo, "commit", "-qm", "delete")
        rc, out = self.run_guard()
        self.assertEqual(rc, 1)
        self.assertIn("deleted on this branch", out)


class TestHeadVsWorktree(BaseRepo):
    def test_committed_churn_detected_even_if_worktree_restored(self):
        """HEAD carries the churn; restoring the worktree to the base blob must
        NOT hide it, because the guard reads the committed (HEAD) blob."""
        p = self.repo / REL
        p.write_text('{"primary_rows": 999}\n')
        git(self.repo, "checkout", "-b", "feature")
        git(self.repo, "add", REL)
        git(self.repo, "commit", "-qm", "modify")
        # restore the worktree copy to the base blob (HEAD still churned)
        git(self.repo, "checkout", "master", "--", REL)
        rc, out = self.run_guard()
        self.assertEqual(rc, 1)
        self.assertIn("differs from its base blob", out)

    def test_clean_head_with_dirty_worktree_passes(self):
        """A dirty working tree on top of a clean HEAD is not a branch change."""
        p = self.repo / REL
        p.write_text('{"primary_rows": 999}\n')
        rc, out = self.run_guard()
        self.assertEqual(rc, 0)


class TestNoMergeBaseFailsClosed(BaseRepo):
    def test_no_master_main_ref_fails(self):
        # rename master away so no merge-base resolves
        git(self.repo, "branch", "-m", "master", "feature")
        # delete any main
        try:
            git(self.repo, "branch", "-D", "main")
        except subprocess.CalledProcessError:
            pass
        rc, out = self.run_guard()
        self.assertEqual(rc, 1)
        self.assertIn("no merge-base", out)


if __name__ == "__main__":
    unittest.main()
