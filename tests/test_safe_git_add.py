#!/usr/bin/env python3
"""gpu-sim-k9w: the bulk-add guard must REFUSE, and must be able to pass.

Reproduces the mistake on gpu-sim-u2o.6.10 -- `git add -A` sweeping 47 files
that master already owns -- in a throwaway repository, and asserts the guard
refuses it. The real repo is never touched: a test that dirties the repository it
is guarding is the defect it tests for.

Three properties, and the second is the one that keeps this honest:
  1. a bulk add over master-owned copies is REFUSED, naming them;
  2. it says WHICH KIND of collision, because "identical" (stale duplicate) and
     "DIFFERS" (an add would overwrite master) call for different responses;
  3. a repo whose untracked files are genuinely its own PASSES -- a guard that
     refuses everything is a guard nobody runs.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "safe_git_add", REPO / "scripts" / "safe_git_add.py")
sga = importlib.util.module_from_spec(_spec)
sys.modules["safe_git_add"] = sga
_spec.loader.exec_module(sga)


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


class Fixture(unittest.TestCase):
    """A repo with a master, a worktree-like stale copy, and a genuine new file."""

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.repo = self.td / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "t@example.com")
        git(self.repo, "config", "user.name", "t")
        (self.repo / "base.txt").write_text("base\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "base")

        # The branch forks BEFORE master gains these files. That ordering is the
        # whole hazard: from the branch's point of view they do not exist and are
        # therefore UNTRACKED, while master owns them. My first fixture committed
        # them first, which left them TRACKED after checkout -- so it reproduced
        # nothing, and a fixture that reproduces nothing is worse than none.
        git(self.repo, "checkout", "-q", "-b", "side")

        # master gains two of them; the branch worktree keeps local copies.
        git(self.repo, "checkout", "-q", "master")
        (self.repo / "owned_by_master.py").write_text("# master's version\n")
        (self.repo / "stable.md").write_text("same\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "master gains two files")
        git(self.repo, "checkout", "-q", "side")

        # In the branch: one byte-identical stale copy, one edited, one of our own.
        (self.repo / "owned_by_master.py").write_text("# master's version\n")
        (self.repo / "stable.md").write_text("SIDE EDITED\n")
        (self.repo / "genuinely_mine.py").write_text("# mine\n")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)


class TestBulkAddIsRefused(Fixture):
    def test_collisions_are_found(self):
        hits = dict(sga.collisions(self.repo))
        self.assertIn("owned_by_master.py", hits)
        self.assertIn("stable.md", hits)
        self.assertNotIn("genuinely_mine.py", hits,
                         "a file the branch owns is not a collision")

    def test_it_distinguishes_stale_duplicates_from_overwrites(self):
        """The two need different responses, so they must not read alike."""
        hits = dict(sga.collisions(self.repo))
        self.assertEqual(hits["owned_by_master.py"], "identical",
                         "byte-identical to master: a stale duplicate")
        self.assertEqual(hits["stable.md"], "DIFFERS",
                         "differs from master: adding this would OVERWRITE it")

    def test_it_refuses(self):
        rc = sga.check(self.repo)
        self.assertEqual(rc, 2, "a bulk add over master-owned files must refuse")


class TestItCanPass(Fixture):
    def test_a_repo_of_its_own_files_passes(self):
        for p in ("owned_by_master.py", "stable.md"):
            (self.repo / p).unlink()
        rc = sga.check(self.repo)
        self.assertEqual(rc, 0,
                         "with nothing of master's here, the guard must pass; "
                         "a guard that refuses everything is one nobody runs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
