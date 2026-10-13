#!/usr/bin/env python3
"""gpu-sim-k9w: the bulk-add guard must REFUSE, and must be able to pass.

Reproduces the mistake on gpu-sim-u2o.6.10 -- `git add -A` sweeping 47 files
that master already owns -- in a throwaway repository, and asserts the guard
refuses it. The real repo is never touched: a test that dirties the repository it
is guarding is the defect it tests for.

Four properties, and the second is the one that keeps this honest:
  1. a bulk add over master-owned copies is REFUSED, naming them;
  2. it says WHICH KIND of collision, because "identical" (stale duplicate) and
     "DIFFERS" (an add would overwrite master) call for different responses;
  3. a repo whose untracked files are genuinely its own PASSES -- a guard that
     refuses everything is a guard nobody runs.
  4. the CLI entry point checks the CALLER's tree (subprocess, cwd set),
     because every other test passes repo in explicitly and never touches
     the path that once resolved the wrong tree with a green suite.
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


class TestEntryPointChecksTheCallersTree(Fixture):
    """The regression this suite could not see.

    main() used to resolve the repo from __file__ and never called
    parse_args(), so run from anywhere it reported on the SCRIPT's tree
    (usually clean) instead of the caller's -- a false all-clear with a
    green suite, because every other test passes repo in explicitly and
    never touches the entry point. These run the real argv path as a
    SUBPROCESS with cwd inside the hazardous tree, so current_repo() and
    argument parsing are exercised, not bypassed.
    """
    SCRIPT = REPO / "scripts" / "safe_git_add.py"

    def run_cli(self, *argv, cwd):
        return subprocess.run([sys.executable, str(self.SCRIPT), *argv],
                              capture_output=True, text=True, cwd=str(cwd))

    def test_cli_refuses_from_inside_the_hazardous_tree(self):
        r = self.run_cli(cwd=self.repo)
        self.assertEqual(r.returncode, 2,
                         "the CLI must refuse from inside the hazardous tree, "
                         "not report on the script's own directory.\n" + r.stdout)
        self.assertIn("owned_by_master.py", r.stdout)
        self.assertIn("stable.md", r.stdout)
        self.assertIn("identical", r.stdout,
                      "the stale-duplicate kind must be named")
        self.assertIn("DIFFERS", r.stdout,
                      "the would-overwrite kind must be named")

    def test_cli_passes_where_the_files_are_our_own(self):
        for p in ("owned_by_master.py", "stable.md"):
            (self.repo / p).unlink()
        r = self.run_cli(cwd=self.repo)
        self.assertEqual(r.returncode, 0,
                         "with nothing of master's here the CLI must pass.\n"
                         + r.stdout)

    def test_cli_honours_repo_from_elsewhere(self):
        """--repo must point the check; the old code swallowed every flag."""
        r = self.run_cli("--repo", str(self.repo), cwd=self.td)
        self.assertEqual(r.returncode, 2,
                         "--repo naming the hazardous tree must refuse.\n"
                         + r.stdout)

    def test_cli_rejects_bogus_flags(self):
        r = self.run_cli("--totally-bogus", cwd=self.repo)
        self.assertNotEqual(r.returncode, 0,
                            "argparse must reject unknown flags, not run silently")


if __name__ == "__main__":
    unittest.main(verbosity=2)