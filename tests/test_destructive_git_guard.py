#!/usr/bin/env python3
"""gpu-sim-edu.1.1: the guard must refuse, and must leave the work recoverable.

Acceptance item 3 is the demanding one: a fixture with a dirty main tree runs
the destructive command WITH the guard, asserts it refuses, and then asserts
`git fsck --unreachable` still surfaces a "WIP on" commit holding the dirty
files. That is the property that actually saved the 2026-10-03 incident, and a
guard that refuses without proving recoverability would trade data loss for data
loss.

Everything runs in a THROWAWAY repository created by the test, never in the real
tree. A test that dirties the main tree to prove a point about the main tree
would be the same defect it is testing for -- and while writing this bead I made
exactly that mistake for real, by creating a worktree path with the file tool
instead of `git worktree add`, so every git command there walked up to the main
repository.
"""
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "guard", REPO / "scripts" / "guard_destructive_git.py")
guard = importlib.util.module_from_spec(_spec)
sys.modules["guard"] = guard
_spec.loader.exec_module(guard)


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


class FixtureRepo(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.repo = self.td / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "t@example.com")
        git(self.repo, "config", "user.name", "t")
        (self.repo / "AGENTS.md").write_text("original\n")
        (self.repo / "CLAUDE.md").write_text("original\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "base")
        (self.repo / "AGENTS.md").write_text("LOCAL EDIT THAT MUST SURVIVE\n")
        (self.repo / "CLAUDE.md").write_text("LOCAL EDIT THAT MUST SURVIVE\n")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def fsck_unreachable(self):
        out = subprocess.run(["git", "-C", str(self.repo), "fsck",
                              "--unreachable", "--no-progress"],
                             capture_output=True, text=True).stdout
        return {ln.split()[-1] for ln in out.splitlines()
                if "unreachable commit" in ln}


class TestItSeesTheDirtyTree(FixtureRepo):
    def test_modified_tracked_files_are_found(self):
        self.assertEqual(sorted(guard.main_tree_dirty(self.repo)),
                         ["AGENTS.md", "CLAUDE.md"])


class TestItRefusesTheIncidentCommand(FixtureRepo):
    def test_checkout_dot_is_refused(self):
        """`git checkout -- .` in a dirty main tree. This IS the incident."""
        res = guard.guard(["git", "checkout", "--", "."], self.repo, run=False)
        self.assertTrue(res["refused"])
        self.assertIn("Ambient CWD is not a target", res["reason"])
        self.assertFalse(res["ran"])

    def test_refusing_leaves_the_edits_intact(self):
        guard.guard(["git", "checkout", "--", "."], self.repo, run=False)
        for f in ("AGENTS.md", "CLAUDE.md"):
            self.assertIn("MUST SURVIVE", (self.repo / f).read_text())

    def test_restore_dot_reset_hard_and_clean_are_refused(self):
        for argv in (["git", "restore", "."], ["git", "reset", "--hard"],
                     ["git", "clean", "-fd"]):
            self.assertTrue(guard.guard(argv, self.repo, run=False)["refused"],
                            f"{argv} was not refused")

    def test_dot_looks_named_but_is_cwd_scoped(self):
        """The subtle one: `-- .` HAS a pathspec, so a naive guard accepts it.

        That is how my first version let the incident command through. It looks
        named; it is not.
        """
        sub, _f, paths = guard.parse(["git", "checkout", "--", "."])
        self.assertEqual(paths, ["."])
        self.assertFalse(guard.is_scoped(["git", "checkout", "--", "."]))


class TestRecoveryIsReal(FixtureRepo):
    """Acceptance 3's second half: prove recovery, do not assert it."""

    def test_stash_create_leaves_a_reachable_wip_commit_with_the_edits(self):
        before = self.fsck_unreachable()
        h = guard.create_recovery_point(self.repo)
        self.assertTrue(h, "git stash create returned no hash")
        # It must NOT have touched the tree, unlike git stash push.
        self.assertIn("MUST SURVIVE", (self.repo / "AGENTS.md").read_text())
        new = self.fsck_unreachable() - before
        self.assertTrue(new, "the recovery commit is not surfaced by fsck")
        found = None
        for c in new:
            subject = git(self.repo, "log", "-1", "--format=%s", c).stdout
            files = git(self.repo, "show", "--name-only", "--format=", c).stdout
            if subject.startswith("WIP on") and "AGENTS.md" in files:
                found = c
        self.assertIsNotNone(found, f"no 'WIP on' commit with AGENTS.md in {new}")
        self.assertIn("MUST SURVIVE", git(self.repo, "show", f"{found}:AGENTS.md").stdout)

    def test_the_real_incident_commit_still_exists(self):
        """f73d442 is the actual recovery. Asserted against THIS repository so
        the bead's design does not decay into folklore. Skips if pruned."""
        p = subprocess.run(["git", "-C", str(REPO), "cat-file", "-t", "f73d442"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            self.skipTest("f73d442 has been pruned")
        self.assertTrue(git(REPO, "log", "-1", "--format=%s", "f73d442")
                        .stdout.strip().startswith("WIP on"))
        self.assertTrue(git(REPO, "show", "f73d442:AGENTS.md").stdout.strip())


class TestScopedCommandsAreAllowed(FixtureRepo):
    """A guard that refuses everything is a guard nobody routes around."""

    def test_git_dash_C_is_not_refused(self):
        res = guard.guard(["git", "-C", ".lab/wt/x", "checkout", "--", "."],
                          self.repo, run=False)
        self.assertFalse(res["refused"],
                         "an explicit worktree target names the tree")

    def test_git_dash_C_parses_the_worktree_not_the_path_as_subcommand(self):
        """The form the rules REQUIRE must be recognised as destructive."""
        sub, _f, _p = guard.parse(["git", "-C", ".lab/wt/x", "checkout", "--", "."])
        self.assertEqual(sub, "checkout")

    def test_equals_form_target_option_is_not_refused(self):
        res = guard.guard(["git", f"--git-dir={self.repo}", "checkout", "--", "."],
                          self.repo, run=False)
        self.assertFalse(res["refused"])

    def test_a_real_pathspec_is_not_refused(self):
        self.assertFalse(guard.guard(["git", "checkout", "--", "AGENTS.md"],
                                     self.repo, run=False)["refused"])

    def test_a_clean_tree_permits_a_cwd_scoped_discard(self):
        git(self.repo, "checkout", "--", ".")
        self.assertEqual(guard.main_tree_dirty(self.repo), [])
        self.assertFalse(guard.guard(["git", "checkout", "--", "."],
                                     self.repo, run=False)["refused"],
                         "with nothing to lose, the guard must not block")


class TestUnparseableIsRefused(FixtureRepo):
    def test_the_guard_refuses_what_it_cannot_read(self):
        with self.assertRaises(ValueError):
            guard.guard(["git", "--flag-then-nothing"], self.repo)


if __name__ == "__main__":
    unittest.main(verbosity=2)