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
        self.assertIn("would discard everything in it without naming a file",
                      res["reason"])
        self.assertEqual(Path(res["guard_checked_tree"]).resolve(),
                         self.repo.resolve())
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


class TestLabRuntimeIsProtected(FixtureRepo):
    """gpu-sim-edu.2: the lab's own runtime must survive `git clean`.

    The role sockets, labd-state.json, the inboxes and the prompts live under
    .lab/ UNTRACKED. That is the whole trap: untracked files do not make the
    tree dirty, so a guard that refuses only on a dirty tree permits exactly the
    command that deletes the lab, precisely when the tree is clean.
    """

    def _fake_repo_with_runtime(self):
        repo = self.td / "labrepo"
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "t@example.com")
        git(repo, "config", "user.name", "t")
        (repo / "tracked.txt").write_text("x\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "base")
        lab = repo / ".lab"
        (lab / "inbox").mkdir(parents=True)
        (lab / "prompts").mkdir()
        (lab / "inbox" / "implementer.md").write_text("hi\n")
        (lab / "labd-state.json").write_text("{}\n")
        (lab / "pi.sock").write_text("")
        return repo

    def test_clean_is_refused_when_it_would_delete_lab_runtime(self):
        repo = self._fake_repo_with_runtime()
        self.assertEqual(guard.main_tree_dirty(repo), [],
                         "lab runtime is untracked, so the tree reads CLEAN")
        res = guard.guard(["git", "clean", "-fd"], repo, run=False)
        self.assertTrue(res["refused"],
                        "a CLEAN tree is exactly when git clean does maximum "
                        "damage; the guard must not permit it")
        self.assertTrue(res["lab_runtime_would_be_removed"])
        self.assertIn("LAB RUNTIME", res["reason"])

    def test_the_refusal_happens_even_though_the_tree_is_clean(self):
        """The specific hole this bead found."""
        repo = self._fake_repo_with_runtime()
        res = guard.guard(["git", "clean", "-fdx"], repo, run=False)
        self.assertTrue(res["refused"])

    def test_a_non_clean_command_is_unaffected(self):
        """checkout of a named file must not be blocked by this rule."""
        repo = self._fake_repo_with_runtime()
        res = guard.guard(["git", "checkout", "--", "tracked.txt"], repo, run=False)
        self.assertFalse(res["refused"])

    def test_a_repo_without_lab_runtime_is_unaffected(self):
        """Can-pass: the rule must not fire where there is nothing to protect."""
        res = guard.guard(["git", "clean", "-fd"], self.repo, run=False)
        self.assertFalse(res["lab_runtime_would_be_removed"],
                         "a repo with no .lab runtime must not be blocked")


class TestEnumerationFailsClosed(FixtureRepo):
    """The reviewer's item: an enumeration that could not run is not empty.

    Returning [] from a failed `git clean -dn` means "no lab runtime at risk",
    which ALLOWS the deletion. That is how a guard becomes confident on the
    strength of not having looked. Every failure path must be (something, False).
    """

    def _stub(self, **kw):
        real_run = guard.subprocess.run
        real_parse = guard.parse

        def run(args, **kwargs):
            if "clean" in args:
                if kw.get("raise_exc") is not None:
                    raise kw["raise_exc"]
                return type("P", (), {"returncode": kw.get("rc", 0),
                                      "stdout": kw.get("out", ""),
                                      "stderr": ""})()
            return real_run(args, **kwargs)
        guard.subprocess.run = run
        return real_run, real_parse

    def test_a_git_error_is_not_an_empty_result(self):
        real, _ = self._stub(rc=128)
        try:
            hits, ok = guard.lab_runtime_paths_would_be_removed(
                self.repo, ["git", "clean", "-fd"])
            self.assertFalse(ok)
        finally:
            guard.subprocess.run = real

    def test_a_timeout_is_not_an_empty_result(self):
        real, _ = self._stub(raise_exc=TimeoutError("boom"))
        try:
            hits, ok = guard.lab_runtime_paths_would_be_removed(
                self.repo, ["git", "clean", "-fd"])
            self.assertFalse(ok)
        finally:
            guard.subprocess.run = real

    def test_an_oserror_is_not_an_empty_result(self):
        real, _ = self._stub(raise_exc=OSError("no git"))
        try:
            hits, ok = guard.lab_runtime_paths_would_be_removed(
                self.repo, ["git", "clean", "-fd"])
            self.assertFalse(ok)
        finally:
            guard.subprocess.run = real

    def test_unrecognised_output_is_not_an_empty_result(self):
        """Output we cannot parse must not be read as 'nothing there'."""
        real, _ = self._stub(out="fatal: something new\nnot a dry-run listing")
        try:
            hits, ok = guard.lab_runtime_paths_would_be_removed(
                self.repo, ["git", "clean", "-fd"])
            self.assertFalse(ok)
        finally:
            guard.subprocess.run = real

    def test_an_unparseable_command_is_not_an_empty_result(self):
        _hits, ok = guard.lab_runtime_paths_would_be_removed(
            self.repo, ["git", "--not-a-real-flag"])
        self.assertFalse(ok)

    def test_the_guard_refuses_when_the_enumeration_fails(self):
        """End to end: the refusal, not just the flag."""
        real, _ = self._stub(rc=128)
        try:
            res = guard.guard(["git", "clean", "-fd"], self.repo, run=False)
        finally:
            guard.subprocess.run = real
        self.assertTrue(res["refused"])
        self.assertIn("could not be completed", res["reason"])

    def test_a_genuine_empty_listing_still_succeeds(self):
        """Can-pass: a repo with no untracked paths enumerates cleanly."""
        hits, ok = guard.lab_runtime_paths_would_be_removed(
            self.repo, ["git", "clean", "-fd"])
        self.assertTrue(ok)
        self.assertEqual(hits, [])


class TestItChecksTheTreeItWillDiscard(FixtureRepo):
    """THE REVIEWER'S BLOCKING ITEM.

    My first version checked dirtiness and took the recovery point in the MAIN
    tree only. So `git -C <worktree> checkout -- .` -- the command form this
    bead's own rules RECOMMEND -- was permitted with no snapshot whenever the
    main tree happened to be clean. That is the incident's shape with the
    location changed: naming a TREE is not naming what you discard.
    """

    def _dirty_worktree(self):
        """A worktree that is dirty while the main tree is clean."""
        wt = self.td / "wt"
        shutil.copytree(self.repo, wt)
        # clean the clone, then dirty it -- the inverse of the fixture
        subprocess.run(["git", "-C", str(wt), "checkout", "--", "."],
                       capture_output=True)
        (wt / "AGENTS.md").write_text("WORKTREE EDIT THAT MUST SURVIVE\n")
        subprocess.run(["git", "-C", str(self.repo), "checkout", "--", "."],
                       capture_output=True)     # main clean
        self.assertEqual(guard.main_tree_dirty(self.repo), [])
        self.assertTrue(guard.main_tree_dirty(wt))
        return wt

    def test_a_dirty_worktree_is_refused_even_though_main_is_clean(self):
        wt = self._dirty_worktree()
        res = guard.guard(["git", "-C", str(wt), "checkout", "--", "."],
                          self.repo, run=False)
        self.assertTrue(res["refused"],
                        "the incident's shape with the location changed: a "
                        "dirty WORKTREE discarded because the MAIN tree is "
                        "clean is exactly what this bead exists to stop")
        self.assertIn(str(wt), res["reason"],
                      "the refusal must NAME the tree it refused to discard, "
                      "or a reader cannot tell which tree was at risk")

    def test_the_guard_reports_which_tree_it_checked(self):
        wt = self._dirty_worktree()
        res = guard.guard(["git", "-C", str(wt), "checkout", "--", "."],
                          self.repo, run=False)
        self.assertEqual(Path(res["guard_checked_tree"]).resolve(), wt.resolve())

    def test_the_recovery_point_comes_from_the_target_tree(self):
        wt = self._dirty_worktree()
        before = subprocess.run(["git", "-C", str(wt), "fsck", "--unreachable",
                                 "--no-progress"], capture_output=True,
                                text=True).stdout
        h = guard.create_recovery_point(wt)
        self.assertTrue(h)
        after = subprocess.run(["git", "-C", str(wt), "fsck", "--unreachable",
                                "--no-progress"], capture_output=True,
                               text=True).stdout
        self.assertNotEqual(before, after,
                            "the snapshot must land in the tree being "
                            "discarded, not in the one you are standing in")

    def test_naming_a_tree_is_not_naming_a_file(self):
        wt = self._dirty_worktree()
        self.assertFalse(guard.paths_are_explicit(
            ["git", "-C", str(wt), "checkout", "--", "."]))
        self.assertTrue(guard.paths_are_explicit(
            ["git", "-C", str(wt), "checkout", "--", "AGENTS.md"]))

    def test_a_clean_worktree_still_passes(self):
        """Can-pass: the guard must not block where there is nothing to lose."""
        wt = self.td / "wt2"
        shutil.copytree(self.repo, wt)
        subprocess.run(["git", "-C", str(wt), "checkout", "--", "."],
                       capture_output=True)
        res = guard.guard(["git", "-C", str(wt), "checkout", "--", "."],
                          self.repo, run=False)
        self.assertFalse(res["refused"])


class TestScopedCommandsAreAllowed(FixtureRepo):
    """A guard that refuses everything is a guard nobody routes around."""

    def test_git_dash_C_is_not_refused(self):
        res = guard.guard(["git", "-C", ".lab/wt/x", "checkout", "--", "F.md"],
                          self.repo, run=False)
        self.assertFalse(res["refused"],
                         "an explicit worktree target names the tree")

    def test_git_dash_C_parses_the_worktree_not_the_path_as_subcommand(self):
        """The form the rules REQUIRE must be recognised as destructive."""
        sub, _f, _p = guard.parse(["git", "-C", ".lab/wt/x", "checkout", "--", "."])
        self.assertEqual(sub, "checkout")

    def test_equals_form_target_option_names_the_same_tree(self):
        """It is recognised as a target, but naming a TREE is no longer enough.

        This test used to assert that `--git-dir=<p> checkout -- .` was allowed.
        That encoded the defect the reviewer caught: naming a tree is not naming
        what you discard. It is now refused on a dirty tree, and allowed only
        when files are named.
        """
        res = guard.guard(["git", f"--git-dir={self.repo}", "checkout", "--", "."],
                          self.repo, run=False)
        self.assertTrue(res["refused"])
        res2 = guard.guard(["git", f"--git-dir={self.repo}", "checkout",
                            "--", "AGENTS.md"], self.repo, run=False)
        self.assertFalse(res2["refused"])

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