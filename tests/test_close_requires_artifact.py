#!/usr/bin/env python3
"""gpu-sim-u2o.15: the close-audit must PASS on a good closure and FAIL on a bare one.

The failure this bead records is not a lie in any one field; it is a close that
asserted completion with nothing behind it. So the check has to do two things,
and the second is the one that keeps it honest:

  * accept a re-closure that cites something real, and
  * REJECT a bare "Closed", and reject a citation to a file that does not exist,
    because naming something absent is the same failure with paperwork attached.

Everything here runs on synthetic beads. The live audit is a separate run
against the real board, so a test can never depend on -- or be changed by --
whatever the board happens to contain today.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "close_guard", REPO / "scripts" / "close_requires_artifact.py")
g = importlib.util.module_from_spec(_spec)
sys.modules["close_guard"] = g
_spec.loader.exec_module(g)


def bead(status="closed", reason="", notes="", labels=("fab-audit-reopened",),
          id="test-1"):
    return {"id": id, "status": status, "close_reason": reason,
            "notes": notes, "labels": list(labels)}


class TestBareClosesAreRejected(unittest.TestCase):
    """The original failure, in its original shape."""

    def test_bare_closed_is_a_violation(self):
        self.assertFalse(g.classify(bead(reason="Closed"))["concrete"])

    def test_confident_but_empty_phrases_are_violations(self):
        for phrase in ("done", "Fixed", "works", "complete", "merged",
                       "resolved", "n/a", "-"):
            self.assertFalse(
                g.classify(bead(reason=phrase))["concrete"],
                f"{phrase!r} was accepted as evidence")

    def test_no_reason_at_all_is_a_violation(self):
        self.assertFalse(g.classify(bead(reason=""))["concrete"])

    def test_prose_with_no_artifact_is_a_violation(self):
        """Confident prose is still not evidence."""
        v = g.classify(bead(reason="All fixes verified and working correctly"))
        self.assertFalse(v["concrete"])


class TestConcretePointersAreAccepted(unittest.TestCase):
    """THE CHECK MUST BE ABLE TO PASS."""

    def test_an_existing_repo_path_is_concrete(self):
        v = g.classify(bead(reason="validated, merged; see "
                                   "experiments/gpu-sim-u2o.17.5/canonical_fixture.json"))
        self.assertTrue(v["concrete"], v)
        self.assertTrue(any("existing artifact" in w for w in v["why"]))

    def test_a_resolvable_commit_is_concrete(self):
        import subprocess
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        self.assertTrue(g.classify(bead(reason=f"merged {head}"))["concrete"])

    def test_a_comparison_figure_is_concrete(self):
        for reason in ("criterion 1 met: 1.86x at 849 files",
                       "reproduced 3/5 cases",
                       "median 0.187 ms across 264 rows"):
            self.assertTrue(g.classify(bead(reason=reason))["concrete"], reason)

    def test_a_path_plus_a_row_count_is_concrete(self):
        v = g.classify(bead(notes="dataset: experiments/gpu-sim-u2o.17.5/"
                               "/canonical_fixture.json, 18 traces"))
        self.assertTrue(v["concrete"])


class TestCitationsToNothingDoNotCount(unittest.TestCase):
    """The failure wearing paperwork."""

    def test_a_path_that_does_not_exist_is_rejected(self):
        v = g.classify(bead(reason="see experiments/nonexistent/ghost.json"))
        self.assertFalse(v["concrete"],
                         "naming a file that is not there is the same failure "
                         "with a citation attached")

    def test_an_unresolvable_commit_is_rejected(self):
        v = g.classify(bead(reason="merged deadbeefdeadbeef"))
        self.assertFalse(v["concrete"])
        # The disqualifier must be REPORTED, not merely applied: a reader
        # needs to see that a commit was cited and could not be found.
        self.assertTrue(any("does not have" in w for w in v["disqualified_by"]), v)

    def test_a_bare_phrase_plus_a_dead_path_is_still_rejected(self):
        v = g.classify(bead(reason="Closed", notes="artifact: nope/missing.json"))
        self.assertFalse(v["concrete"])


class TestScopeIsExactlyTheAuditLabel(unittest.TestCase):
    def test_only_audit_beads_are_in_scope(self):
        issues = [bead(id="a", reason="Closed"),
                  bead(id="b", reason="Closed", labels=["unrelated"])]
        rec = g.audit(issues)
        self.assertEqual(rec["beads_in_scope"], 1)
        self.assertEqual([v["id"] for v in rec["violations"]], ["a"])

    def test_a_wont_fix_close_is_not_a_false_completion(self):
        """A declined close is a DIFFERENT kind of close.

        The acceptance targets claiming DONE with nothing behind it. Demanding a
        measurement artifact from a bead that deliberately declined the work
        would be a category error, and it would train people to attach fake
        dataset paths to wont-fix closures.
        """
        issues = [bead(id="w", reason="won't-fix-as-originally-scoped")]
        rec = g.audit(issues)
        self.assertEqual(rec["violations"], [])
        self.assertIn("w", rec["wont_fix_closures_excluded"])

    def test_open_beads_are_not_judged(self):
        rec = g.audit([bead(status="open", reason="Closed")])
        self.assertEqual(rec["closed_in_scope"], 0)
        self.assertTrue(rec["satisfies_the_acceptance"])

    def test_the_audit_can_pass(self):
        """Empty in-scope-and-closed is a PASS, not a vacuous one."""
        rec = g.audit([bead(status="open")])
        self.assertTrue(rec["satisfies_the_acceptance"])
        self.assertEqual(rec["violations"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)