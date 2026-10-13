#!/usr/bin/env python3
"""Tests for the prose/artifact agreement lint. gpu-sim-u2o.6.11.1.

Criterion 1 is "it FIRES on the historical instances", verified by running it
against the committed artifacts. Criterion 3 is the harder one and the one that
matters more: on 6.6.1 it must NOT fire for a claim the artifact SUPPORTS. A
lint that flags everything is as useless as the hand-checking it replaces, and
worse, it trains people to ignore it.

That asymmetry is why the negative controls here are not decoration. The 8dw
case is the reason the whole tool refuses to adjudicate: the repaired preflight
legitimately passes, a summary claimed it failed, and 'fixing' the code to match
the summary would have introduced a FALSE NEGATIVE to close a false positive.

Run:  python3 tests/test_lint_prose_numbers.py
"""
import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "lint", REPO / "experiments" / "gpu-sim-u2o.6.11.1" / "lint_prose_numbers.py")
lint = importlib.util.module_from_spec(_spec)
# register BEFORE exec: @dataclass resolves its own module through
# sys.modules at class-creation time, and a dynamically loaded module that was
# never registered makes that lookup return None
sys.modules["lint"] = lint
_spec.loader.exec_module(lint)


class TestFiresOnHistory(unittest.TestCase):
    """Criterion 1: each historical instance is caught against committed data."""

    def test_6_6_1_worst_case_mdd(self):
        f = []
        lint.check_6_6_1_worst_case_mdd(f)
        self.assertTrue(f, "6.6.1's wrong MDD cell was not caught")
        # the artifact's own declared worst case is 76.3 at N=10
        self.assertEqual(f[0].artifact_value, 76.3)
        self.assertEqual(f[0].comment_value, 59.3)

    def test_6_6_3_discriminates_ordering(self):
        f = []
        lint.check_6_6_3_constant_boolean(f)
        self.assertTrue(f, "6.6.3's constant-boolean-as-discriminator was not caught")
        self.assertIn("discriminates", f[0].artifact_value)

    def test_6_10_seconds_vs_milliseconds(self):
        f = []
        lint.check_6_10_ratio(f)
        self.assertTrue(f, "6.10's unit-ratio was not caught")
        # the operands give ~2713x, the prose reported the printed 3x
        self.assertIn("2713", f[0].artifact_value)
        self.assertEqual(f[0].comment_value, 3.0)

    def test_8dw_false_where_true(self):
        f = []
        lint.check_8dw_claim_vs_output(f)
        self.assertTrue(f, "8dw's False-where-True claim was not caught")
        self.assertIn("passed=True", f[0].artifact_value)

    def test_6_11_typed_count(self):
        f = []
        lint.check_6_11_counts(f)
        self.assertTrue(f, "6.11's typed removal count was not caught")
        self.assertEqual(f[0].comment_value, 33)


class TestDoesNotFireOnSupportedClaims(unittest.TestCase):
    """Criterion 3: the negative control, and the reason the tool is usable."""

    def test_6_6_1_matching_value_does_not_fire(self):
        """A comment quoting the artifact's OWN 76.3% must be silent.

        This is the empty-config-equivalent case: the claim is supported, so
        firing would be a false positive on a correct sentence.
        """
        f = []
        lint.check_6_6_1_worst_case_mdd(
            f, comments="worst case = the empty cell: 10 trials 76.3%")
        self.assertEqual(f, [], "fired on a claim the artifact supports")

    def test_6_10_correct_ratio_does_not_fire(self):
        f = []
        lint.check_6_10_ratio(
            f, comments="it printed 2713x where the real ratio is 2713x")
        self.assertEqual(f, [], "fired on a ratio the operands support")

    def test_8dw_true_claim_does_not_fire(self):
        """The repaired preflight legitimately PASSES one probe. That is not a bug.

        The lint must not report a passing probe as a contradiction -- doing so
        is how a false positive gets 'fixed' by breaking correct code.
        """
        f = []
        lint.check_8dw_claim_vs_output(
            f, comments="the repaired module answers False to the impossible "
                        "launch and True to the empty config")
        self.assertEqual(f, [])

    def test_6_6_3_no_constant_claim_does_not_fire(self):
        f = []
        lint.check_6_6_3_constant_boolean(
            f, comments="power.draw and utilization.gpu separate the regimes")
        self.assertEqual(f, [])


class TestReportsWithoutAdjudicating(unittest.TestCase):
    """Criterion 2: it reports both values and decides nothing."""

    def test_finding_carries_both_values_and_paths(self):
        f = []
        lint.check_6_6_1_worst_case_mdd(f)
        r = f[0].render()
        for needle in ("prose", "artifact", "comment", "disagree", "76.3", "59.3"):
            self.assertIn(needle, r)
        self.assertIn("experiments/gpu-sim-u2o.6.6.1/metrics.json", r)

    def test_no_finding_proposes_a_change(self):
        """The tool must not say which value is right, in any wording."""
        banned = ("should be", "must be", "correct value", "fix by",
                  "wrong value", "is incorrect")
        all_f: list = []
        for fn in lint.CHECKS:
            fn(all_f)
        self.assertTrue(all_f, "expected the historical findings")
        for finding in all_f:
            text = finding.render().lower()
            for b in banned:
                self.assertNotIn(b, text, f"lint adjudicated: {b!r}")


class TestFailClosed(unittest.TestCase):
    def test_a_crashing_check_is_reported_not_swallowed(self):
        """A check that cannot run must not read as a check that passed."""
        def boom(findings, comments=None):
            raise RuntimeError("simulated failure")
        saved = lint.CHECKS
        lint.CHECKS = (boom,)
        try:
            rc = lint.main(["lint"])
        finally:
            lint.CHECKS = saved
        self.assertEqual(rc, 1)

    def test_main_returns_nonzero_on_findings(self):
        self.assertEqual(lint.main(["lint"]), 1)


class TestSpeed(unittest.TestCase):
    def test_cheap_enough_for_every_commit(self):
        """Criterion 6: seconds, not minutes -- it has to be runnable always."""
        import time
        t0 = time.time()
        lint.main(["lint"])
        self.assertLess(time.time() - t0, 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
