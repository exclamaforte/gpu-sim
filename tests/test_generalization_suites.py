#!/usr/bin/env python3
"""Tests for the generalization suites. gpu-sim-u2o.10.3.

The load-bearing test is the negative control: a degenerate split MUST raise.
Without it this module is the same fail-open instrument 7.7 found, just
prettier -- and 7.7's split reported `leakage_train_test: 0` for a split with
no test set at all, because the leakage check compared two empty sets.

Run:  python3 tests/test_generalization_suites.py
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.generalization_suites import (  # noqa: E402
    MIN_EVAL, MIN_TRAIN, Suite, SuiteError, feasibility, suite_unseen_family,
    suite_unseen_occupancy, suite_unseen_size,
)


def row(fam="add", n=4096, grid=64, block=64, **kw):
    r = {"op_type": fam, "problem_size": n,
         "launch_config": {"grid": [grid, 1, 1], "block": [block, 1, 1]},
         "median_runtime_ms": 0.003}
    r.update(kw)
    return r


def corpus():
    rows = []
    for fam in ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"):
        for e in (12, 14, 16, 18):
            n = 1 << e
            for block, frac in ((64, 1.0), (128, 0.5), (256, 0.125), (1024, 0.01)):
                rows.append(row(fam, n, max(1, int(n * frac / block)), block))
    return rows


class TestDegenerateSplitsRaise(unittest.TestCase):
    """The negative control. A suite that cannot be evaluated is not a suite."""

    def test_empty_eval_raises(self):
        s = Suite("t", "u", train=corpus(), eval_=[])
        with self.assertRaises(SuiteError) as cm:
            s.validate()
        self.assertIn("EMPTY eval set", str(cm.exception))
        self.assertIn("two empty sets", str(cm.exception))

    def test_empty_train_raises(self):
        with self.assertRaises(SuiteError) as cm:
            Suite("t", "u", train=[], eval_=corpus()[:5]).validate()
        self.assertIn("empty TRAIN", str(cm.exception))

    def test_too_few_train_rows_raises(self):
        with self.assertRaises(SuiteError) as cm:
            Suite("t", "u", train=corpus()[:MIN_TRAIN - 1], eval_=corpus()[:9]).validate()
        self.assertIn("below the floor", str(cm.exception))

    def test_too_few_eval_rows_raises(self):
        with self.assertRaises(SuiteError) as cm:
            Suite("t", "u", train=corpus(), eval_=corpus()[:MIN_EVAL - 1]).validate()
        self.assertIn("no error bar", str(cm.exception))

    def test_a_corpora_too_thin_to_split_is_reported_unsupported(self):
        """Not an exception: an honest 'this corpus cannot support it'."""
        thin = [row("f1", 1 << 12) for _ in range(4)]
        f = feasibility(thin)
        for s in f["suites"]:
            self.assertFalse(s["supported"], s["suite"])
            self.assertTrue(s["reason"].strip())


class TestSuitesHoldOutTheRightUnit(unittest.TestCase):
    """A suite that holds out the wrong thing measures the wrong thing."""

    def test_s1_evaluates_strictly_larger_than_any_trained(self):
        s = suite_unseen_size(corpus())
        s.validate()
        self.assertGreater(min(s.eval_[0]["problem_size"] for _ in [0]),
                           max(r["problem_size"] for r in s.train))
        self.assertFalse(set(r["problem_size"] for r in s.train)
                         & set(r["problem_size"] for r in s.eval_))

    def test_s2_holds_out_whole_families(self):
        s = suite_unseen_family(corpus())
        s.validate()
        train_fams = {r["op_type"] for r in s.train}
        eval_fams = {r["op_type"] for r in s.eval_}
        self.assertFalse(train_fams & eval_fams,
                         "a family appears on both sides: the split leaks")
        self.assertTrue(all(r["op_type"] in eval_fams for r in s.eval_))

    def test_s3_holds_out_a_whole_regime(self):
        s = suite_unseen_occupancy(corpus())
        s.validate()
        held = s.held_out[0]
        self.assertTrue(held.startswith("block="))
        self.assertTrue(any(f"block={held.split('=')[1]}" in s.held_out[0]
                            for _ in [0]))

    def test_every_suite_validates_on_a_capable_corpus(self):
        f = feasibility(corpus())
        self.assertEqual(f["n_supported"], f["n_total"], f)
        for s in f["suites"]:
            self.assertTrue(s["validated"], s["suite"])


class TestFeasibilityIsDerivedNotAsserted(unittest.TestCase):
    """I pre-wrote the opposite verdict once. The check must never carry a
    hard-coded conclusion, because that is the failure this bead is adjacent to."""

    def test_the_summary_reports_what_the_suites_returned(self):
        """train + eval must be the rows the suite's own unit selects.

        My first version of this test reconstructed S3 by string-matching
        "block=64" against the held-out label, which silently counted every
        block-64 row instead of the (block, coverage) REGIME, and failed
        128 != 64. The test was wrong, not the module -- and it was wrong in
        the way this lab has been wrong all night: matching on a substring of a
        label instead of on the quantity the label names.
        """
        def regime_of(r):
            n = r["problem_size"]
            threads = r["launch_config"]["grid"][0] * r["launch_config"]["block"][0]
            cov = threads / n if n else 0.0
            cls = "full" if cov >= 0.9 else ("partial" if cov >= 0.1 else "poor")
            return (r["launch_config"]["block"][0], cls)

        f = feasibility(corpus())
        for s in f["suites"]:
            if s["unit"] == "problem_size":
                held = {int(x) for x in s["held_out"]}
                want = sum(1 for r in corpus() if r["problem_size"] in held)
            elif s["unit"] == "structural_family":
                want = sum(1 for r in corpus() if r["op_type"] in s["held_out"])
            else:
                want = sum(1 for r in corpus()
                           if f"block={regime_of(r)[0]},"
                              f"coverage={regime_of(r)[1]}" in s["held_out"][0])
            self.assertEqual(s["n_eval"], want, s["suite"])
            self.assertEqual(s["n_train"], len(corpus()) - want, s["suite"])

    def test_shrinking_the_corpus_can_only_reduce_support(self):
        """Monotone: less data cannot enable a suite."""
        big, small = feasibility(corpus()), feasibility(corpus()[:40])
        self.assertGreaterEqual(big["n_supported"], small["n_supported"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
