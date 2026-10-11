#!/usr/bin/env python3
"""Tests for the expected-cardinality gate. gpu-sim-u2o.8.3.

The gate exists because of a specific omission the PI caught: 8.3 reported
`distinct_vectors: 5` for 264 rows -- correctly named, correctly computed,
correctly reported -- and nothing asked what 5-over-264 IMPLIES.

So the tests are built from the four real collapses this lab produced, not
from hypothetical ones:

    6.6.3  a `discriminates` boolean true for every signal, ordering hidden
    6.6.4  a band table disagreeing with its own headline
    7.7    token digest:  8 distinct for 264 rows  (0.030)
    8.3    feature vector as written: 5 distinct for 264 rows (0.019)

and one that works: the corrected feature vector, 165 of 264 (0.625).

Run:  python3 tests/test_representation_cardinality.py
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.representation_cardinality import (  # noqa: E402
    DEFAULT_FLOOR_RATIO, CardinalityError, assert_cardinality,
    check_cardinality,
)

N_ROWS = 264          # 7.1's real measured batch
CEILING = 264         # every row is a distinct (op_type, n, grid, block)


class TestFiresOnTheRealCollapses(unittest.TestCase):
    def test_8_3_feature_vector_as_written_is_caught(self):
        """5 of 264. The number that was sitting in my own metrics.json."""
        with self.assertRaises(CardinalityError) as cm:
            assert_cardinality(N_ROWS, 5, context="feature vector as written",
                               ceiling=CEILING,
                               ceiling_basis="distinct configs")
        msg = str(cm.exception)
        self.assertIn("259 of 264", msg)
        self.assertIn("INDISTINGUISHABLE", msg)

    def test_7_7_token_digest_is_caught(self):
        with self.assertRaises(CardinalityError):
            assert_cardinality(N_ROWS, 8, context="token digest",
                               ceiling=CEILING,
                               ceiling_basis="distinct configs")

    def test_message_states_the_implication_not_just_the_ratio(self):
        """The implication is the part that was missing. A message reading
        'ratio below floor' would be the same omission in code form."""
        with self.assertRaises(CardinalityError) as cm:
            assert_cardinality(N_ROWS, 5, context="x", ceiling=CEILING)
        msg = str(cm.exception)
        self.assertIn("learns about", msg)
        self.assertIn("Legitimate ceiling is 264", msg)

    def test_a_derived_value_being_correctly_named_is_not_enough(self):
        """Explicit, because this is the whole point of the gate."""
        with self.assertRaises(CardinalityError):
            assert_cardinality(N_ROWS, 5, context="distinct_vectors")


class TestPassesOnWorkingRepresentations(unittest.TestCase):
    def test_corrected_feature_vector_passes(self):
        rep = assert_cardinality(N_ROWS, 165,
                                 context="corrected feature vector",
                                 ceiling=CEILING,
                                 ceiling_basis="distinct configs")
        self.assertTrue(rep.passed)
        self.assertEqual(rep.verdict, CardinalityError.PARTIAL)

    def test_a_perfect_representation_is_distinct(self):
        rep = check_cardinality(N_ROWS, CEILING, context="perfect",
                                ceiling=CEILING)
        self.assertEqual(rep.verdict, CardinalityError.DISTINCT)

    def test_floor_is_not_a_target(self):
        """A representation that separates MORE than the floor is fine."""
        rep = check_cardinality(100, 90, context="over-separating",
                                floor_ratio=DEFAULT_FLOOR_RATIO)
        self.assertTrue(rep.passed)


class TestTheFloorIsADecisionNotAFittedConstant(unittest.TestCase):
    def test_floor_sits_in_the_gap(self):
        """Below the corrected 0.625, an order of magnitude above the 0.019
        collapse. If someone tunes the data this stops holding."""
        corrected = 165 / N_ROWS
        collapsed = 5 / N_ROWS
        self.assertLess(DEFAULT_FLOOR_RATIO, corrected)
        self.assertGreater(DEFAULT_FLOOR_RATIO, collapsed * 5)

    def test_floor_can_be_lowered_deliberately(self):
        """Lowering is allowed -- the guard is that it must be a decision."""
        rep = check_cardinality(N_ROWS, 5, context="deliberate",
                                floor_ratio=0.01)
        self.assertTrue(rep.passed)
        self.assertEqual(rep.floor_ratio, 0.01)

    def test_zero_rows_is_refused(self):
        with self.assertRaises(CardinalityError):
            check_cardinality(0, 0, context="nothing")


class TestAgainstRealArtifacts(unittest.TestCase):
    def test_committed_features_jsonl_meets_the_gate(self):
        p = REPO / "experiments/gpu-sim-u2o.8.3/features.jsonl"
        if not p.exists():
            self.skipTest("run populate_feature_vector.py first")
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        distinct = len({tuple(f["vector"]) for f in rows})
        self.assertTrue(assert_cardinality(len(rows), distinct,
                                           context="committed features",
                                           ceiling=len(rows)).passed)

    def test_committed_metrics_records_the_gate_verdict(self):
        p = REPO / "experiments/gpu-sim-u2o.8.3/metrics.json"
        if not p.exists():
            self.skipTest("run populate_feature_vector.py first")
        m = json.loads(p.read_text())
        gate = m["cardinality_gate"]
        # the defect is REPORTED as a measurement, not asserted away
        self.assertEqual(gate["as_written"]["verdict"], CardinalityError.COLLAPSED)
        # and the working representation PASSES
        self.assertTrue(gate["corrected"]["passed"])
        # and the ceiling was measured, not assumed (456: both corpora)
        self.assertEqual(gate["n_distinct_configs_in_the_rows"], 456)

    def test_no_residual_collision_on_the_full_corpus(self):
        # Supersedes the old FAMILY-axis expectation: with the operator mix
        # (add/mul/div on different units) and call sites in the vector,
        # add/mul/div/copy/relu at the same launch no longer share a vector.
        # 456/456 distinct means there IS no residual to name -- and if a
        # future change re-collapses it, the item-5 gate fails loudly.
        p = REPO / "experiments/gpu-sim-u2o.8.3/metrics.json"
        if not p.exists():
            self.skipTest("run populate_feature_vector.py first")
        m = json.loads(p.read_text())
        self.assertEqual(m["distinct_vectors"]["corrected"], 456)
        self.assertEqual(m["item5_collision_groups"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
