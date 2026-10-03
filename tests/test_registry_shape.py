#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1: the registry must have ONE shape, and must not KeyError.

The reviewer's item: the registry was "hand-typed, unverified by any test ...
and shaped inconsistently enough to raise KeyError for a downstream consumer".

It had THREE distinct field sets. `perproc_7_1_driver` was missing five fields
a consumer would reasonably index -- `label_range_x` among them -- so
`PROTOCOLS[name]["label_range_x"]` raised KeyError far from the registry, naming
nothing useful.

The suite below pins the uniform shape AND the two things normalisation must not
cost: a real measurement must survive (rows are still 264 for all three), and a
consumer asking a question the registry cannot answer must get a testable value
rather than an exception.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from epic_u2o.corpus_protocol import (  # noqa: E402
    PROTOCOLS, REGISTRY_FIELDS, REGISTRY_MISSING, registry_field,
)


class TestOneShape(unittest.TestCase):
    def test_every_entry_carries_every_registered_field(self):
        for name, entry in PROTOCOLS.items():
            missing = [f for f in REGISTRY_FIELDS if f not in entry]
            self.assertEqual(missing, [],
                             f"{name} is missing {missing}; a consumer indexing "
                             f"that field gets a KeyError at their call site")

    def test_there_is_exactly_one_field_set(self):
        shapes = {frozenset(e) for e in PROTOCOLS.values()}
        self.assertEqual(len(shapes), 1,
                         f"{len(shapes)} distinct field sets in one registry")


class TestNormalisationCostNothing(unittest.TestCase):
    """The dangerous failure is a uniformity fix that erases a measurement."""

    def test_rows_survived_normalisation(self):
        """All three protocols really did measure 264 rows.

        `rows` was NOT missing anywhere, so the null-filling must not have
        touched it. If it had, the corpus decision's own reasoning -- which
        compares protocols that 'both measured 264 rows' -- would have been
        quietly deleted.
        """
        for name, entry in PROTOCOLS.items():
            self.assertEqual(entry.get("rows"), 264,
                             f"{name} lost its rows count to normalisation")

    def test_medians_survived_normalisation(self):
        for name, entry in PROTOCOLS.items():
            self.assertIsInstance(entry.get("median_ai_cv"), float,
                                  f"{name} lost its measured median_ai_cv")

    def test_a_missing_measurement_is_null_with_a_reason(self):
        """Absent is explicit, not an exception waiting downstream."""
        e = PROTOCOLS["perproc_7_1_driver"]
        self.assertIsNone(e["label_range_x"])
        self.assertIn("label_range_x", e["not_measured_reason"])


class TestConsumersCannotKeyError(unittest.TestCase):
    def test_unknown_field_returns_a_falsy_sentinel(self):
        v = registry_field("single_run_7_1", "no_such_field")
        self.assertIs(v, REGISTRY_MISSING)
        self.assertFalse(v)

    def test_unknown_protocol_returns_the_sentinel(self):
        self.assertIs(registry_field("no_such_protocol", "rows"),
                      REGISTRY_MISSING)

    def test_a_real_field_still_reads_through(self):
        self.assertEqual(registry_field("single_run_7_1", "rows"), 264)


if __name__ == "__main__":
    unittest.main(verbosity=2)
