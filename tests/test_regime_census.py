#!/usr/bin/env python3
"""gpu-sim-u2o.11.3: the regime census fails narrow mixes and total-alone artifacts.

The rule is pre-registered in experiments/gpu-sim-u2o.11.3/census.py BEFORE
any stage 2-5 row is measured. These tests pin the rule's teeth: a 90%+
one-regime mix FAILS, and a total without a census FAILS item 9.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.11.3"))

from census import census, require_census  # noqa: E402


class TestCensusVerdict(unittest.TestCase):
    def test_ninety_percent_one_regime_fails(self):
        r = census({"latency->latency": 95, "latency->memory": 3,
                    "memory->memory": 2})
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(any("share" in s for s in r["reasons"]))

    def test_balanced_three_class_mix_passes(self):
        r = census({"latency->latency": 500, "latency->memory": 500,
                    "memory->memory": 150})
        self.assertEqual(r["verdict"], "PASS")

    def test_thin_memory_bound_fails(self):
        r = census({"latency->latency": 800, "latency->memory": 700,
                    "memory->memory": 32})
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(any("memory->memory" in s for s in r["reasons"]))

    def test_total_without_census_fails_item_9(self):
        with self.assertRaises(ValueError):
            require_census({"total_rows": 2000})

    def test_census_must_sum_to_total(self):
        with self.assertRaises(ValueError):
            require_census({"total_rows": 2000,
                            "regime_census": {"per_class": {"a": 100}}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
