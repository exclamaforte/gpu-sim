#!/usr/bin/env python3
"""Tests for the bottleneck comparison. gpu-sim-u2o.12.2.

This bead is a comparison, and a comparison is exactly where a wrong number is
most comfortable: two plausible rates, a division, and a conclusion. So the
tests are built from the ways this comparison could be WRONG IN A WAY THAT
STILL LOOKS RIGHT -- a mismatched population, a rate substituted for a
measurement, a non-positive consumption, and the arithmetic slip this lab has
now made three times (which side is the numerator).

Run:  python3 tests/test_bottleneck_comparison.py
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "cmpb", REPO / "experiments" / "gpu-sim-u2o.12.2" / "compare_bottleneck.py")
cmpb = importlib.util.module_from_spec(_spec)
sys.modules["cmpb"] = cmpb
_spec.loader.exec_module(cmpb)


def gen(**kw):
    base = {"measured_ex_per_sec": 2.389, "overhead_free_ceiling_ex_per_sec": 7911.0}
    base.update(kw)
    return base


def rung(label, ex_s):
    return {"label": label, "params": 1000, "batch": 8, "seq": 128,
            "median_step_s": 8.0 / ex_s if ex_s else 0.0,
            "examples_per_sec_training_can_consume": ex_s}


class TestFailClosed(unittest.TestCase):
    def test_unreadable_source_is_refused_not_substituted(self):
        with self.assertRaises(cmpb.ComparisonError) as cm:
            cmpb.read(("no-such-ref-xyz", "nope.json"))
        self.assertIn("worse than no comparison", str(cm.exception))

    def test_empty_training_list_is_refused(self):
        with self.assertRaises(cmpb.ComparisonError):
            cmpb.compare(gen(), [])

    def test_non_positive_consumption_is_refused(self):
        """A rung reporting 0 or negative ex/s would silently invert the
        verdict, making a starved model look fed."""
        with self.assertRaises(cmpb.ComparisonError) as cm:
            cmpb.compare(gen(), [rung("broken", 0.0)])
        self.assertIn("non-positive", str(cm.exception))

    def test_population_mismatch_is_refused(self):
        """rows read != n_measured would compare against a different set."""
        import inspect
        src = inspect.getsource(cmpb.derive_generation)
        self.assertIn("!= summary[\"n_measured\"]", src)
        self.assertIn("different population", src)

    def test_launches_per_row_variation_is_refused(self):
        import inspect
        src = inspect.getsource(cmpb.derive_generation)
        self.assertIn("disagree on launches_per_row", src)


class TestThePerRowCostIsNotThe6_7Formula(unittest.TestCase):
    """The correction this comparison depends on.

    6.7 divided the TWO invocation medians per row by the FORTY launches per
    row, understating the kernel by exactly n_invocations. A test that does not
    pin the correct formula will let that slip back in, because the wrong one
    produces a plausible number either way.
    """

    def test_kernel_cost_is_launches_times_median(self):
        g = cmpb.derive_generation()
        self.assertAlmostEqual(
            g["gpu_ms_per_row"],
            g["launches_per_row"] * g["kernel_us_per_launch"] / 1000.0,
            places=9)

    def test_it_is_not_the_sum_of_two_medians_divided_by_forty(self):
        g = cmpb.derive_generation()
        wrong = g["kernel_us_per_launch"] / g["launches_per_row"] * 2
        self.assertGreater(g["kernel_us_per_launch"], wrong * 10,
                           "the 6.7 undercount has come back")

    def test_overhead_is_just_under_one_hundred_percent(self):
        g = cmpb.derive_generation()
        self.assertGreater(g["overhead_fraction_of_wall"], 0.99)
        self.assertLess(g["overhead_fraction_of_wall"], 1.0)
        # and specifically NOT the 99.998% reported by 6.7
        self.assertLess(g["overhead_fraction_of_wall"], 0.99998)


class TestVerdicts(unittest.TestCase):
    def test_every_rung_starves_today(self):
        c = cmpb.compare(gen(), [rung("small", 14582.7), rung("large", 232.6)])
        self.assertEqual(sorted(c["rungs_starved_today"]), ["large", "small"])

    def test_a_rung_above_the_ceiling_still_starves_after_the_fix(self):
        """The non-obvious finding: removing process overhead is not enough for
        the small model, because the 40-launch protocol itself costs more."""
        c = cmpb.compare(gen(), [rung("small", 14582.7), rung("medium", 1661.5)])
        self.assertEqual(c["rungs_starved_even_after_overhead_is_removed"],
                         ["small"])
        self.assertNotIn("medium",
                         c["rungs_starved_even_after_overhead_is_removed"])

    def test_ceiling_is_labelled_an_upper_bound(self):
        self.assertIn("UPPER BOUND", cmpb.derive_generation()["ceiling_is"])

    def test_shortfall_direction_is_consistent(self):
        """shortfall_factor_today must be training/generation, not the
        inverse. I got this backwards once already."""
        c = cmpb.compare(gen(), [rung("large", 232.6)])
        row = c["per_rung"][0]
        self.assertGreater(row["shortfall_factor_today"], 1.0)
        self.assertAlmostEqual(
            row["shortfall_factor_today"],
            row["training_consumes_ex_per_sec"] / row["generation_now_ex_per_sec"],
            places=6)


class TestAgainstCommittedArtifacts(unittest.TestCase):
    def metrics(self):
        p = REPO / "experiments" / "gpu-sim-u2o.12.2" / "metrics.json"
        if not p.exists():
            self.skipTest("run compare_bottleneck.py first")
        return json.loads(p.read_text())

    def test_decision_is_recorded_and_is_no(self):
        self.assertFalse(self.metrics()["decision"]["open_training_optimization_work"])

    def test_caveats_are_present(self):
        """A decision without its caveats is the thing this lab keeps making."""
        c = self.metrics()["caveats"]
        self.assertGreaterEqual(len(c), 3)
        joined = " ".join(c)
        self.assertIn("Throughput, not end-to-end", joined)
        self.assertIn("SYNTHETIC", joined)

    def test_correction_to_6_7_is_recorded(self):
        c = self.metrics()["correction_to_6_7"]
        self.assertGreater(c["correct_us"], c["6_7_reported_us"] * 10)
        self.assertIn("20x", c["why"])

    def test_cross_check_against_6_7_agrees(self):
        self.assertLess(self.metrics()["cross_check_vs_6_7"]["disagreement_pct"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
