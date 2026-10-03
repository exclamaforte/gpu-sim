#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.7: a comparison must state the baseline's spread before calling an effect.

The module this tests exists because of two measured incidents: a +3.9%
single-replicate reading that a repeat showed was inside the noise, and the v6
anchor's own cross-run spread of median 1.018 / p90 1.235 / max 1.559 across
106 comparable cells.

THE TEST THAT SHAPES THE SUITE, from gpu-sim-u2o.6.11.1: a check that only ever
reports disagreement is indistinguishable from one whose accounting was simply
wrong. Here that means a comparison function that can only ever say
"inconclusive" is worthless. So these tests assert BOTH directions -- ratios
inside the band are inconclusive, and ratios clearly outside it are reported as
real effects -- plus the third case that must never be mistaken for either: no
spread recorded means NO RESOLUTION, not a pass.

Run:  python3 tests/test_replicate_discipline.py
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.measurement_stats import (  # noqa: E402
    spread_from_runs, verdict_for_ratio,
)


class TestSpreadFromRuns(unittest.TestCase):
    def test_joins_on_the_label_across_runs(self):
        s = spread_from_runs([("j1", {"a": 1.0, "b": 2.0}),
                              ("j2", {"a": 1.05, "b": 2.40})], "demo")
        self.assertEqual(s.n_runs, 2)
        self.assertEqual(s.n_cells, 2)
        self.assertEqual(s.run_job_ids, ["j1", "j2"])

    def test_configs_missing_from_any_run_are_excluded_not_zeroed(self):
        """A config present in only one run has no ratio and must not become 1.0.

        Treating a missing config as unchanged would manufacture cells sitting
        exactly on 1.0 and pull the median toward 'no movement', which flatters
        whatever the spread is being used to justify.
        """
        s = spread_from_runs([("j1", {"a": 1.0, "b": 2.0, "c": 5.0}),
                              ("j2", {"a": 1.05, "b": 2.40})], "demo")
        self.assertEqual(s.n_cells, 2, "the one-sided config must be excluded")

    def test_single_run_is_refused_not_defaulted_to_zero(self):
        """THE most important refusal in this file.

        A spread from one run is zero by construction. Returning 0.0 would give
        every later comparison a band of width zero, so ANY difference would
        read as an effect -- the exact inversion this bead exists to stop.
        """
        with self.assertRaises(ValueError) as cm:
            spread_from_runs([("j1", {"a": 1.0})], "demo")
        self.assertIn("at least 2 runs", str(cm.exception))

    def test_all_zero_config_is_refused(self):
        with self.assertRaises(ValueError):
            spread_from_runs([("j1", {"a": 0.0}), ("j2", {"a": 0.0})], "demo")

    def test_band_brackets_the_median(self):
        s = spread_from_runs([("j1", {"a": 1.0}), ("j2", {"a": 1.2}),
                              ("j3", {"a": 1.5})], "demo")
        lo, hi = s.band()
        self.assertLessEqual(lo, s.median)
        self.assertLessEqual(s.median, hi)
        self.assertLessEqual(s.minimum, lo)
        self.assertGreaterEqual(s.maximum, hi)

    def test_record_is_versioned_and_serialisable(self):
        s = spread_from_runs([("j1", {"a": 1.0}), ("j2", {"a": 1.1})],
                             "anchor-v6", spread_version=3)
        d = s.to_dict()
        self.assertEqual(d["spread_version"], 3)
        self.assertEqual(d["baseline_id"], "anchor-v6")
        self.assertIn("resolution_band", d)
        self.assertEqual(len(d["resolution_band"]), 2)


class TestVerdictReportsBothDirections(unittest.TestCase):
    """THE CHECK MUST BE ABLE TO PASS, in both directions."""

    def setUp(self):
        self.spread = spread_from_runs(
            [("j1", {"a": 1.0, "b": 2.0}), ("j2", {"a": 1.05, "b": 2.40})],
            "demo")

    def test_ratio_inside_the_band_is_inconclusive(self):
        """The probe ratio is DERIVED from the band, not typed.

        A hardcoded 1.05 was wrong the moment the statistic changed from
        max/median to the max/min envelope, and it failed on the change rather
        than telling me the band had moved -- which is exactly the hand-typed
        constant this project keeps removing. Deriving it means the test states
        the RULE (inside the band is inconclusive) instead of a number that
        happens to be inside one particular band.
        """
        lo, hi = self.spread.band()
        inside = round((lo + hi) / 2, 6)
        v = verdict_for_ratio(inside, self.spread)
        self.assertEqual(v["verdict"], "inconclusive",
                         f"ratio {inside} sits inside band [{lo}, {hi}] and "
                         f"must be inconclusive")
        self.assertIn("indistinguishable", v["explanation"])

    def test_ratio_above_the_band_is_a_real_slower_effect(self):
        """If this ever fails, the module is a 'nothing matters' machine."""
        v = verdict_for_ratio(1.5, self.spread)
        self.assertEqual(v["verdict"], "effect_slower")
        self.assertIn("exceeds measured run-to-run noise", v["explanation"])

    def test_ratio_below_the_band_is_a_real_faster_effect(self):
        """Below p10, and derived from the band for the same reason."""
        lo, _ = self.spread.band()
        v = verdict_for_ratio(round(lo * 0.9, 6), self.spread)
        self.assertEqual(v["verdict"], "effect_faster")

    def test_missing_spread_claims_no_resolution_and_is_not_a_pass(self):
        v = verdict_for_ratio(1.001, None)
        self.assertEqual(v["verdict"], "no_spread_recorded")
        self.assertIsNone(v["resolution_band"])
        self.assertNotEqual(v["verdict"], "inconclusive")
        self.assertNotEqual(v["verdict"], "effect_slower")


class TestAgainstTheRealAnchorSpread(unittest.TestCase):
    """The spread the bead was filed for, from the committed variance probe."""

    @classmethod
    def setUpClass(cls):
        p = REPO / "experiments" / "gpu-sim-u2o.6.6.2" / "variance_probe.json"
        if not p.exists():
            raise unittest.SkipTest("variance_probe.json not in this tree yet")
        import json
        cls.probe = json.loads(p.read_text())

    def _spread(self):
        o = self.probe["overall_v8_over_v6"]
        from epic_u2o.measurement_stats import SpreadRecord
        return SpreadRecord(
            spread_version=1, baseline_id="v6-anchor",
            n_runs=2, n_cells=o.get("n_cells", 106),
            run_job_ids=["20260928t032717-*", "20260928t053729-*"],
            p10=o["p10"], median=o["median"], p90=o["p90"],
            minimum=o["min"], maximum=o["max"])

    def test_the_anchor_band_is_wide(self):
        """The reason this bead exists: p90 is ~1.235, so small effects are invisible."""
        s = self._spread()
        self.assertGreater(s.p90, 1.2,
                           "the v6 anchor's own cross-run spread should be wide; "
                           "if this is not, the probe or this test is wrong")

    def test_a_four_percent_regression_is_inconclusive_against_this_anchor(self):
        """The practical consequence, stated as an assertion.

        A single-run reading of 1.04x -- the size of effect someone would
        normally report as real -- falls inside the anchor's own spread. That is
        the claim this bead makes, and it should fail loudly if it ever stops
        being true.
        """
        v = verdict_for_ratio(1.04, self._spread())
        self.assertEqual(v["verdict"], "inconclusive",
                         "a 4% single-run difference must not be reported as an "
                         "effect against an anchor whose p90 is above 1.2")

    def test_a_fifty_percent_regression_is_an_effect(self):
        v = verdict_for_ratio(1.5, self._spread())
        self.assertEqual(v["verdict"], "effect_slower")


if __name__ == "__main__":
    unittest.main(verbosity=2)