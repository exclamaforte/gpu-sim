#!/usr/bin/env python3
"""Tests for the analytic roofline baseline. gpu-sim-u2o.9.4.

One class per acceptance criterion, and the tests are built from the ways a
BASELINE goes wrong while still looking rigorous:

  C1  a flat 2*n*4 where the kernel touches three arrays
  C2  a first-element ceiling selection, or an unstated candidate set
  C3  a reconciliation against fresh output, which is an identity
  C4  a per-family number that is really a per-kernel measurement
  C5  a ceiling fitted on a contended cell
  C6  one global error bar across regimes that differ 2x
  C7  clocks.sm used as a contention detector

The one I would most expect to regress is C3, because a fresh computation and
a committed one look identical when they agree.

Run:  python3 tests/test_roofline_baseline.py
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.roofline_baseline import (  # noqa: E402
    ACCOUNTING, CONTENTION, DEVICE, RooflineError,
    assert_no_clock_contention, assert_not_above_spec_peak, bytes_moved,
    is_ceiling_eligible, predicted_us, select_ceiling,
)


class TestC1ByteConvention(unittest.TestCase):
    def test_saxpy_is_three_arrays_not_two(self):
        """y[j] = alpha*x[j] + y[j] reads x, reads y, writes y."""
        self.assertEqual(ACCOUNTING["k_saxpy"].arrays, 3)
        self.assertEqual(bytes_moved("k_saxpy", 1000), 3 * 4 * 1000)

    def test_copy_and_relu_are_two(self):
        for k in ("k_copy", "k_relu"):
            self.assertEqual(ACCOUNTING[k].arrays, 2, k)

    def test_degenerate_kernels_are_not_flat_formula(self):
        """k_noop_store touches ONE element; a flat formula overstates it ~5e6."""
        self.assertEqual(bytes_moved("k_noop_store", 1 << 26), 4)
        self.assertEqual(bytes_moved("k_empty", 1 << 26), 0)
        self.assertLess(bytes_moved("k_noop_store", 1 << 26),
                        2 * 4 * (1 << 26) / 1e5)

    def test_strided_scatter_random_are_distinguished(self):
        """Criterion 1 names these three cases; they must not share a rule."""
        cases = {k: ACCOUNTING[k].special_case for k in
                 ("k_strided_read", "k_scatter_write", "k_random_access")}
        self.assertEqual(cases, {"k_strided_read": "strided",
                                 "k_scatter_write": "scatter",
                                 "k_random_access": "random"})

    def test_gemm_flops_use_sqrt_of_problem_size(self):
        """problem_size is D*D, so FLOPs are 2*D^3 with D=sqrt(n)."""
        d = 1024
        p = predicted_us("k_tiled_gemm", d * d, 1, 256)
        self.assertEqual(p["flops"], 2 * d ** 3)

    def test_unknown_kernel_is_refused_not_flat(self):
        with self.assertRaises(RooflineError) as cm:
            bytes_moved("k_mystery", 1000)
        self.assertIn("flat formula", str(cm.exception))


class TestC2CeilingSelection(unittest.TestCase):
    def test_ceiling_is_argmax_not_first_element(self):
        cands = [{"label": "a", "kernel": "k_copy", "n_problem_size": 1,
                  "median_ms": 1.0, "bytes_moved": 1, "achieved_gbps": 100.0},
                 {"label": "b", "kernel": "k_copy", "n_problem_size": 1,
                  "median_ms": 1.0, "bytes_moved": 1, "achieved_gbps": 900.0}]
        r = select_ceiling(cands)
        self.assertEqual(r["ceiling_from_cell"], "b")
        self.assertEqual(r["ceiling_GBps"], 900.0)

    def test_top_candidates_are_emitted(self):
        cands = [{"label": f"c{i}", "kernel": "k_copy", "n_problem_size": 1,
                  "median_ms": 1.0, "bytes_moved": 1,
                  "achieved_gbps": 1000.0 - i} for i in range(6)]
        r = select_ceiling(cands)
        self.assertEqual(len(r["top_candidates"]), 4)
        self.assertEqual([c["rank"] for c in r["top_candidates"]], [1, 2, 3, 4])

    def test_membership_rule_is_stated(self):
        r = select_ceiling([{"label": "a", "kernel": "k_copy",
                             "n_problem_size": 1, "median_ms": 1.0,
                             "bytes_moved": 1, "achieved_gbps": 100.0}])
        self.assertIn("largest problem size", r["membership_rule"])

    def test_empty_candidate_set_is_refused(self):
        with self.assertRaises(RooflineError):
            select_ceiling([])

    def test_ceiling_above_spec_peak_is_refused(self):
        """My first run selected 3,855 GB/s on a 1,792 GB/s part. Impossible
        means a bug, not a high number."""
        with self.assertRaises(RooflineError) as cm:
            select_ceiling([{"label": "a", "kernel": "k_copy",
                             "n_problem_size": 1, "median_ms": 1.0,
                             "bytes_moved": 1, "achieved_gbps": 3855.0}])
        self.assertIn("IMPOSSIBLE", str(cm.exception).upper())

    def test_non_largest_n_is_ineligible(self):
        ok, why = is_ceiling_eligible("k_copy", 1 << 22, False, "uncontended",
                                      is_largest_n=False)
        self.assertFalse(ok)
        self.assertIn("largest problem size", why)


class TestC5ContentionRefusal(unittest.TestCase):
    def test_contended_cell_is_refused(self):
        ok, why = is_ceiling_eligible("k_copy", 1 << 26, False, "contended")
        self.assertFalse(ok)
        self.assertIn("reads HIGH", why)

    def test_l2_resident_is_refused(self):
        ok, why = is_ceiling_eligible("k_copy", 1 << 22, True, "uncontended")
        self.assertFalse(ok)
        self.assertIn("L2-resident", why)

    def test_scatter_and_random_are_refused(self):
        for k in ("k_strided_read", "k_scatter_write", "k_random_access"):
            ok, _ = is_ceiling_eligible(k, 1 << 26, False, "uncontended")
            self.assertFalse(ok, k)


class TestC6RegimeWeighting(unittest.TestCase):
    def test_multipliers_differ_by_about_two_x(self):
        self.assertGreater(CONTENTION["memory-bound"] /
                           CONTENTION["launch-bound"], 1.5)

    def test_a_single_global_bar_is_not_offered(self):
        self.assertNotIn("global", str(CONTENTION).lower().replace(
            "a single global error bar is wrong", ""))

    def test_compute_bound_multiplier_is_declared_unmeasured(self):
        """It is a float, so the honesty has to live in the recorded text.

        A multiplier presented identically to the two measured ones would let a
        reader treat it as a measurement.
        """
        self.assertIn("NOT MEASURED", CONTENTION["note"] + str(CONTENTION))
        self.assertEqual(CONTENTION["compute-bound"], 1.0)


class TestC7NoClockDetector(unittest.TestCase):
    def test_clocks_sm_is_refused(self):
        for s in ("clocks.sm", "clocks.max.sm"):
            with self.assertRaises(RooflineError):
                assert_no_clock_contention(s)

    def test_power_and_utilization_are_allowed(self):
        for s in ("power.draw", "utilization.gpu"):
            assert_no_clock_contention(s)      # must not raise

    def test_the_measured_spreads_are_recorded(self):
        from epic_u2o.roofline_baseline import (CLOCKS_SPREAD, POWER_SPREAD,
                                                UTIL_SPREAD)
        self.assertLess(CLOCKS_SPREAD, POWER_SPREAD)
        self.assertLess(POWER_SPREAD, UTIL_SPREAD)


class TestMaxNotMin(unittest.TestCase):
    def test_prediction_is_the_max_of_the_two_bounds(self):
        p = predicted_us("k_saxpy", 1 << 20, 1, 256)
        self.assertAlmostEqual(p["predicted_us"],
                               max(p["memory_bound_us"], p["compute_bound_us"]))
        # and a memory-bound kernel is bound by MEMORY, so min() would have
        # picked the ~0 FLOP branch -- the 6.6.4 defect
        self.assertEqual(p["binding_resource"], "memory")
        self.assertLess(p["compute_bound_us"], p["memory_bound_us"])


class TestAgainstCommittedArtifact(unittest.TestCase):
    def metrics(self):
        p = REPO / "experiments" / "gpu-sim-u2o.9.4" / "metrics.json"
        if not p.exists():
            self.skipTest("run fit_roofline.py first")
        return json.loads(p.read_text())

    def test_reconciliation_ran_against_the_committed_path(self):
        r = self.metrics()["criterion_3_reconciliation"]
        self.assertIn("COMMITTED", r["checked_against"])
        self.assertGreater(r["n_cells_reconciled"], 0)
        self.assertEqual(r["n_mismatches"], 0)

    def test_all_seven_criteria_are_present_in_the_artifact(self):
        m = self.metrics()
        for i in range(1, 8):
            key = f"criterion_{i}_"
            self.assertTrue(any(k.startswith(key) for k in m),
                            f"criterion {i} not evidenced")

    def test_census_names_the_single_kernel_families(self):
        c = self.metrics()["criterion_4_census"]
        self.assertEqual(sorted(c["single_kernel_families"]), ["gemm", "reduce"])
        self.assertGreater(c["n_kernels"], 5)

    def test_the_wrong_hypothesis_is_recorded_not_deleted(self):
        """A hypothesis that turned out wrong is a result; deleting it would
        leave the next reader unable to see what was checked."""
        f = self.metrics()["headline_finding"]
        self.assertIn("wrong", f["my_first_hypothesis_was_wrong"].lower())
        self.assertIn("STANDS", f["ceiling_unchanged"][
            "difference_is_launch_config_not_bytes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
