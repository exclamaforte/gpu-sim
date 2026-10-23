#!/usr/bin/env python3
"""Tests for real throughput evidence. gpu-sim-u2o.6.7.

This bead was reopened because its backing artifact was fabricated:
runs/logs/throughput_demo_run.jsonl repeated the identical "trials"
[5.0, 5.1, 4.9] across its rows. So the tests are built from the ways a
throughput number can be produced with nothing behind it, and the fabricated
file is the fixture -- because it is a real example, not a hypothetical one.

A test that only checked the arithmetic would pass against the fabricated
artifact, which is the whole reason that artifact existed.

Run:  python3 tests/test_throughput_evidence.py
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.throughput_evidence import (  # noqa: E402
    ThroughputEvidenceError, bottleneck_split, cross_check, row_trials,
    throughput_from_timestamps, throughput_from_wall,
)

FABRICATED = (REPO.parents[2] / "runs/logs/throughput_demo_run.jsonl") if len(REPO.parents) > 2 else (REPO / "nonexistent_throughput_demo_run.jsonl")
TRIALS_KEY = "invocation_medians_ms"


def good_row(i: int) -> dict:
    return {
        "kernel_hash": f"{i:016x}",
        TRIALS_KEY: [0.0025 + i * 1e-5, 0.0026 + i * 1e-5],
        "median_runtime_ms": 0.0025 + i * 1e-5,
        "measured_at": f"2026-09-28T00:{41 + i // 60:02d}:{i % 60:02d}-0700",
    }


class TestFabricatedArtifactIsRefused(unittest.TestCase):
    """The real file, used as the fixture."""

    def rows(self):
        if not FABRICATED.exists():
            self.skipTest("fabricated artifact not present on this host")
        return [json.loads(l) for l in FABRICATED.read_text().splitlines()
                if l.strip()]

    def test_it_is_refused(self):
        with self.assertRaises(ThroughputEvidenceError):
            throughput_from_wall(self.rows(), 0.15, "fabricated")

    def test_refused_for_the_fabrication_not_a_schema_mismatch(self):
        """It must fail on the FABRICATION, not on 'no such field'.

        My first version read only 7.1's key, so the fabricated file was
        rejected as having no trials at all and the identical-trials check --
        the one that matters -- never ran. A guard that refuses for the wrong
        reason leaves its important path untested.
        """
        nonempty = [r for r in self.rows() if r.get("trials")]
        self.assertTrue(nonempty, "fixture should have some rows with trials")
        with self.assertRaises(ThroughputEvidenceError) as cm:
            throughput_from_wall(nonempty, 0.10, "fabricated")
        self.assertIn("IDENTICAL trial set", str(cm.exception))

    def test_both_schema_keys_are_read(self):
        self.assertEqual(row_trials({"trials": [1.0, 2.0]}), [1.0, 2.0])
        self.assertEqual(row_trials({TRIALS_KEY: [3.0]}), [3.0])
        self.assertEqual(row_trials({"trials": []}), [])


class TestGuards(unittest.TestCase):
    def test_no_rows_is_refused(self):
        with self.assertRaises(ThroughputEvidenceError):
            throughput_from_wall([], 10.0, "empty")

    def test_zero_wall_is_refused(self):
        """A zero wall clock yields an infinite rate, which is how a broken
        harness becomes a headline number."""
        with self.assertRaises(ThroughputEvidenceError) as cm:
            throughput_from_wall([good_row(i) for i in range(5)], 0.0, "zero")
        self.assertIn("not a duration", str(cm.exception))

    def test_rows_without_measurements_are_refused(self):
        rows = [good_row(i) for i in range(4)]
        rows[2] = {**rows[2], TRIALS_KEY: []}
        with self.assertRaises(ThroughputEvidenceError) as cm:
            throughput_from_wall(rows, 1.0, "gap")
        self.assertIn("no measured trials", str(cm.exception))

    def test_identical_trials_across_rows_are_refused(self):
        rows = [{**good_row(i), TRIALS_KEY: [5.0, 5.1, 4.9]} for i in range(6)]
        with self.assertRaises(ThroughputEvidenceError) as cm:
            throughput_from_wall(rows, 1.0, "identical")
        self.assertIn("hand-authored", str(cm.exception))

    def test_unstamped_rows_are_refused(self):
        rows = [good_row(i) for i in range(4)]
        rows[1] = {k: v for k, v in rows[1].items() if k != "measured_at"}
        with self.assertRaises(ThroughputEvidenceError):
            throughput_from_timestamps(rows)

    def test_single_timestamp_is_refused(self):
        rows = [{**good_row(i), "measured_at": "2026-09-28T00:41:00-0700"}
                for i in range(4)]
        with self.assertRaises(ThroughputEvidenceError) as cm:
            throughput_from_timestamps(rows)
        self.assertIn("zero", str(cm.exception))


class TestRealRowsWork(unittest.TestCase):
    def test_real_batch_yields_a_rate(self):
        rows = [good_row(i) for i in range(30)]
        r = throughput_from_wall(rows, 10.0, "test")
        self.assertAlmostEqual(r.examples_per_sec, 3.0)
        self.assertAlmostEqual(r.examples_per_hour, 10800.0)

    def test_disagreement_is_reported_not_averaged(self):
        """30 rows one second apart span 29s against a 10s harness wall.

        The two derivations genuinely disagree here, and the point is that the
        disagreement is REPORTED rather than averaged into a single number that
        is wrong in a new way.
        """
        rows = [good_row(i) for i in range(30)]
        a = throughput_from_wall(rows, 10.0, "harness")
        b = throughput_from_timestamps(rows)
        c = cross_check(a, b)
        self.assertGreater(c["disagreement_pct"], 50.0)
        self.assertFalse(c["agree_within_tolerance"])
        self.assertEqual(c["a_derivation"], "harness")
        self.assertIn("measured_at", c["b_derivation"])

    def test_agreement_is_reported_too(self):
        rows = [good_row(i) for i in range(30)]
        a = throughput_from_wall(rows, 29.0, "harness")
        b = throughput_from_timestamps(rows)
        self.assertTrue(cross_check(a, b)["agree_within_tolerance"])


class TestBottleneckSplit(unittest.TestCase):
    def test_overhead_is_separated_from_kernel_time(self):
        rows = [good_row(i) for i in range(20)]
        s = bottleneck_split(rows, 100.0, launches_per_row=40)
        self.assertGreater(s["overhead_fraction_of_wall"], 0.99)
        self.assertLess(s["kernel_fraction_of_wall"], 0.01)
        self.assertIn("per-EXAMPLE OVERHEAD", s["verdict"])

    def test_zero_launches_refused(self):
        with self.assertRaises(ThroughputEvidenceError):
            bottleneck_split([good_row(i) for i in range(3)], 10.0, 0)

    def test_kernel_bound_batch_is_reported_as_such(self):
        """Not every batch is overhead-dominated, and saying so matters.

        Trials are DISTINCT per row on purpose: four rows carrying the same
        [100.0, 100.0] are refused by the identical-trials guard first, which
        is correct behaviour but tests the wrong thing. My first version of
        this fixture did exactly that and errored instead of asserting.
        """
        # wall 1.0s against 4 rows x 2 trials x ~100ms = ~0.81s of kernel, so
        # the kernel IS the dominant term. My first fixture used a 10s wall and
        # got 8% kernel, which is the overhead case, not the one under test.
        rows = [{**good_row(i), TRIALS_KEY: [100.0 + i, 100.0 + i],
                 "median_runtime_ms": 100.0 + i} for i in range(4)]
        s = bottleneck_split(rows, 1.0, launches_per_row=1)
        self.assertIn("GPU-bound", s["verdict"])


class TestAgainstCommittedRealData(unittest.TestCase):
    def test_real_7_1_batch_passes_every_guard(self):
        r = subprocess.run(
            ["git", "show",
             "lab/gpu-sim-u2o.7.1:experiments/gpu-sim-u2o.7.1/measured.jsonl"],
            cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("7.1 dataset not on this ref")
        rows = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
        res = throughput_from_wall(rows, 110.52331376075745, "real")
        self.assertGreater(res.examples_per_sec, 0)
        # and nowhere near the fabricated claim of 99 ex/s
        self.assertLess(res.examples_per_sec, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
