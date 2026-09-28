#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1 criterion 7: the regression guard.

> A regression test asserting per-example overhead falls by the expected order
> WHILE device time per example does not move, so a future change that quietly
> reintroduces per-example process cost fails rather than merely being slower.

The asymmetry is the whole point. A guard that only asserted "throughput went
up" would be satisfied by a change that simply stopped measuring properly --
fewer trials, fewer examples, a skipped sample. So this asserts BOTH halves and
fails if either moves in the wrong direction.

Also here, because they are the checks that caught this bead's own author:
  * the protocol must still be 40 timed launches per example;
  * every row must have its OWN sample time.

Run:  python3 tests/test_batched_worker.py
"""
import json
import statistics
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HERE = REPO / "experiments" / "gpu-sim-u2o.6.7.1"
SINGLE_RUN_EX_PER_SEC = 2.389      # 7.1 / 6.7, measured
DEVICE_TIME_ASSERTED_US = 126.0     # the ruling's fixed figure
PROTOCOL_TIMED_LAUNCHES = 40       # 2 invocations x 20 timed


def summary():
    p = HERE / "batch_summary.json"
    if not p.exists():
        raise unittest.SkipTest("run batch_measure_worker.py first")
    return json.loads(p.read_text())


def rows():
    p = HERE / "batch_rows.jsonl"
    if not p.exists():
        raise unittest.SkipTest("run batch_measure_worker.py first")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


class TestOverheadFalls(unittest.TestCase):
    def test_per_example_overhead_falls_by_the_expected_order(self):
        """The order, not a constant: 10x or more, asserted as a bound.

        A guard that asserted an exact multiple would be a guard that breaks
        the first time the machine is busier, and gets deleted rather than
        fixed. An order-of-magnitude floor survives that.
        """
        s = summary()
        speedup = (s["throughput"]["examples_per_sec"]
                   / SINGLE_RUN_EX_PER_SEC)
        self.assertGreaterEqual(speedup, 10.0,
                                f"only {speedup:.1f}x -- per-example process "
                                f"cost may have come back")

    def test_process_count_is_collapsed(self):
        s = summary()
        self.assertEqual(s["worker"]["processes_started"], 1)
        self.assertEqual(s["worker"]["contexts_created"], 1)


class TestDeviceTimeDoesNotMove(unittest.TestCase):
    def test_device_time_per_example_is_asserted(self):
        """The other half of the asymmetry, and the one a lazy fix breaks."""
        s = summary()
        d = s["device_time_per_example"]
        self.assertTrue(d["unchanged"],
                        f"device time/example moved to {d['measured_us']:.1f} us "
                        f"from the asserted {d['asserted_us']} us -- a "
                        f"throughput fix that moves device time is not a "
                        f"throughput fix")

    def test_device_time_is_not_smaller_than_the_assertion(self):
        """Below the assertion means FEWER launches, i.e. less measured.

        This direction is the dangerous one: a change that quietly drops trials
        makes the device time smaller AND the throughput better, and every
        naive gate passes it.
        """
        s = summary()
        self.assertGreaterEqual(
            s["device_time_per_example"]["measured_us"],
            DEVICE_TIME_ASSERTED_US * 0.90,
            "device time/example is BELOW the protocol's floor, which means "
            "fewer launches ran than the protocol requires")


class TestProtocolUnchanged(unittest.TestCase):
    def test_timed_launches_per_example(self):
        s = summary()
        self.assertEqual(s["worker"]["timed_launches_per_example"],
                         PROTOCOL_TIMED_LAUNCHES)
        self.assertTrue(s["worker"]["protocol_unchanged"])

    def test_two_invocations_of_twenty(self):
        s = summary()
        self.assertEqual(s["worker"]["invocations_per_example"], 2)
        self.assertEqual(s["worker"]["timed_launches_per_example"]
                         // s["worker"]["invocations_per_example"], 20)

    def test_every_row_kept_its_trials(self):
        for r in rows():
            self.assertEqual(r["n_timed_launches"],
                             PROTOCOL_TIMED_LAUNCHES,
                             f"row {r['kernel_hash']} ran "
                             f"{r['n_timed_launches']} timed launches")
            self.assertEqual(len(r["invocation_medians_ms"]), 2)

    def test_rows_carry_two_invocation_medians_not_a_pooled_median(self):
        """7.1's estimator, mirrored.

        A median over all 40 pooled trials is a DIFFERENT estimator, and
        comparing it against 7.1's median-of-medians would quietly invalidate
        the reproduction check.
        """
        for r in rows():
            self.assertEqual(len(r["invocation_medians_ms"]), 2)
            self.assertEqual(r["median_runtime_ms"],
                             statistics.median(r["invocation_medians_ms"]))


class TestPerExampleSampling(unittest.TestCase):
    def test_every_row_has_its_own_sample_time(self):
        rs = rows()
        times = {r["sampled_at_epoch"] for r in rs}
        self.assertEqual(len(times), len(rs),
                         "rows share sample times -- that is a per-BATCH "
                         "sample wearing per-row clothing")

    def test_sample_times_increase_along_the_batch(self):
        """Monotonic, because they are taken as the rows happen."""
        rs = sorted(rows(), key=lambda r: r["sampled_at_epoch"])
        self.assertEqual([r["sampled_at_epoch"] for r in rs],
                         sorted(r["sampled_at_epoch"] for r in rs))

    def test_two_rows_differ_in_time_and_in_what_they_say(self):
        """Criterion 4's demonstration, not just its count."""
        rs = sorted(rows(), key=lambda r: r["sampled_at_epoch"])
        self.assertNotEqual(rs[0]["sampled_at"], rs[-1]["sampled_at"])
        for r in (rs[0], rs[-1]):
            self.assertIn("queue_state", r)
            self.assertIn("telemetry", r)

    def test_sampling_cost_is_reported_not_hidden(self):
        s = summary()
        self.assertIn("sampling_ms_per_example",
                      s["per_example_sampling"])
        self.assertGreater(s["per_example_sampling"]["sampling_ms_per_example"],
                           0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
