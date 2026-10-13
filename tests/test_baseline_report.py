"""gpu-sim-u2o.9.4: the report's claims hold against committed paths.

A report bead's failure mode is prose drifting from numbers. These pin the
load-bearing claims: the arc reconciliation, the contention census, the
cancellation statement, and the per-regime table. CPU-only.
"""
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "gpu-sim-u2o.9.4"


def metrics():
    p = EXP / "metrics.json"
    if not p.exists():
        raise unittest.SkipTest("run report_verify.py first")
    return json.loads(p.read_text())


class TestArcReconciliation(unittest.TestCase):
    def test_arc_numbers_match_committed_beads(self):
        m = metrics()
        self.assertEqual(m["arc"]["g8x_before"], 7.7238)
        self.assertEqual(m["arc"]["zhq_after"], 0.7258)
        self.assertEqual(m["arc"]["p46_after"], 0.6581)
        self.assertEqual(m["arc"]["p46_bias_after"], -0.0245)
        self.assertEqual(m["arc"]["u83_distinct_corrected"], 456)
        self.assertEqual(m["arc"]["u83_item5_groups"], 0)

    def test_overall_is_117x_not_a_typo(self):
        m = metrics()
        self.assertAlmostEqual(
            m["arc"]["g8x_before"] / m["arc"]["p46_after"], 11.7, places=1)


class TestContentionBound(unittest.TestCase):
    def test_all_corpus_cells_uncontended(self):
        n, bad = 0, []
        for c in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
                  "experiments/gpu-sim-e5q/measured.jsonl"):
            for line in (REPO / c).read_text().splitlines():
                r = json.loads(line)
                n += 1
                q = r["queue_state"]
                if q["n_other_running"] != 0 or q["n_other_active"] != 0:
                    bad.append(r["kernel_hash"])
        self.assertEqual(n, 456)
        self.assertEqual(bad, [])

    def test_report_states_anchor_side_unknown(self):
        text = (EXP / "REPORT.md").read_text()
        self.assertIn("records NO", text)
        self.assertIn("stated, not assumed", text)


class TestCancellationStated(unittest.TestCase):
    def test_overall_bias_not_reported_as_accuracy(self):
        text = (EXP / "REPORT.md").read_text()
        self.assertIn("cancellation, not accuracy", text)
        self.assertIn("-0.0245", text)

    def test_per_regime_signs_oppose(self):
        m = metrics()
        reg = {k: v[1] for k, v in m["arc"]["p46_regimes"].items()}
        self.assertLess(reg["latency->latency"], 0)
        self.assertGreater(reg["latency->memory"], 0)
        self.assertGreater(reg["memory->memory"], 0)

    def test_families_are_single_kernel(self):
        m = metrics()
        for op, v in m["families"].items():
            self.assertEqual(v["n_rows"], 57, op)
            self.assertEqual(v["n_kernels"], 1, op)

    def test_nothing_excluded_silently(self):
        self.assertEqual(metrics()["excluded_rows"], 0)


if __name__ == "__main__":
    unittest.main()
