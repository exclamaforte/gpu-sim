"""gpu-sim-u2o.7.1 records rework: the ledger parses and the stale hint is gone.

ITEM 1: experiments/gpu-sim-u2o.7.1/job.json was invalid JSON (a Python
parenthesised string-concat in the "reason" field). This test asserts it
parses -- not by eyeball -- and that the killed_underused ledger entry kept
its content.
ITEM 2: regime_hint shipped unlabelled next to measured_regime and was wrong
wherever it mattered. The key is REMOVED from all 264 rows; measured_regime
is the only regime field. This test pins: 264 rows, no regime_hint anywhere,
measured_regime present and launch_bound in every row.
CPU-only. No GPU, no new rows, no number touched.
"""
import json
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "experiments" / "gpu-sim-u2o.7.1"


class TestJobJsonParses(unittest.TestCase):
    def test_job_json_is_valid_json(self):
        d = json.loads((EXP / "job.json").read_text())
        self.assertEqual(d["bead"], "gpu-sim-u2o.7.1")
        killed = [j for j in d["superseded_jobs"]
                  if j.get("state") == "killed_underused"]
        self.assertEqual(len(killed), 1)
        self.assertIn("0.7%", killed[0]["reason"])
        self.assertIn("unqueued CPU phase", killed[0]["reason"])


class TestRegimeHintGone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(l) for l in
                    (EXP / "measured.jsonl").read_text().splitlines()]

    def test_264_rows(self):
        self.assertEqual(len(self.rows), 264)

    def test_no_regime_hint_in_any_row(self):
        bad = [r["kernel_hash"] for r in self.rows if "regime_hint" in r]
        self.assertEqual(bad, [], f"{len(bad)} rows still ship regime_hint")

    def test_measured_regime_authoritative_everywhere(self):
        bad = [r["kernel_hash"] for r in self.rows
               if r.get("measured_regime") != "launch_bound"]
        self.assertEqual(bad, [],
                         f"{len(bad)} rows lack launch_bound measured_regime")


if __name__ == "__main__":
    unittest.main()
