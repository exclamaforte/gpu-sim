#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.3 rework checks: the acceptance criteria name fields the
artifact must use, so assert them mechanically.

Standing lesson (6.6.6.1 scope lint, u2o.15 ancestry, and this bead's own
item 1): a specification that names a field must be checked against the
artifact before it is issued. Each test below PRINTS the keys it expects and
FAILS when one is absent, so a future reshape that drops a criterion field
goes red rather than silently unexecutable.

CPU only. No GPU, no ts job, no run_kernel.
"""
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "experiments" / "gpu-sim-u2o.6.6.3"
METRICS = ART / "metrics.json"

CONDS = ("uncontended", "queued_routed", "unqueued")

# Criterion 1's vocabulary, extended where the criterion's two-condition
# wording cannot name this bead's three conditions. key_map in each cell
# states the relation; this list states what a checker may rely on.
EXPECTED_FLAT_KEYS = (
    "median_uncontended_us",
    "median_queued_routed_us",
    "median_unqueued_us",
    "ratio_queued_routed",
    "ratio_unqueued",
    "ai_cv_uncontended",
    "ai_cv_queued_routed",
    "ai_cv_unqueued",
    "key_map",
)


class TestCriterionFieldsExist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = json.loads(METRICS.read_text())

    def test_cells_span_three_regimes(self):
        cells = self.m["cells"]
        print(f"cells present: {sorted(cells)}")
        for name in ("launch_bound", "memory_bound", "compute_bound"):
            self.assertIn(name, cells, f"regime cell {name} missing")

    def test_criterion_1_flat_keys_per_cell(self):
        for name, cell in self.m["cells"].items():
            for key in EXPECTED_FLAT_KEYS:
                self.assertIn(
                    key, cell,
                    f"cells.{name} lacks criterion key {key}; a checker "
                    f"written from criterion 1 cannot execute")
            print(f"cells.{name}: all {len(EXPECTED_FLAT_KEYS)} "
                  f"criterion keys present")

    def test_flat_keys_are_aliases_not_recomputations(self):
        for name, cell in self.m["cells"].items():
            for cond in CONDS:
                self.assertEqual(
                    cell[f"median_{cond}_us"], cell[cond]["median_us"],
                    f"cells.{name} flat median diverged from nested block")
                self.assertEqual(
                    cell[f"ai_cv_{cond}"], cell[cond]["ai_cv"],
                    f"cells.{name} flat ai_cv diverged from nested block")

    def test_criterion_4_provenance_on_every_invocation(self):
        for name, cell in self.m["cells"].items():
            for cond in CONDS:
                rows = cell[cond]["per_invocation"]
                self.assertGreater(len(rows), 0)
                for row in rows:
                    q = row.get("queue_state")
                    self.assertIsNotNone(
                        q, f"cells.{name}.{cond} invocation row has no "
                           f"queue_state; criterion 4 as written is unmet")
                    self.assertEqual(q.get("condition"), cond)
                    self.assertIsNotNone(q.get("own_job_id"))
                    self.assertIsNotNone(q.get("n_other_active"))
                    self.assertIn("overlapping_job_ids", q)

    def test_no_always_true_discriminates_boolean(self):
        blob = json.dumps(self.m["detection"])
        self.assertNotIn(
            "discriminates", blob,
            "detection still carries the always-true boolean the PI ruled "
            "out; ordering lives in relative_spread")
        spreads = {s: v["relative_spread"]
                   for s, v in self.m["detection"]["signals"].items()}
        print(f"signal spreads: {spreads}")
        self.assertLess(spreads["clocks.sm"], 0.15)

    def test_rule_floor_regime_travel_together(self):
        rule = self.m["discard_rule"]
        for key in ("drop_if", "detection_floor_travels_with_rule",
                    "regime_weighting", "dropped_clauses", "derivation"):
            self.assertIn(key, rule, f"discard_rule lacks {key}")
        drops = " ".join(rule["drop_if"])
        for sig in ("power.draw", "utilization.gpu", "temperature.gpu"):
            self.assertIn(sig, drops)
        self.assertNotIn("clocks.sm", drops,
                         "the dropped clocks.sm clause is back in drop_if")
        self.assertIn("clocks.sm", " ".join(rule["dropped_clauses"]),
                      "the drop of the clocks.sm clause must be recorded, "
                      "not silent")
        rw = rule["regime_weighting"]
        self.assertIn("memory_bound", rw)
        self.assertIn("launch_bound", rw)

    def test_rework_provenance_names_no_remeasurement(self):
        prov = self.m.get("rework_provenance", {})
        self.assertIn("20260928t072008-b34ddec9", prov.get("ts_job", ""),
                      "artifact must name the ts job its numbers came from")
        self.assertIn("superseded_20260928_rule", self.m,
                      "the superseded rule must be preserved for audit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
