#!/usr/bin/env python3
"""The fifth mechanism must stay a NAMED UNKNOWN, gpu-sim-u2o.6.1.

The PI's ruling, which this test enforces:

> The right disposition is to record 'a fifth mechanism is unidentified' AS the
> finding, and not to reach for a sixth candidate. A named unknown tells the next
> person exactly where the search stopped; an invented mechanism tells them the
> search is finished when it is not.

A named unknown is a fragile artifact. It is the kind of record that gets
"improved" by someone who finds a plausible story, and the plausible story is
indistinguishable from a finding until someone tries to discriminate it. So:

  * the unknown must be present and explicitly labelled unidentified;
  * the hypotheses that ARE eliminated must still be marked eliminated, so a
    future round cannot quietly reopen one;
  * the discriminating measurement must still be named, because an unknown with
    no next step decays into an excuse;
  * and the noise floor the decision was made against must be MEASURED, not a
    round number, so "just outside" stays checkable.

Run:  python3 tests/test_fifth_mechanism.py
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ARTIFACT = (REPO / "experiments" / "gpu-sim-u2o.6.7.1"
            / "fifth_mechanism_and_decision.json")


def art() -> dict:
    if not ARTIFACT.exists():
        raise unittest.SkipTest("fifth_mechanism_and_decision.json not present")
    return json.loads(ARTIFACT.read_text())


class TestTheUnknownIsRecordedAsAnUnknown(unittest.TestCase):
    def test_the_finding_is_labelled_unidentified(self):
        f = art()["FINDING_A_fifth_mechanism_UNIDENTIFIED"]
        self.assertIn("UNIDENTIFIED", f["statement"])
        self.assertTrue(f["what_REMAINS"].strip())

    def test_no_sixth_candidate_was_invented(self):
        """A plausible story is the failure mode this test exists to catch."""
        f = art()["FINDING_A_fifth_mechanism_UNIDENTIFIED"]
        self.assertIn("explicitly_NOT_reached_for", f)
        self.assertIn("sixth", f["explicitly_NOT_reached_for"].lower())

    def test_every_eliminated_hypothesis_is_still_marked(self):
        f = art()["FINDING_A_fifth_mechanism_UNIDENTIFIED"]
        verdicts = {h["hypothesis"]: h["verdict"] for h in f["what_is_ELIMINATED_or_REDUCED"]}
        for h, v in (("warm state between examples", "REFUTED"),
                     ("the tolerance rule", "REFUTED"),
                     ("process isolation", "REFUTED")):
            self.assertIn(h, verdicts, f"{h} vanished from the record")
            self.assertEqual( verdicts[h], v,
                             f"{h} was reclassified; a later round reopened a "
                             f"refuted hypothesis without saying so")
        self.assertIn("the executable and its memory environment", verdicts)
        self.assertIn("QUARTER", verdicts["the executable and its memory environment"])


class TestTheUnknownStillHasANextStep(unittest.TestCase):
    def test_the_discriminating_measurement_is_named(self):
        m = art()["THE_MISSING_CELL"]
        self.assertIsNone(m["MISSING"]["result"],
                          "the missing cell has a result, so it is no longer "
                          "missing -- update this test rather than leave it")
        self.assertIn("7.1's per-kernel binaries", m["MISSING"]["executable"])
        self.assertTrue(m["cost"].strip())

    def test_the_discriminating_measurement_is_actually_discriminating(self):
        """It has to separate two NAMED alternatives, not just be 'more data'."""
        m = art()["THE_MISSING_CELL"]
        self.assertIn("shared context", m["why_it_is_the_discriminating_measurement"])
        self.assertIn("run_kernel", m["why_it_is_the_discriminating_measurement"])

    def test_the_three_filled_cells_are_recorded(self):
        m = art()["THE_MISSING_CELL"]
        self.assertEqual(len(m["filled"]), 3)
        for c in m["filled"]:
            self.assertIsNotNone(c["result"])


class TestTheDecisionIsAgainstAMeasuredFloor(unittest.TestCase):
    def test_the_noise_floor_is_measured_not_assumed(self):
        d = art()["DECISION_is_9_4_pct_small_enough_to_proceed"]
        floor = d["the_noise_floor_it_must_clear"]
        self.assertGreater(floor["value_7_1_rows"], 0.0)
        self.assertGreater(floor["value_6_6_1_priced"], 0.0)

    def test_the_ratio_is_recomputed_not_asserted(self):
        """`ratio_to_noise_floor` must be the quotient, so a stale one fails."""
        d = art()["DECISION_is_9_4_pct_small_enough_to_proceed"]
        expected = d["the_residual"] / d["the_noise_floor_it_must_clear"]["value_7_1_rows"]
        self.assertAlmostEqual(d["ratio_to_noise_floor"], expected, places=2)

    def test_claims_are_split_by_type(self):
        """A disposition that does not say WHICH claims may proceed is a
        disposition nobody can act on."""
        d = art()["DECISION_is_9_4_pct_small_enough_to_proceed"]
        self.assertTrue(d["proceed_for"])
        self.assertTrue(d["do_not_proceed_for"])


class TestTheCostReframingIsCarried(unittest.TestCase):
    def test_the_build_cost_finding_is_present_with_its_arithmetic(self):
        b = art()["FINDING_B_the_dataset_was_never_expensive_to_regenerate"]
        a = b["arithmetic"]
        self.assertIn("BUILD cost", b["statement"])
        self.assertGreater(a["per_kernel_binaries_one_context"]["ex_per_sec"],
                           100 * a["7_1_single_run"]["ex_per_sec"],
                           "the reframing rests on the generation being much "
                           "faster; if that stopped being true the claim fails")
        self.assertLess(a["7_1_single_run"]["wall_sec"] / 1000.0,
                        a["per_kernel_binaries_one_context"]["wall_sec"],
                        "7.1's wall clock is not actually dominated by the "
                        "per-example overhead, so the reframing would be wrong")


if __name__ == "__main__":
    unittest.main(verbosity=2)
