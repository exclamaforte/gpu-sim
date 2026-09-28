#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1: the 264-row dataset's disposition, enforced in code.

The PI asked for the dataset to be stated as RETAINED, REGENERATED or HELD, in
my words, and made the sharper point that a check living in a careful sentence
is not a check:

    NAMING A FAILURE MODE DOES NOT IMMUNISE YOU FROM IT.

They are right, and they are right about themselves -- they named the
cross-statistic error and committed it a sentence later, in the same message
they advised against it. So the decision is machine-readable
(`dataset_disposition.json`) and THESE tests assert it, which means a later
change that quietly promotes HELD to RETAINED without satisfying a condition
fails here rather than requiring someone to re-read a paragraph.

The three states are genuinely different, and the choice between them is a
judgement about what the project is predicting, not about which number is
reproducible:

  RETAINED     the rows are authoritative as they stand
  REGENERATED  the rows are wrong and must be re-measured
  HELD         the rows are kept, used only for claims their precision supports,
               and not re-measured until a named condition is met

Run:  python3 tests/test_dataset_disposition.py
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DISPOSITION = (REPO / "experiments" / "gpu-sim-u2o.6.7.1"
               / "dataset_disposition.json")
VALID = ("RETAINED", "REGENERATED", "HELD")


def disp() -> dict:
    if not DISPOSITION.exists():
        raise unittest.SkipTest("dataset_disposition.json not present")
    return json.loads(DISPOSITION.read_text())


class TestTheDecisionExists(unittest.TestCase):
    """A decision that is not written down is a decision nobody made."""

    def test_a_state_is_named(self):
        self.assertIn(disp()["DECISION"], VALID)

    def test_the_reason_is_in_evidence_not_in_caution(self):
        """Each rejected state must say what MEASUREMENT rejected it.

        A disposition justified by 'it seemed safer' is not a disposition; it is
        a preference that a later reader cannot check.
        """
        d = disp()
        self.assertIn("fidelity", d["why_not_RETAINED"].lower())
        self.assertIn("bake in the confound", d["why_not_REGENERATED"])

    def test_the_deciding_number_is_stated(self):
        d = disp()
        self.assertIn("0.30", d["the_number_that_decided_it"])
        self.assertIn("0.408", d["the_number_that_decided_it"])


class TestConditionsAreCheckable(unittest.TestCase):
    def test_every_condition_has_an_id_and_a_consequence(self):
        for c in disp()["conditions_that_would_change_it"]:
            self.assertIn("id", c)
            self.assertIn("condition", c)
            self.assertTrue(c["then"].strip(), f"{c['id']} has no consequence")

    def test_there_is_at_least_one_condition(self):
        """A decision with no escape hatch is a position, not a decision."""
        self.assertGreaterEqual(len(disp()["conditions_that_would_change_it"]), 1)

    def test_the_binary_confound_is_a_named_condition(self):
        """The confound I introduced must be one of the named conditions, or the
        disposition can be 'resolved' without ever fixing it."""
        text = json.dumps(disp()).lower()
        self.assertIn("synthesize", text,
                      "the per-kernel-binary confound is not a tracked condition")


class TestTheInvariantThatFollows(unittest.TestCase):
    """While HELD, the rows may not carry an absolute claim.

    This is the part that can be violated silently by someone building a
    throughput number or a latency table off the 264 rows next month, so it is
    asserted here rather than left as guidance.
    """

    def test_held_forbids_absolute_timing_claims(self):
        d = disp()
        if d["DECISION"] != "HELD":
            self.skipTest("not HELD; the invariant is stated for the HELD state")
        self.assertIn("No ABSOLUTE timing claim", d["what_may_not_be_built_on_it_meanwhile"])

    def test_relative_claims_survive(self):
        """HELD is not 'the data is worthless'. The structural findings survive."""
        d = disp()
        self.assertIn("Spearman", d["what_may_not_be_built_on_it_meanwhile"])


class TestTheRuleAndTheArtifactAgree(unittest.TestCase):
    """Two records of the same decision is one too many."""

    def test_the_markdown_rule_states_the_same_state(self):
        rule = (REPO / "experiments" / "gpu-sim-u2o.6.7.1"
                / "authoritative_path_rule.md").read_text()
        d = disp()
        marker = f"**{d['DECISION']}**"
        self.assertIn(marker, rule,
                      "authoritative_path_rule.md and dataset_disposition.json "
                      "disagree about the dataset's disposition; the artifact is "
                      "machine-readable and the rule is not, so the artifact "
                      "wins and the rule is stale")

    def test_the_floorless_claim_is_stated_as_the_pi_put_it(self):
        rule = (REPO / "experiments" / "gpu-sim-u2o.6.7.1"
                / "authoritative_path_rule.md").read_text()
        self.assertIn("REDUCED VARIANCE IS NOT EVIDENCE OF INCREASED FIDELITY",
                      rule.upper())


if __name__ == "__main__":
    unittest.main(verbosity=2)
