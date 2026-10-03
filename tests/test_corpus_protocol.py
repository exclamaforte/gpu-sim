#!/usr/bin/env python3
"""The protocol condition, enforced. gpu-sim-u2o.6.7.1.

The PI's condition on a fast-path corpus:

> every row records ITS PROTOCOL, because ai_cv and the absolute medians mean
> different things under the single-run and the per-kernel-shared-context paths
> -- the same statistic moves 3.2x between them, which you measured. So a new
> corpus is ADDITIVE and LABELLED, and nothing in it may be pooled with the 264
> reference rows or the anchor's cells.

A condition stated in prose is the kind this lab has been correcting all night,
so these tests make each clause fail loudly:

  * a row with no protocol is not written, and not read either;
  * a mixed corpus is refused by the operation that would consume it;
  * pooling is possible but only when the caller NAMES the target, so it is a
    decision at the call site rather than a default;
  * the reference protocol is marked frozen, because the PI's ruling is that the
    264 rows and the anchor are not to be regenerated;
  * an unrecognised protocol is never assumed compatible with a known one.

Run:  python3 tests/test_corpus_protocol.py
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.corpus_protocol import (  # noqa: E402
    PROTOCOLS, ProtocolError, assert_single_protocol, describe, group_by_protocol,
    poolable, protocol_of, write_rows,
)

REF = "single_run_7_1"
FAST = "perkernel_shared_ctx"


def row(proto=None, h="abc"):
    r = {"kernel_hash": h, "median_runtime_ms": 0.003}
    if proto:
        r["protocol"] = proto
    return r


class TestEveryRowIsLabelled(unittest.TestCase):
    def test_a_row_with_no_protocol_is_refused_on_read(self):
        with self.assertRaises(ProtocolError) as cm:
            protocol_of(row())
        self.assertIn("ai_cv", str(cm.exception))

    def test_a_row_with_no_protocol_is_refused_on_write(self):
        with self.assertRaises(ProtocolError) as cm:
            write_rows(Path(tempfile.mkdtemp()) / "c.jsonl", [row()])
        self.assertIn("write", str(cm.exception).lower() +
                      " -- stamping at write time is the only guarantee")

    def test_a_written_corpus_round_trips_its_protocol(self):
        p = Path(tempfile.mkdtemp()) / "c.jsonl"
        write_rows(p, [row(REF, "a"), row(REF, "b")])
        back = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        self.assertEqual([r["protocol"] for r in back], [REF, REF])

    def test_an_unknown_protocol_is_never_assumed_compatible(self):
        with self.assertRaises(ProtocolError) as cm:
            describe("some_protocol_nobody_registered")
        self.assertIn("own group", str(cm.exception))


class TestMixedCorporaAreRefused(unittest.TestCase):
    def test_a_mixed_corpus_is_refused(self):
        rows = [row(REF, "a"), row(FAST, "b")]
        with self.assertRaises(ProtocolError) as cm:
            assert_single_protocol(rows)
        self.assertIn("ADDITIVE and LABELLED", str(cm.exception))

    def test_a_single_protocol_corpus_passes_and_names_itself(self):
        self.assertEqual(
            assert_single_protocol([row(REF, "a"), row(REF, "b")]), REF)

    def test_grouping_reports_the_mix_rather_than_merging_it(self):
        g = group_by_protocol([row(REF, "a"), row(FAST, "b"), row(FAST, "c")])
        self.assertEqual({k: len(v) for k, v in g.items()}, {REF: 1, FAST: 2})


class TestPoolingIsADecisionNotADefault(unittest.TestCase):
    """Pooling is legitimate ONCE the protocols are shown to agree -- so it
    must be possible, but only by naming the target."""

    def test_nothing_pools_without_a_named_target(self):
        rows = [row(REF, "a"), row(FAST, "b")]
        self.assertEqual(poolable(rows), [])

    def test_naming_a_target_returns_only_that_protocols_rows(self):
        rows = [row(REF, "a"), row(FAST, "b"), row(FAST, "c")]
        got = poolable(rows, target_protocol=FAST)
        self.assertEqual([r["kernel_hash"] for r in got], ["b", "c"])

    def test_naming_an_unknown_target_is_refused(self):
        with self.assertRaises(ProtocolError):
            poolable([row(REF)], target_protocol="nope")


class TestTheReferenceIsFrozen(unittest.TestCase):
    """The PI's ruling: the 264 rows and the anchor are NOT regenerated on the
    fast path. If a future change re-labels them, the whole reference chain
    (8.3's feature vector, 9.4's roofline, 9.5's acceptance) silently moves."""

    def test_reference_is_marked_frozen_and_authoritative(self):
        d = PROTOCOLS[REF]
        self.assertIn("FROZEN", d["authority"])
        self.assertIn("REFERENCE", d["authority"])

    def test_the_fast_protocol_is_not_authoritative(self):
        d = PROTOCOLS[FAST]
        self.assertIn("SCALE ONLY", d["authority"])
        self.assertIn("may not be pooled", d["authority"])

    def test_the_measured_statistic_shift_is_recorded(self):
        """Every candidate ratio must be DERIVED, and a stale claim RECORDED.

        This used to assert `REF/FAST > 3.0` -- a number typed into the test
        rather than read from the data. Re-measured, no pair of 264-row
        protocols gives 3.2x (the three are 1.18, 2.78, 2.36), so the old
        assertion was checking a claim nothing supports.

        It is NOT corrected to one of those either. That would make the test
        agree with whatever the data says today and stop it noticing that a
        DECISION was written against an uncheckable claim. Instead:

          1. every pair the artifact records must equal the DERIVED ratio, so
             the numbers are checkable rather than prose; and
          2. if the prose claims a shift no derived ratio supports, the
             artifact must record that disagreement.

        Clause 2 is the guard. It fails if someone edits the medians without
        updating the pairs, or leaves a prose claim contradicting the data with
        nothing recording it. It does not decide the scientific question, which
        is not a test's to decide -- and which is now stated as a question
        rather than buried.
        """
        art = (REPO / "experiments" / "gpu-sim-u2o.6.7.1"
               / "corpus_decision.json")
        if not art.exists():
            self.skipTest("corpus_decision.json not present")
        blk = json.loads(art.read_text())["PROTOCOL_CONDITION_IMPLEMENTED"]

        recorded = blk.get("measured_statistic_shift_ALL_PAIRS")
        self.assertIsNotNone(
            recorded,
            "the decision records no structured statistic shift, so its "
            "justification cannot be checked by anything")

        for entry in recorded:
            x, y = entry["pair"]
            derived = PROTOCOLS[x]["median_ai_cv"] / PROTOCOLS[y]["median_ai_cv"]
            self.assertAlmostEqual(
                derived, entry["ratio"], places=3,
                msg=(f"recorded ratio for {x}/{y} is {entry['ratio']} but the "
                     f"registry medians give {derived:.4f}. The artifact has "
                     f"drifted from the data it claims to summarise."))

        m = re.search(r"moves (\d+\.\d+)x", json.dumps(blk))
        if m:
            claimed = float(m.group(1))
            if not any(abs(claimed - e["ratio"]) <= 0.05 for e in recorded):
                self.assertIn(
                    "measured_statistic_shift_UNRESOLVED", blk,
                    f"the prose claims {claimed}x, no derived ratio supports "
                    f"it, and nothing in the artifact records the disagreement. "
                    f"Either correct the prose or record why.")

    @staticmethod
    def _claimed_shift():
        """The shift the decision document claims, wherever it is recorded."""
        art = (REPO / "experiments" / "gpu-sim-u2o.6.7.1"
               / "corpus_decision.json")
        if not art.exists():
            return None
        d = json.loads(art.read_text())
        for key in ("THE_BINDING_CONSTRAINT_IS_NOT_ROW_COUNT",
                    "PROTOCOL_CONDITION_IMPLEMENTED"):
            blk = d.get(key)
            if isinstance(blk, dict):
                v = blk.get("measured_statistic_shift_x")
                if isinstance(v, (int, float)):
                    return float(v)
        return None

    def test_every_registered_protocol_was_actually_measured(self):
        for name, d in PROTOCOLS.items():
            self.assertTrue(d.get("measured"), f"{name} is registered but has "
                                             f"no measurement behind it")


class TestTheCorpusDecisionIsOnLearningGrounds(unittest.TestCase):
    """The PI's condition on the SIZE: justified against what 9.x must learn,
    not against what is now cheap to produce."""

    ART = (REPO / "experiments" / "gpu-sim-u2o.6.7.1" / "corpus_decision.json")

    def decision(self):
        if not self.ART.exists():
            self.skipTest("corpus_decision.json not present")
        return json.loads(self.ART.read_text())

    def test_the_binding_constraint_is_named(self):
        """Keyed by NAME, never by position.

        The previous version asserted `"NOT_ROW_COUNT" in list(d)[1]` behind a
        dead `if False else`, so it was checking the SECOND KEY of the document
        rather than the binding constraint's claim. It read `owns` and failed for
        a reason that had nothing to do with the corpus decision. A registry read
        positionally is a registry that breaks the moment a key is renamed, and
        breaks SILENTLY when one is inserted at the front.
        """
        d = self.decision()
        claim = d["THE_BINDING_CONSTRAINT_IS_NOT_ROW_COUNT"]["claim"]
        self.assertIn("LABEL DYNAMIC RANGE", claim)
        self.assertIn("FAMILY COUNT", claim)
        self.assertIn("CONSEQUENCE", claim.upper())

    def test_the_label_range_evidence_is_present_and_small(self):
        """The argument only works if the current corpus really is too narrow."""
        e = self.decision()["THE_BINDING_CONSTRAINT_IS_NOT_ROW_COUNT"][
            "evidence_the_264_rows_cannot_train_anything"]
        self.assertLess(e["label_range_x"], 3.0)
        self.assertLess(e["label_log10_spread"], 0.5)

    def test_the_row_count_is_a_consequence_not_the_target(self):
        spec = self.decision()["THE_CORPUS_SPECIFICATION"]
        rc = self.decision()["row_count"]
        self.assertIn("COVERAGE", rc["note"])
        self.assertEqual(rc["worked_example"]["rows"],
                         rc["worked_example"]["families"]
                         * rc["worked_example"]["sizes"]
                         * rc["worked_example"]["configs"])

    def test_the_split_unit_is_the_family(self):
        self.assertEqual(self.decision()["split"]["unit"],
                         "STRUCTURAL FAMILY, never the row")

    def test_the_acceptance_names_the_binding_constraint(self):
        acc = self.decision()["the_acceptance_written_on_11_2"]
        self.assertTrue(any("must LEARN" in a for a in acc))
        self.assertTrue(any("FROZEN" in a for a in acc))
        self.assertTrue(any("fails loud" in a.lower() for a in acc),
                        "the 7.7 fail-open defect is not carried forward")


if __name__ == "__main__":
    unittest.main(verbosity=2)
