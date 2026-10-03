#!/usr/bin/env python3
"""gpu-sim-edu.1: the dispatch filter must hide finished work AND must be able to pass.

Criterion 3 wants a fixture holding at least one validated-unmerged bead and at
least one dep-blocked bead, with neither offered. Criterion 2 wants dep-blocked
beads absent from the real list.

The requirement that shapes everything else here comes from gpu-sim-u2o.6.11.1:
A CHECK THAT ONLY EVER REPORTS DISAGREEMENT IS INDISTINGUISHABLE FROM ONE THAT
HAD THE ACCOUNTING WRONG. A filter that dropped every bead would satisfy every
exclusion assertion in this file and be useless in production. So there are
tests that assert ordinary ready work SURVIVES, and an integration test that
fails if the filter ever empties the list.

Run:  python3 tests/test_bd_ready_filter.py
"""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "bd_ready_filtered", REPO / "scripts" / "bd_ready_filtered.py")
brf = importlib.util.module_from_spec(_spec)
sys.modules["bd_ready_filtered"] = brf
_spec.loader.exec_module(brf)


def issue(**kw):
    base = {"id": "x", "title": "t", "priority": 0, "status": "open",
            "labels": [], "dependencies": [], "created_at": "2026-01-01T00:00:00Z"}
    base.update(kw)
    return base


def dep(kind, other):
    return {"issue_id": "x", "depends_on_id": other, "type": kind}


class TestFixtureExclusions(unittest.TestCase):
    """Criterion 3, on a fixture. No database, no GPU, no network."""

    def setUp(self):
        self.status = {"blocker-open": "open", "blocker-done": "closed"}
        self.fixture = [
            # (1) validated but unmerged -- finished, must not be offered
            issue(id="validated-unmerged", labels=["validated"]),
            issue(id="validated-pending", labels=["fab-audit-reopened",
                                                   "validated-pending-merge"]),
            # (2) blocked by an OPEN issue -- must not be offered
            issue(id="dep-blocked", dependencies=[dep("blocks", "blocker-open")]),
            # ordinary work -- MUST be offered, or this filter is a shredder
            issue(id="plain-ready", labels=["fab-audit-reopened"]),
            issue(id="needs-rework-only", labels=["needs-rework"]),
            # under review AND labelled validated: still outstanding, must show
            issue(id="in-review", labels=["validated", "needs-review"]),
            # blocked only by a CLOSED issue -- legitimately offerable
            issue(id="blocker-closed", dependencies=[dep("blocks", "blocker-done")]),
            # parent-child edges are not blockers
            issue(id="has-parent", dependencies=[dep("parent-child", "blocker-open")]),
        ]

    def classify(self):
        return brf.classify(self.fixture, self.status)

    def test_validated_unmerged_is_not_offered(self):
        offer, _ = self.classify()
        ids = {i["id"] for i in offer}
        self.assertNotIn("validated-unmerged", ids)
        self.assertNotIn("validated-pending", ids)

    def test_dep_blocked_by_open_issue_is_not_offered(self):
        offer, _ = self.classify()
        self.assertNotIn("dep-blocked", {i["id"] for i in offer})

    def test_plain_ready_work_survives(self):
        """The assertion that stops this being a shredder."""
        offer, _ = self.classify()
        ids = {i["id"] for i in offer}
        self.assertIn("plain-ready", ids)
        self.assertIn("needs-rework-only", ids)
        self.assertIn("blocker-closed", ids,
                      "a blocker that is CLOSED does not block; excluding this "
                      "would hide work that is genuinely startable")
        self.assertIn("has-parent", ids,
                      "a parent-child edge is not a blocker")

    def test_outstanding_verdict_beats_a_merge_pending_label(self):
        """A bead under review is not finished, even if it says `validated`."""
        offer, _ = self.classify()
        self.assertIn("in-review", {i["id"] for i in offer},
                      "hiding a bead whose verdict is still outstanding would "
                      "hide work in progress")

    def test_every_exclusion_states_a_reason(self):
        _, dropped = self.classify()
        for issue, why in dropped:
            self.assertTrue(why.strip(), f"{issue['id']} dropped with no reason")
        self.assertEqual(len(dropped), 3)

    def test_offered_and_dropped_partition_the_input(self):
        offer, dropped = self.classify()
        self.assertEqual(len(offer) + len(dropped), len(self.fixture))


class TestOrdering(unittest.TestCase):
    def test_priority_beats_recency(self):
        items = [issue(id="old-p0", priority=0, created_at="2020-01-01"),
                 issue(id="new-p2", priority=2, created_at="2026-01-01")]
        ordered = [i["id"] for i in sorted(items, key=brf.order_key)]
        self.assertEqual(ordered, ["old-p0", "new-p2"],
                         "recency must not outrank priority")

    def test_order_is_deterministic(self):
        items = [issue(id="b"), issue(id="a")]
        first = [i["id"] for i in sorted(items, key=brf.order_key)]
        second = [i["id"] for i in sorted(list(reversed(items)), key=brf.order_key)]
        self.assertEqual(first, second)


class TestAgainstTheRealDatabase(unittest.TestCase):
    """The real list. Skips rather than passes silently if bd is unavailable."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.ready = brf.bd_json(["ready"])
            cls.status = brf.statuses_by_id()
        except Exception as exc:
            raise unittest.SkipTest(f"bd unavailable: {exc}")

    def test_no_offered_bead_is_finished_awaiting_merge(self):
        offer, _ = brf.classify(self.ready, self.status)
        leaks = [i["id"] for i in offer if brf.is_finished_awaiting_merge(i)]
        self.assertEqual(leaks, [], f"criterion 1 violated: {leaks}")

    def test_no_offered_bead_has_an_open_blocker(self):
        offer, _ = brf.classify(self.ready, self.status)
        leaks = [i["id"] for i in offer if brf.open_blockers(i, self.status)]
        self.assertEqual(leaks, [], f"criterion 2 violated: {leaks}")

    def test_the_filter_does_not_empty_the_list(self):
        """THE CHECK MUST BE ABLE TO PASS.

        If this ever fails, the filter is not filtering -- it is deleting. That
        failure would otherwise look exactly like a clean, quiet dispatch.
        """
        offer, _ = brf.classify(self.ready, self.status)
        self.assertGreater(
            len(offer), 0,
            "the filter offered NOTHING. An empty dispatch and a correct "
            "dispatch are indistinguishable downstream.")
        if len(self.ready) > 5:
            self.assertGreaterEqual(
                len(offer), len(self.ready) * 0.5,
                f"filter removed {len(self.ready) - len(offer)} of "
                f"{len(self.ready)} ready beads; that is not a label filter, "
                f"that is a shredder")

    def test_blocked_children_of_this_beads_issue_stay_hidden(self):
        """The two concrete cases that motivated the bead, when present.

        They are dep-blocked via `blocks` edges, so bd itself already excludes
        them. Asserting it anyway is the point: the property should hold even
        if bd's semantics change, and a silent regression here is exactly the
        failure that cost two sessions.
        """
        for bid in ("gpu-sim-u2o.6.6.6.2", "gpu-sim-u2o.6.6.6.3"):
            if not any(i["id"] == bid for i in self.ready):
                continue
            self.fail(f"{bid} appeared in bd ready despite a blocks edge")


if __name__ == "__main__":
    unittest.main(verbosity=2)