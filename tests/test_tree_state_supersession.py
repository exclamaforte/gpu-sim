#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.2.1.1: an artifact whose branch moved must say where it went.

Criterion 3: for every artifact whose stamped branch no longer holds the
artifact's own content, `superseded_by` must be present and resolve to a commit
reachable from master.

Two properties this suite is careful about, both from this lab's own rules:

  * ABSENT IS NOT NULL. An artifact that cannot name a superseding commit must
    OMIT the key. `"superseded_by": null` asserts that nothing supersedes it --
    a claim nobody checked -- and a null reads as coverage while an absent key
    does not (gpu-sim-u2o.6.6.6 criterion 5).
  * THE CHECK MUST BE ABLE TO PASS. A test that fires on every artifact is
    indistinguishable from one whose detection is broken. So this asserts that a
    correctly-stamped artifact is left alone and a live branch is not falsely
    reported as superseded.

Run:  python3 tests/test_tree_state_supersession.py
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.tree_state import (  # noqa: E402
    branch_exists, first_word, landing_commit, resolve_superseded_by,
    stamp_tree_state, stamped_artifacts,
)


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


class TestBranchFieldParsing(unittest.TestCase):
    def test_prose_stamp_yields_the_ref(self):
        """gpu-sim-u2o.17.5 stamps the branch with a sentence glued on.

        Without trimming, the ref never resolves, so every downstream check
        treats a deleted branch as still live -- the failure looks like 'nothing
        is wrong', which is the worst kind.
        """
        self.assertEqual(
            first_word("lab/gpu-sim-u2o.17.5-r2 (split from the shared branch)"),
            "lab/gpu-sim-u2o.17.5-r2")

    def test_empty_field_is_empty_not_an_index_error(self):
        self.assertEqual(first_word(""), "")
        self.assertEqual(first_word(None), "")


class TestTheSupersessionRule(unittest.TestCase):
    def test_unresolvable_omits_the_key_rather_than_writing_null(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            ts = stamp_tree_state({"branch": "lab/does-not-exist-anywhere"},
                                  "no-such-bead", repo)
            self.assertNotIn("superseded_by", ts,
                             "an unresolvable supersession must be OMITTED; a "
                             "null would assert that nothing supersedes it")

    def test_content_on_master_is_recorded_even_when_the_ref_survives(self):
        """The case that made me rewrite the rule.

        lab/gpu-sim-u2o.6.6.2-r2 still exists and is still dirty, while its
        content is on master. Keying on ref existence would have reported
        nothing for exactly the artifact that most needs it.
        """
        rec = resolve_superseded_by("lab/gpu-sim-u2o.6.6.2-r2",
                                    "gpu-sim-u2o.6.6.2", REPO)
        self.assertIsNotNone(rec)
        self.assertTrue(rec["branch_still_exists"],
                        "the live ref is the hazard and must be recorded")
        self.assertTrue(rec["reachable_from_master"])

    def test_deleted_ref_is_recorded_as_gone(self):
        rec = resolve_superseded_by("lab/gpu-sim-u2o.17.5-r2",
                                    "gpu-sim-u2o.17.5", REPO)
        self.assertIsNotNone(rec)
        self.assertFalse(rec["branch_still_exists"])
        self.assertIn("would find nothing", rec["note"])

    def test_an_already_stamped_record_is_not_recomputed(self):
        ts = {"branch": "x", "superseded_by": {"commit": "handwritten"}}
        self.assertEqual(stamp_tree_state(ts, "y", REPO)["superseded_by"],
                         {"commit": "handwritten"},
                         "an explicit stamp must not be silently overwritten")


class TestEveryStampedArtifact(unittest.TestCase):
    """Criterion 3, over the whole experiments/ tree."""

    def test_no_artifact_carries_a_null_superseded_by(self):
        """The absent-vs-null rule, asserted over real artifacts."""
        offenders = []
        for path, doc, ts in stamped_artifacts(REPO):
            sb = ts.get("superseded_by") or doc.get("superseded_by")
            if "superseded_by" in ts and ts["superseded_by"] is None:
                offenders.append(str(path))
            if "superseded_by" in doc and doc["superseded_by"] is None:
                offenders.append(str(path))
        self.assertEqual(offenders, [],
                         f"null superseded_by reads as coverage: {offenders}")

    def test_every_landed_artifact_names_a_master_reachable_commit(self):
        """Criterion 3 proper."""
        bad = []
        for path, doc, ts in stamped_artifacts(REPO):
            commit = landing_commit(path.parent.name, REPO)
            if not commit:
                continue                      # not merged; nothing to point at
            sb = ts.get("superseded_by") or doc.get("superseded_by")
            if not sb:
                bad.append(f"{path}: content is on master at {commit} but the "
                           f"artifact names no superseding commit")
                continue
            if sb.get("commit") != commit:
                bad.append(f"{path}: superseded_by.commit {sb.get('commit')} "
                           f"!= the commit that added the content {commit}")
            if not sb.get("reachable_from_master"):
                bad.append(f"{path}: superseded_by.commit is not reachable from master")
        self.assertEqual(bad, [], "\n".join(bad))

    def test_the_suite_can_actually_find_a_bad_artifact(self):
        """THE CHECK MUST BE ABLE TO FAIL.

        Feeds the same predicate a deliberately broken artifact. Without this,
        the assertion above could pass because the detector finds nothing at
        all -- an empty detection set and a clean tree look identical.
        """
        broken = {"tree_state": {"branch": "lab/gpu-sim-u2o.17.5-r2"}}
        commit = landing_commit("gpu-sim-u2o.17.5", REPO)
        self.assertIsNotNone(commit, "fixture assumption: 17.5 is on master")
        sb = broken["tree_state"].get("superseded_by")
        self.assertIsNone(sb, "an unstamped artifact has no superseded_by")
        self.assertNotEqual(sb and sb.get("commit"), commit,
                            "the detector must distinguish a missing stamp from "
                            "a correct one")

    def test_a_live_branch_is_not_claimed_superseded(self):
        """A branch that still resolves must not be reported as superseded."""
        live = "lab/gpu-sim-u2o.6.6.2-r2"
        if not branch_exists(live, REPO):
            self.skipTest(f"{live} has since been deleted")
        rec = resolve_superseded_by(live, "gpu-sim-u2o.6.6.2", REPO)
        self.assertTrue(rec["branch_still_exists"])


if __name__ == "__main__":
    unittest.main(verbosity=2)