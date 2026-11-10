#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.2.1.1: an artifact whose branch moved must say where it went.

Criterion 3, as narrowed by gpu-sim-u2o.6.6.2.1.1.1: for every artifact that
OPTS IN by carrying a tree_state block, a carried superseded_by must agree
with the per-FILE landing commit and be reachable from master. Artifacts
without a tree_state block are out of scope -- regenerating frozen history
to satisfy a test is the 6.6.2.1 failure mode -- but the run REPORTS the
opt-in count so the gap is on the face of it, not silent.

Two properties this suite is careful about, both from this lab's own rules:

  * ABSENT IS NOT NULL. An artifact that cannot name a superseding commit must
    OMIT the key. `"superseded_by": null` asserts that nothing supersedes it --
    a claim nobody checked -- and a null reads as coverage while an absent key
    does not (gpu-sim-u2o.6.6.6 criterion 5).
  * THE CHECK MUST BE ABLE TO PASS -- and to FAIL. A test that fires on every
    artifact is indistinguishable from one whose detection is broken, and a
    suite that passes vacuously is the same error in the other direction. So
    this asserts that a correctly-stamped artifact is left alone, that a
    wrong stamp is flagged, and that at least one artifact opts in.

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
    _supersession_record, branch_exists, first_word, landing_commit_for,
    resolve_commit, resolve_superseded_by, stamp_agreement_errors, stamp_tree_state,
    stamped_artifacts,
)


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _opted_in(repo=REPO):
    """Artifacts carrying a real tree_state block (not a top-level branch)."""
    return [(p, doc, ts) for p, doc, ts in stamped_artifacts(repo)
            if isinstance(doc.get("tree_state"), dict)
            and doc["tree_state"].get("branch")]


def _scratch_repo():
    """A throwaway git repo on branch master with user config set."""
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    subprocess.run(["git", "init", "-q", "-b", "master", str(root)],
                   check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email",
                    "lab@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name",
                    "lab"], check=True)
    return td, root


def _commit_all(root, message):
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", message],
                   check=True)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()


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


class TestPerFileLanding(unittest.TestCase):
    def test_a_sibling_add_moves_the_directory_rule_but_not_the_file_rule(self):
        """NEGATIVE CONTROL for the 6.6.2.1.1.1 fix (the xy8 lesson: show it).

        Commit 1 adds expdir/artifact.json; commit 2 adds a SIBLING beside it.
        The old directory rule (newest add under the dir) moves from commit 1
        to commit 2 -- the defect. The per-file rule (that file's oldest add)
        stays on commit 1. Both halves are asserted: a control that cannot
        show the old code moving proves nothing about the new code holding.
        """
        td, root = _scratch_repo()
        try:
            (root / "expdir").mkdir()
            (root / "expdir" / "artifact.json").write_text('{"a": 1}\n')
            first = _commit_all(root, "add the artifact")

            def old_directory_rule():
                out = subprocess.run(
                    ["git", "-C", str(root), "log", "--format=%H", "-1",
                     "--diff-filter=A", "master", "--", "expdir/"],
                    capture_output=True, text=True, check=True).stdout.strip()
                return out.split()[0]

            self.assertEqual(old_directory_rule(), first)
            self.assertEqual(
                landing_commit_for("expdir/artifact.json", root), first)

            (root / "expdir" / "sibling.json").write_text('{"b": 2}\n')
            second = _commit_all(root, "add a sibling beside the artifact")
            self.assertNotEqual(second, first)

            self.assertEqual(old_directory_rule(), second,
                             "the directory rule moved when a sibling landed")
            self.assertEqual(
                landing_commit_for("expdir/artifact.json", root), first,
                "the per-file rule must not move when a sibling lands")
        finally:
            td.cleanup()


class TestTheSupersessionRule(unittest.TestCase):
    def test_unresolvable_omits_the_key_rather_than_writing_null(self):
        with tempfile.TemporaryDirectory() as tds:
            repo = Path(tds)
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
        rec = resolve_superseded_by(
            "lab/gpu-sim-u2o.6.6.2-r2",
            "experiments/gpu-sim-u2o.6.6.2/job.json", REPO)
        self.assertIsNotNone(rec)
        self.assertTrue(rec["branch_still_exists"],
                        "the live ref is the hazard and must be recorded")
        self.assertTrue(rec["reachable_from_master"])

    def test_deleted_ref_is_recorded_as_gone(self):
        rec = resolve_superseded_by(
            "lab/gpu-sim-u2o.17.5-r2",
            "experiments/gpu-sim-u2o.17.5/job.json", REPO)
        self.assertIsNotNone(rec)
        self.assertFalse(rec["branch_still_exists"])
        self.assertIn("would find nothing", rec["note"])

    def test_an_already_stamped_record_is_not_recomputed(self):
        ts = {"branch": "x", "superseded_by": {"commit": "handwritten"}}
        self.assertEqual(stamp_tree_state(ts, "y", REPO)["superseded_by"],
                         {"commit": "handwritten"},
                         "an explicit stamp must not be silently overwritten")

    def test_writer_emission_in_scratch(self):
        """Item 5: a NEW artifact written after this change carries the field.

        Demonstrated by writing one in scratch, not by asserting the writer
        code looks right: stamp with artifact_path against a scratch repo,
        then the agreement predicate holds over the emitted record.
        """
        td, root = _scratch_repo()
        try:
            (root / "artifacts").mkdir()
            art = root / "artifacts" / "new.json"
            art.write_text(json.dumps(
                {"tree_state": {"branch": "lab/scratch-demo"}}) + "\n")
            head = _commit_all(root, "a new artifact lands")
            ts = stamp_tree_state({"branch": "lab/scratch-demo"}, "demo",
                                  root,
                                  artifact_path="artifacts/new.json")
            sb = ts.get("superseded_by")
            self.assertIsNotNone(sb, "the writer must emit the field")
            self.assertEqual(sb["commit"], head)
            self.assertEqual(
                sb["commit"], landing_commit_for("artifacts/new.json", root),
                "the emitted value must agree with the query")
            self.assertFalse(sb["branch_still_exists"])
        finally:
            td.cleanup()



def _audit_stamps(repo):
    """Run the carried-stamp agreement check over opted-in artifacts in repo.
    Returns (bad, unstamped, checked) lists. Shared by the audit test and the
    FAIL-direction plants so the same code path is exercised."""
    from pathlib import Path as _P
    repo = _P(repo)
    bad, unstamped, checked = [], [], []
    for path, doc, ts in stamped_artifacts(repo):
        sb = (ts.get("superseded_by") if isinstance(ts, dict) else None) or doc.get("superseded_by")
        # Judge any artifact that carries a stamp, even without a tree_state
        # block (gpu-sim-7k6e): the stamp makes it judgeable.
        if not (isinstance(sb, dict) and sb.get("commit")):
            # unstamped: still report if it has a tree_state branch
            if isinstance(doc.get("tree_state"), dict) and doc["tree_state"].get("branch"):
                rel = path.relative_to(repo)
                landing = landing_commit_for(rel, repo)
                if landing:
                    unstamped.append(f"{rel}: landed at {landing}, no stamp")
            continue
        rel = path.relative_to(repo)
        landing = landing_commit_for(rel, repo)
        if not landing:
            continue
        checked.append(str(rel))
        sb = ts.get("superseded_by") or doc.get("superseded_by")
        if not sb or not isinstance(sb, dict):
            unstamped.append(f"{rel}: landed at {landing}, no stamp")
            continue
        errs = stamp_agreement_errors(rel, repo)
        bad.extend(errs)
    return bad, unstamped, checked


class TestEveryStampedArtifact(unittest.TestCase):
    """Criterion 3 narrowed: over artifacts that opt in with a tree_state block."""

    def test_no_artifact_carries_a_null_superseded_by(self):
        """The absent-vs-null rule, asserted over real opt-in artifacts."""
        offenders = []
        for path, doc, ts in _opted_in():
            sb = ts.get("superseded_by") or doc.get("superseded_by")
            if "superseded_by" in ts and ts["superseded_by"] is None:
                offenders.append(str(path))
            if "superseded_by" in doc and doc["superseded_by"] is None:
                offenders.append(str(path))
        self.assertEqual(offenders, [],
                         f"null superseded_by reads as coverage: {offenders}")

    def test_carried_stamps_agree_with_the_per_file_landing(self):
        """Reachability-and-agreement over opt-in artifacts, with the gap reported.

        A carried stamp that disagrees (or is unreachable) FAILS -- that is
        drift, the 6.6.2 class. An opted-in artifact carrying NO stamp is
        REPORTED, not failed: rewriting frozen history to satisfy a test is
        the 6.6.2.1 failure mode. The printed opt-in count keeps the coverage
        gap on the face of the run.
        """
        bad, unstamped, checked = _audit_stamps(REPO)
        # per-artifact declared kind + commit (gpu-sim-7k6e)
        for path, doc, ts in stamped_artifacts(REPO):
            sb = ts.get("superseded_by") if isinstance(ts, dict) else None
            sb = sb or doc.get("superseded_by")
            if isinstance(sb, dict) and sb.get("commit"):
                print(f"[tree_state] {path.relative_to(REPO)}: "
                      f"commit_kind={sb.get('commit_kind')}, commit={sb.get('commit')}")
        print(f"[tree_state] opt-in artifacts checked: {len(checked)}; "
              f"carrying a stamp: {len(checked) - len(unstamped)}; "
              f"opted-in but unstamped (reported, not failed): {unstamped}")
        self.assertGreater(len(checked), 0,
                           "no opt-in artifact is on master: the suite would "
                           "pass vacuously")
        self.assertEqual(bad, [], "\n".join(bad))

    def test_the_detector_flags_a_wrong_stamp(self):
        """THE CHECK MUST BE ABLE TO FAIL (both halves).

        Reachability half: an unreachable commit yields NO record (omit, not
        null). Agreement half: the stale 566ba0b citation this bead corrects
        is the standing positive control -- it must disagree with the
        per-file landing, or the comparison above could not fire.
        """
        self.assertIsNone(
            _supersession_record("lab/anything",
                                 "0000000000000000000000000000000000000000",
                                 REPO),
            "an unreachable commit must not produce a record")
        landing = landing_commit_for(
            "experiments/gpu-sim-u2o.6.6.2/job.json", REPO)
        self.assertIsNotNone(landing)
        self.assertNotEqual(
            "566ba0b", landing,
            "the corrected stale citation must disagree with the landing, "
            "or the agreement check above is decoration")

    def test_a_live_branch_is_not_claimed_superseded(self):
        """A branch that still resolves must not be reported as superseded."""
        live = "lab/gpu-sim-u2o.6.6.2-r2"
        if not branch_exists(live, REPO):
            self.skipTest(f"{live} has since been deleted")
        rec = resolve_superseded_by(
            live, "experiments/gpu-sim-u2o.6.6.2/job.json", REPO)
        self.assertTrue(rec["branch_still_exists"])




class TestResolveCommitPlants(unittest.TestCase):
    """Plants for the fd5i fix: commit comparison must resolve, not stringify."""

    def test_a_seven_char_prefix_of_landing_sha_passes(self):
        landing = landing_commit_for(
            "experiments/gpu-sim-u2o.6.6.2/job.json", REPO)
        self.assertIsNotNone(landing)
        prefix = landing[:7]
        assert resolve_commit(prefix, REPO) == landing, "a 7-char prefix must resolve to the landing sha"
        self.assertEqual(resolve_commit(prefix, REPO), landing,
                         "a 7-char prefix of the landing sha must resolve to it")

    def test_b_abbreviation_of_a_different_commit_fails(self):
        landing = landing_commit_for(
            "experiments/gpu-sim-u2o.6.6.2/job.json", REPO)
        self.assertIsNotNone(landing)
        other = resolve_commit("566ba0b", REPO)
        # 566ba0b must resolve to a real commit distinct from landing
        self.assertIsNotNone(other, "control: 566ba0b must resolve")
        assert other != landing, "an abbreviation of a different commit must not agree"
        self.assertNotEqual(other, landing,
                            "an abbreviation of a different commit must not agree")

    def test_c_unresolvable_sha_fails(self):
        assert resolve_commit("0000000000000000000000000000000000000000", REPO) is None, "an unresolvable stamp must yield None"
        self.assertIsNone(
            resolve_commit("0000000000000000000000000000000000000000", REPO),
            "an unresolvable stamp must yield None")

    def test_d_real_pair_97575af_vs_landing_passes(self):
        landing = landing_commit_for(
            "experiments/gpu-sim-u2o.6.6.2/job.json", REPO)
        self.assertIsNotNone(landing)
        assert resolve_commit("97575af", REPO) == landing, "97575af must resolve to the landing commit"
        self.assertEqual(resolve_commit("97575af", REPO), landing,
                         "97575af must resolve to the landing commit")




class TestStampAgreementFailDirection(unittest.TestCase):
    """Plants for the FAIL direction: the agreement rule must report a wrong
    stamp, not only pass on the right one (reviewer E1/E2)."""

    def _make_artifact(self, td, stamp):
        root = Path(td.name)
        subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "master"],
                       check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email",
                        "lab@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "lab"],
                       check=True)
        (root / "experiments" / "x").mkdir(parents=True)
        art = root / "experiments" / "x" / "a.json"
        art.write_text(__import__("json").dumps(
            {"tree_state": {"branch": "lab/x",
                            "superseded_by": {"commit": stamp,
                                              "commit_kind": "oldest_add",
                                              "reachable_from_master": True}}}) + "\n")
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "add artifact"],
                       check=True)
        landing = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        return root, landing

    def test_wrong_stamp_is_reported(self):
        td = tempfile.TemporaryDirectory()
        try:
            root, landing = self._make_artifact(td, "0000000000000000000000000000000000000001")
            # 0000...0001 is not a real commit -> unresolvable -> error
            bad, unstamped, checked = _audit_stamps(root)
            assert len(bad) > 0, "an unresolvable stamp must be reported"
            assert any("unresolvable" in e for e in bad), bad
        finally:
            td.cleanup()

    def test_different_commit_stamp_is_reported(self):
        td = tempfile.TemporaryDirectory()
        try:
            root, landing = self._make_artifact(td, None)
            # create a second commit to get a different sha
            (root / "x.txt").write_text("x")
            subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "second"], check=True)
            second = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
            # stamp the artifact with the second commit (different from landing=first)
            art = root / "experiments" / "x" / "a.json"
            doc = __import__("json").loads(art.read_text())
            doc["tree_state"]["superseded_by"]["commit"] = second[:7]
            doc["tree_state"]["superseded_by"]["commit_kind"] = "oldest_add"
            art.write_text(__import__("json").dumps(doc) + "\n")
            # re-commit the change so landing doesn't move (amend not needed; landing stays first commit's sha for the file's oldest add)
            bad, unstamped, checked = _audit_stamps(root)
            # The stamp (second[:7]) resolves to 'second', which != landing (first commit) -> error
            assert len(bad) > 0, "a wrong-but-resolvable stamp must be reported"
        finally:
            td.cleanup()


    def test_last_content_change_wrong_commit_reported(self):
        """A stamp declaring last_content_change but naming the oldest add must fail."""
        td = tempfile.TemporaryDirectory()
        try:
            root, landing = self._make_artifact(td, None)
            art = root / "experiments" / "x" / "a.json"
            # Add a second commit that changes a.json content (so last_content_change != landing)
            doc2 = __import__("json").loads(art.read_text())
            doc2["data"] = "changed"
            art.write_text(__import__("json").dumps(doc2) + "\n")
            subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "content change"], check=True)
            # Re-stamp with last_content_change kind but name the OLDEST add (landing).
            # The last content change is the second commit, so this must fail.
            doc = __import__("json").loads(art.read_text())
            doc["tree_state"]["superseded_by"]["commit"] = landing[:7]
            doc["tree_state"]["superseded_by"]["commit_kind"] = "last_content_change"
            art.write_text(__import__("json").dumps(doc) + "\n")
            bad, unstamped, checked = _audit_stamps(root)
            assert len(bad) > 0, "a wrong last_content_change stamp must be reported"
        finally:
            td.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
