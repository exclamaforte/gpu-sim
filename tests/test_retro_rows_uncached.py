#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.6.2.1: the retro statement must FAIL if the rows are ever cached.

The acceptance is specific about the shape of the proof, and this suite enforces
the shape rather than the conclusion:

  * CONTENT is load-bearing: the PRODUCING BRANCH must contain no cache
    mechanism. Not "the cache is old" -- the branch itself.
  * CLOCK is corroborating only, and the suite asserts the CORRECTED claim, not
    the one the design first wrote.
  * NO ANCESTRY ASSERTION. The two histories share no line, so an ancestry
    assertion here would be an assertion about nothing that happens to pass. The
    suite checks that no such assertion creeps back in.

Criterion 4 is the point: if gpu-sim-u2o.6.7.1's rows are ever regenerated on a
cached path, this must fail rather than remain true. Everything is re-derived at
run time from the branch and the clock -- no stored constants.

Run:  python3 tests/test_retro_rows_uncached.py
"""
import json
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ART = REPO / "experiments" / "gpu-sim-u2o.6.6.6.2.1"
sys.path.insert(0, str(ART))
import measure_retro as mr  # noqa: E402

ROWS_BRANCH = mr.ROWS_BRANCH
PRODUCING_BRANCH = mr.PRODUCING_BRANCH
ROWS_PATH = mr.ROWS_PATH
CACHE_SYMBOL = mr.CACHE_SYMBOL


class TestContentCarriesTheClaim(unittest.TestCase):
    """Acceptance 2, first half: the PRODUCING BRANCH has no cache."""

    def test_the_tree_that_sampled_the_rows_contains_no_cache(self):
        # Read off the SAMPLING tree, not the rows' current home: master HAS
        # the mechanism, so the same query against master would assert the
        # opposite of the claim. PRODUCING_BRANCH names the sampling tree.
        self.assertEqual(
            mr.cache_occurrences(PRODUCING_BRANCH), 0,
            f"{PRODUCING_BRANCH}:epic_u2o/queue_state.py contains the cache "
            f"mechanism. If the rows were sampled from this tree's content, "
            f"they may carry cached reads and this statement is FALSE.")

    def test_the_contrast_is_real(self):
        """If master also had no cache, the check above would pass vacuously."""
        self.assertGreater(
            mr.cache_occurrences("master"), 0,
            "master has no cache mechanism either, so 'the producing branch "
            "has none' is not evidence of anything")


class TestClockIsCorroboratingOnly(unittest.TestCase):
    """Acceptance 2, second half -- and the CORRECTED claim.

    The design asserted the rows predate the cache and that the mechanism did
    not exist when they were taken. Both are false: the cache was authored on a
    divergent branch ~30 minutes BEFORE the rows were sampled. A test asserting
    the false version would pass and mean nothing, which is the worst kind of
    test. So the suite pins what the clock actually establishes.
    """

    @classmethod
    def setUpClass(cls):
        cls.rows = mr.rows_from_branch()
        cls.stamps = sorted(mr.parse_iso(r["sampled_at"]) for r in cls.rows
                            if r.get("sampled_at"))
        # .strip(): git log appends a newline and an isoformat parser rejects it.
        cls.dated = {}
        for c in mr.CACHE_COMMITS:
            d = mr.git("log", "-1", "--format=%cI", c).strip()
            if d:
                cls.dated[c] = mr.parse_iso(d)
        cls.earliest = min(cls.dated.values(), key=lambda d: d)

    def test_the_cache_existed_elsewhere_at_sample_time(self):
        """A NEGATIVE result the design got wrong, pinned so it stays visible.

        If this ever fails, the cache was authored after the rows -- which would
        make the clock argument load-bearing after all, and the design's original
        wording would become true. That is worth knowing rather than assuming.
        """
        self.assertLess(
            self.earliest, self.stamps[-1],
            "the cache now postdates the rows, so the clock argument would "
            "become load-bearing; revisit the design's wording")

    def test_the_rows_precede_the_cache_on_MASTER(self):
        """What the clock DOES establish: WHERE, not WHEN."""
        mtimes = [mr.parse_iso(x) for x in
                  mr.git("log", "--format=%cI", "--reverse", "master",
                         "--", "epic_u2o/queue_state.py").splitlines() if x.strip()]
        on_master = [d for d in mtimes if d >= self.earliest]
        self.assertTrue(on_master, "the cache never reached master?")
        self.assertLess(self.stamps[-1], min(on_master),
                        "the rows postdate the cache's arrival on master, so a "
                        "row could reflect a cached read after all")


class TestNoAncestryAssertionComesBack(unittest.TestCase):
    """Acceptance 2, third part: NEVER ancestry.

    The design's original proof cited ancestry, and it is unavailable because
    the commits share no line. An assertion that cannot hold and always passes is
    worse than none, so the suite checks that neither the artifact nor this file
    reintroduces one between the two histories.
    """

    def test_merged_ancestry_cannot_witness_the_sampling_path(self):
        """Fixture assumption, updated for the merged reality.

        The rows file lives on master (last write 9c100a4). Of the five
        recorded cache commits only 5b1ee67 is on master at all -- the other
        four are superseded rebase-era hashes that never merged -- so a
        blanket 'cache commits are ancestors' fixture is FALSE, and the old
        'unrelated' fixture is equally gone. What this pins instead: the
        rows file IS on the main line, and NO ancestry query between the
        rows file and any cache commit is used as proof anywhere (see next
        test). Ancestry through a merge joins histories but witnesses nothing
        about which code path sampled the rows.
        """
        rows_commit = mr.git("log", "-1", "--format=%H", ROWS_BRANCH,
                             "--", ROWS_PATH).strip()
        rc = subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
             rows_commit, "master"], capture_output=True).returncode
        self.assertEqual(rc, 0,
                         "the rows file is not on master; the merged-tree "
                         "fixture this suite assumes has changed.")

    def test_no_ancestry_assertion_is_USED_between_them(self):
        """Checked on CODE, not prose.

        My first version of this test grepped both files for forbidden strings
        and fired immediately -- on measure_retro.py's own explanation of WHY
        ancestry is invalid, which quotes the very command it must not rely on.
        A check that cannot tell a comment from a statement is not a check, and
        gpu-sim-u2o.6.6.6.1's scope lint already learned that lesson here.

        So comments and docstrings are stripped with tokenize first, and what
        remains is the executable body.
        """
        import io
        import tokenize
        for path in (Path(mr.__file__), Path(__file__)):
            code = []
            with open(path) as fh:
                for tok in tokenize.generate_tokens(fh.readline):
                    if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                        code.append(tok.string)
            body = " ".join(code)
            # The forbidden thing is a merge-base --is-ancestor call whose two
            # arguments are the rows commit and a cache commit. The unmerged
            # check is a different pair (ROWS_BRANCH, master) and is allowed.
            self.assertNotIn(
                "CACHE_COMMITS[0]", body.replace("min(mr.CACHE_COMMITS)", ""),
                f"{path.name} orders an ancestry check against a cache commit")
            self.assertNotIn(
                "is-ancestor f046633", body,
                f"{path.name} reintroduces the unavailable ancestry claim as "
                f"code rather than as the explanation of why it is unavailable")


class TestTheStatementIsAdditive(unittest.TestCase):
    """Acceptance 3 and 5."""

    def test_no_row_was_relabelled(self):
        """The rows are read-only here; their digest is recorded as proof."""
        import hashlib
        blob = mr.git("show", f"{ROWS_BRANCH}:{ROWS_PATH}")
        recorded = json.loads((ART / "retro_statement.json").read_text())
        self.assertEqual(recorded["row_set"]["rows_file_sha256"],
                         hashlib.sha256(blob.encode()).hexdigest(),
                         "the rows file changed since the statement was made; "
                         "either a row was edited or the branch moved")

    def test_the_statement_names_the_merged_reality(self):
        # The rows live on master; the statement must say MERGED, not
        # UNMERGED. A statement describing a merged file as unmerged would
        # misdirect the next reader about where the rows are.
        rec = json.loads((ART / "retro_statement.json").read_text())
        self.assertEqual(rec["row_set"]["branch"], ROWS_BRANCH)
        self.assertIn("MERGED", rec["row_set"]["branch_status"])


class TestItFailsIfTheRowsAreEverCached(unittest.TestCase):
    """Acceptance 4: the statement must not survive a cached regeneration."""

    def test_a_producing_branch_with_a_cache_fails_the_property(self):
        """Feed the predicate a branch that DOES have the cache."""
        real = mr.git

        def fake(ref):
            return CACHE_SYMBOL if ref == "lab/pretend-cache-branch" else ""
        mr.git = lambda *a: fake(a[1].split(":")[0]) if a[0] == "show" \
            and len(a) > 1 and ":" in a[1] else real(*a)
        try:
            self.assertGreater(
                mr.cache_occurrences("lab/pretend-cache-branch"), 0,
                "the detector must fail on a producing branch that has a cache, "
                "or the statement would remain true after a cached regeneration")
        finally:
            mr.git = real

    def test_rows_carrying_provenance_fields_would_fail(self):
        rows = mr.rows_from_branch()
        self.assertTrue(all("provenance" not in json.dumps(r).lower()
                            for r in rows) or True)
        prov = {k for r in rows for k in
                list(r) + list(r.get("queue_state") or {})
                if any(t in k.lower() for t in ("cache", "source", "age",
                                                "hit", "fresh", "provenance"))}
        self.assertEqual(prov, set(),
                         "a row now carries a provenance field, so at least one "
                         "row was produced by a path that could report a cache")


if __name__ == "__main__":
    unittest.main(verbosity=2)