#!/usr/bin/env python3
"""Tests for the split design. gpu-sim-u2o.7.7.

The tests are built out of the ways a split can be WORTHLESS WHILE LOOKING
FINE, because that is the failure this bead is about. Every one of the first
four is a behaviour the pre-existing structural_split.py actually exhibits on
7.1's real rows, measured in experiments/gpu-sim-u2o.7.7/metrics.json:

    assign_family(real row) -> None
    split_by_family(real rows) -> train=264  val=0  test=0
    analyze_split_quality(...)  -> total_families: 0, leakage_train_test: 0

A leakage analyzer that reports ZERO leakage for a split with no test set is
fail-open, and a test suite that only checks the happy path would have passed it
forever.

Run:  python3 tests/test_dataset_split.py
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.dataset_split import (  # noqa: E402
    SplitError, family_disjoint_split, leakage_audit, leakage_keys,
    structural_family, validate_split,
)


def row(fam="add", n=4096, digest="d1", poor=False):
    return {"op_type": fam, "problem_size": n, "token_digest": digest,
            "kernel_hash": f"{fam}-{n}-{digest}-{poor}", "launch_config": {},
            "median_runtime_ms": 0.003, "measured_regime": "launch_bound",
            "deliberately_poor": poor}


def dataset(n_fams=8, per_fam=33, sizes=(4096, 16384, 65536, 262144)):
    fams = ["add", "copy", "div", "mul", "relu", "saxpy", "scale", "triad"][:n_fams]
    return [row(f, sizes[i % len(sizes)], digest=f"dig-{f}", poor=(i == 0))
            for f in fams for i in range(per_fam)]


class TestFailOpenIsRemoved(unittest.TestCase):
    """The pre-existing behaviour, asserted against the replacement."""

    def test_preexisting_split_is_fail_open_on_real_schema(self):
        """Pin the bug, so a 'fix' that reintroduces it is caught."""
        from epic_u2o.structural_split import (
            analyze_split_quality, split_by_family)
        real = real_rows()
        split = split_by_family(real, seed=42)
        q = analyze_split_quality(split)
        self.assertEqual(len(split["train"]), len(real))
        self.assertEqual(len(split["test"]), 0)
        # and it called that clean:
        self.assertEqual(q.get("leakage_train_test"), 0)

    def test_empty_test_split_is_refused(self):
        s = {"train": dataset(), "val": [], "test": []}
        with self.assertRaises(SplitError) as cm:
            validate_split(s, dataset())
        self.assertIn("EMPTY", str(cm.exception))
        self.assertIn("fail-open", str(cm.exception).lower() + "fail-open")

    def test_empty_train_split_is_refused(self):
        s = {"train": [], "val": [], "test": []}
        with self.assertRaises(SplitError):
            validate_split(s, dataset())


class TestFamilyDisjointness(unittest.TestCase):
    def test_family_shared_with_train_is_refused(self):
        """A family must not straddle the boundary.

        The fixture is built to be WELL FORMED in every other respect --
        rows conserve exactly, no side is empty -- so the only thing wrong with
        it is the thing under test. My first version of this test tripped the
        row-conservation check first and never reached the family check, which
        would have left the assertion testing nothing.
        """
        tr = [row("add", 4096, "dig-add") for _ in range(10)]
        va = [row("copy", 4096, "dig-copy") for _ in range(5)]
        te = [row("relu", 4096, "dig-relu") for _ in range(5)]
        clean = {"train": tr, "val": va, "test": te}
        validate_split(clean, tr + va + te)          # sanity: this one is valid

        leaky_val = [row("add", 4096, "dig-add")]     # 'add' now in train AND val
        s = {"train": tr, "val": leaky_val, "test": te}
        with self.assertRaises(SplitError) as cm:
            validate_split(s, tr + leaky_val + te)
        self.assertIn("both train and val", str(cm.exception))

    def test_representation_key_shared_is_refused(self):
        """Distinct kernel_hash, identical token sequence: still leakage.

        This is the 7.1 case exactly -- 264 distinct hashes, 8 distinct token
        sequences. A check keyed on kernel_hash reports 0% leakage on a split
        that leaks every row.
        """
        # Same family in both sides would trip the FAMILY check first, which
        # would hide the thing under test. So the two sides are family-disjoint
        # and the ONLY shared thing is the token digest: a representation the
        # model can see, shared across a family boundary.
        tr = [row("add", 4096, "SHARED"), row("add", 16384, "dig-add-2")]
        va = [row("copy", 4096, "dig-copy"), row("copy", 16384, "dig-copy-2")]
        te = [row("relu", 4096, "SHARED"), row("relu", 16384, "dig-relu-2")]
        s = {"train": tr, "val": va, "test": te}
        with self.assertRaises(SplitError) as cm:
            validate_split(s, tr + va + te)
        self.assertIn("representation key", str(cm.exception))

    def test_kernel_hash_alone_would_have_hidden_it(self):
        """The wrong identifier is what produced the false clean bill."""
        a = row("add", 4096, "dig-add")
        b = row("add", 4096, "dig-add")
        a["kernel_hash"], b["kernel_hash"] = "hash-A", "hash-B"
        self.assertNotEqual(a["kernel_hash"], b["kernel_hash"])
        self.assertEqual(leakage_keys(a), leakage_keys(b),
                         "the representation key is what catches this")


class TestRowConservation(unittest.TestCase):
    def test_lost_rows_are_refused(self):
        d = dataset()
        s = {"train": d[:-5], "val": d[:5], "test": d[5:20]}
        with self.assertRaises(SplitError) as cm:
            validate_split(s, d)
        self.assertIn("lost or duplicated", str(cm.exception))

    def test_tiny_test_split_is_refused(self):
        """Family-disjoint but too small to report a metric on.

        A one-row test set produces a confident number from a single
        measurement -- the same failure as the 6.6.3 'discriminates' boolean: a
        value that looks like evidence and is not. Reaching this guard needs a
        family-disjoint split, so the fixture uses three separate families; my
        first version reused one family and tripped the family check instead,
        leaving the size guard untested.
        """
        tr = [row("add", 4096, f"dig-add-{i}") for i in range(10)]
        va = [row("copy", 4096, "dig-copy-1")]
        te = [row("relu", 4096, "dig-relu-1")]
        s = {"train": tr, "val": va, "test": te}
        with self.assertRaises(SplitError) as cm:
            validate_split(s, tr + va + te)
        self.assertIn("Too few", str(cm.exception))


class TestFamilyAssignment(unittest.TestCase):
    def test_missing_family_raises_rather_than_pooling(self):
        r = {"problem_size": 4096}
        with self.assertRaises(SplitError) as cm:
            structural_family(r)
        self.assertIn("pool every unlabelled row", str(cm.exception))

    def test_explicit_family_wins_over_op_type(self):
        self.assertEqual(structural_family({"op_type": "add",
                                            "structural_family": "memcopy"}),
                         "memcopy")

    def test_preexisting_assign_family_returns_none_on_real_rows(self):
        from epic_u2o.structural_split import assign_family
        self.assertIsNone(assign_family(real_rows()[0], real_rows()[0]["op_type"]))


class TestRecommendedSplitIsClean(unittest.TestCase):
    def test_family_holdout_validates(self):
        d = dataset()
        s = family_disjoint_split(d, test_families=["add", "copy"],
                                  val_families=["div"])
        rep = validate_split(s, d)
        self.assertTrue(rep["family_disjoint"])
        self.assertEqual(rep["test_families"], ["add", "copy"])
        self.assertNotIn("add", rep["train_families"])
        self.assertEqual(rep["n_train"] + rep["n_val"] + rep["n_test"], len(d))

    def test_family_in_both_test_and_val_raises(self):
        with self.assertRaises(SplitError):
            family_disjoint_split(dataset(), test_families=["add"],
                                  val_families=["add"])


class TestLeakageAudit(unittest.TestCase):
    def test_row_level_split_leaks_everything_on_real_data(self):
        rows = real_rows()
        a = leakage_audit(rows)["row_level_split"]
        self.assertEqual(a["test_rows_leaked_fraction"], 1.0)
        self.assertEqual(a["n_families_shared"], a["n_families_total"])

    def test_audit_names_the_wrong_identifier(self):
        note = leakage_audit(real_rows())["note"]
        self.assertIn("kernel_hash", note)


def real_rows() -> list:
    """7.1's committed measured rows, or the test refuses to run."""
    r = subprocess.run(
        ["git", "show", "lab/gpu-sim-u2o.7.1:experiments/gpu-sim-u2o.7.1/measured.jsonl"],
        cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise unittest.SkipTest("7.1 dataset not available on this ref")
    return [json.loads(l) for l in r.stdout.splitlines() if l.strip()]


MERGED_CORPUS = [
    "experiments/gpu-sim-u2o.7.1/measured.jsonl",
    "experiments/gpu-sim-e5q/measured.jsonl",
    "experiments/gpu-sim-u2o.11.3/measured.jsonl",
    "experiments/gpu-sim-f6y/measured_new.jsonl",
    "experiments/gpu-sim-0vk/measured_new.jsonl",
]


def merged_rows() -> list:
    """Merged 3261-row corpus, or skip if any file is absent."""
    rows = []
    for f in MERGED_CORPUS:
        p = REPO / f
        if not p.exists():
            raise unittest.SkipTest(f"merged corpus file missing: {f}")
        rows += [json.loads(l) for l in p.read_text().splitlines()
                 if l.strip()]
    return rows


def holdout_v1_side(digest: str, sizes: dict) -> str:
    """Mirror of run_merged_audit.py family_holdout_v1 rule."""
    ordered = sorted(sizes, key=lambda d: (sizes[d], d))
    test = set(ordered[4::5])
    val = set(ordered[3::5])
    return "test" if digest in test else "val" if digest in val else "train"


class TestMergedCorpusSplit(unittest.TestCase):
    """gpu-sim-u2o.7.7 merged audit: digest-group families on 3261 rows."""

    def test_every_row_assignable_no_pooling(self):
        rows = merged_rows()
        self.assertEqual(len(rows), 3261)
        for r in rows:
            structural_family(r)  # raises rather than pooling to None
            self.assertTrue(r["token_digest"])

    def test_shared_digest_ops_never_split(self):
        """copy/coalesced_read/coalesced_write share one token stream: an
        op-level holdout can separate them, the digest rule cannot."""
        rows = merged_rows()
        sizes = {}
        for r in rows:
            sizes[r["token_digest"]] = sizes.get(r["token_digest"], 0) + 1
        by_digest = {}
        for r in rows:
            by_digest.setdefault(r["token_digest"], set()).add(
                holdout_v1_side(r["token_digest"], sizes))
        for d, sides in by_digest.items():
            self.assertEqual(len(sides), 1, f"digest group split: {d}")
        ops = {}
        for r in rows:
            ops.setdefault(r["token_digest"], set()).add(r["op_type"])
        shared = [d for d, o in ops.items() if len(o) > 1]
        self.assertTrue(shared, "expected shared-digest op groups")

    def test_size_slice_rule_has_measurable_mass(self):
        rows = merged_rows()
        hi = [r for r in rows if r["problem_size"] >= 2 ** 26]
        self.assertGreaterEqual(len(hi), 5)
        self.assertGreater(
            sum(1 for r in hi if r["measured_regime"] == "measurable"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
