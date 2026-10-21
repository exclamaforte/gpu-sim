#!/usr/bin/env python3
"""Tests for the prose-number provenance lint (gpu-sim-4u8).

The lint exists because the PI wrote unverified numbers into
experiments/gpu-sim-pkb/results.md twice. These tests pin both directions:

  * it MUST fire on the historical rejected version (commit e4a8a86), specifically
    on the hand-typed table row, the uncommitted run count, and the digest sentence;
  * it MUST NOT fire on the corrected version (commit c92871e), or on any genuine
    claim an artifact supports.

The second direction matters more. A lint that fires on everything is as useless as
the hand-checking it replaces, and worse -- it trains people to ignore it. That trap
is documented at length in experiments/gpu-sim-u2o.6.11.1/lint_prose_numbers.py and
is the reason every case below has a paired negative control.
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_prose_numbers.py"
PKB = REPO / "experiments" / "gpu-sim-pkb"
REJECTED = PKB / "NEGCTL_rejected_results.md"
CORRECTED = PKB / "results.md"


def run(doc: Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(SCRIPT), str(doc)],
                       capture_output=True, text=True, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


class TestProseNumbers(unittest.TestCase):
    def test_fires_on_the_rejected_version(self):
        """Criterion 2: fires on the version a reviewer rejected."""
        self.assertTrue(REJECTED.is_file(), "fixture missing")
        code, out = run(REJECTED)
        self.assertEqual(code, 1, out)
        # the hand-typed table row: the script reported 2.00, the prose said 1.00
        self.assertIn("1.00", out)
        # the uncommitted run count
        self.assertIn("6/6 runs", out)
        self.assertIn("only 2 trace file(s) are committed", out)

    def test_does_not_fire_on_the_corrected_version(self):
        """Criterion 3: the corrected document passes."""
        self.assertTrue(CORRECTED.is_file(), "corrected results.md missing")
        code, out = run(CORRECTED)
        self.assertEqual(code, 0, out)
        self.assertIn("OK:", out)

    def test_does_not_fire_on_a_genuine_claim(self):
        """Criterion 4: a claim the artifact supports must pass.

        This is the trap the older prose lint documents: a check that flags
        everything is worse than no check. The number below is copied verbatim
        from reduction_output.txt ('byte_factor=1.00').
        """
        tmp = PKB / "GENUINE_tmp_results.md"
        try:
            tmp.write_text(
                "# genuine\n\n"
                "Cite: reduction_output.txt and traces/trace_S524288_n2_m1.log\n\n"
                "| S | factor |\n|---|---|\n| 524288 | 1.00 |\n\n"
                "Both ranks agree at 1.00.\n"
            )
            code, out = run(tmp)
            self.assertEqual(code, 0, out)
        finally:
            tmp.unlink(missing_ok=True)

    def test_fires_when_no_artifact_is_cited(self):
        """A results document citing nothing is the failure this lint is for."""
        tmp = PKB / "NOARTIFACT_tmp_results.md"
        try:
            tmp.write_text("# bare\n\n| S | factor |\n|---|---|\n| 524288 | 1.00 |\n")
            code, out = run(tmp)
            self.assertEqual(code, 1, out)
            self.assertIn("cites no artifact file", out)
        finally:
            tmp.unlink(missing_ok=True)

    def test_rounding_does_not_false_positive(self):
        """A document quoting 1.25 for an artifact line '1.2500' is supported."""
        tmp = PKB / "ROUNDING_tmp_results.md"
        try:
            tmp.write_text(
                "# rounding\n\nCite: reduction_output.txt\n\n"
                "| S | factor |\n|---|---|\n| 65536 | 1.25 |\n"
            )
            code, out = run(tmp)
            self.assertEqual(code, 0, out)
        finally:
            tmp.unlink(missing_ok=True)

    def test_cross_bead_citation_is_a_finding(self):
        """Review (b): borrowing another bead's artifact must be declared."""
        tmp = PKB / "CROSSBEAD_tmp_results.md"
        try:
            tmp.write_text(
                "# cross-bead\n\nCite: experiments/gpu-sim-4tw/sweep.txt\n\n"
                "| S | factor |\n|---|---|\n| 524288 | 1.00 |\n"
            )
            code, out = run(tmp)
            self.assertEqual(code, 1, out)
            self.assertIn("from ANOTHER bead without declaring it", out)
        finally:
            tmp.unlink(missing_ok=True)

    def test_wildcard_citation_is_a_finding(self):
        """Review (c): a citation that names no file proves nothing."""
        tmp = PKB / "WILDCARD_tmp_results.md"
        try:
            tmp.write_text("# wild\n\nCite: traces/*.log\n\n| S | factor |\n|---|---|\n| 524288 | 1.00 |\n")
            code, out = run(tmp)
            self.assertEqual(code, 1, out)
            self.assertIn("is a wildcard", out)
        finally:
            tmp.unlink(missing_ok=True)

    def test_run_count_without_a_size_is_a_finding(self):
        """Review (3): '6/6 runs' with no size is unverifiable, not passing."""
        tmp = PKB / "NOSIZECOUNT_tmp_results.md"
        try:
            tmp.write_text(
                "# nosize\n\nCite: reduction_output.txt\n\nWe ran it 6/6 runs cleanly.\n"
            )
            code, out = run(tmp)
            self.assertEqual(code, 1, out)
            self.assertIn("names no size", out)
        finally:
            tmp.unlink(missing_ok=True)

    def test_fenced_quotation_is_not_a_claim(self):
        """A verbatim quote of rejected text in a fence must not count as a claim."""
        tmp = PKB / "FENCE_tmp_results.md"
        try:
            tmp.write_text(
                "# fence\n\nCite: reduction_output.txt\n\n"
                "The rejected version said:\n\n```\nfactor 9.99 after dedup\n```\n"
                "| S | factor |\n|---|---|\n| 524288 | 1.00 |\n"
            )
            code, out = run(tmp)
            self.assertEqual(code, 0, out)
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
