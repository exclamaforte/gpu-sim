#!/usr/bin/env python3
"""Tests for 11.5 (CPU): every generated results line appears verbatim in
results_writeup.md; no synthetic-fixture language remains."""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.11.5"))

import gen_writeup_115  # noqa: E402


class TestWriteup(unittest.TestCase):
    def test_numbers_match_artifacts(self):
        text = (REPO / "epic_u2o" / "results_writeup.md").read_text()
        lines = gen_writeup_115.table()
        self.assertGreater(len(lines), 3)
        for line in lines:
            self.assertIn(line, text, f"drift: {line[:60]}")

    def test_no_synthetic_fixture_language(self):
        text = (REPO / "epic_u2o" / "results_writeup.md").read_text()
        for phrase in ("synthetic fixtures", "replace with measured",
                       "validates the plumbing only", "no real-GPU timings"):
            self.assertNotIn(phrase, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
