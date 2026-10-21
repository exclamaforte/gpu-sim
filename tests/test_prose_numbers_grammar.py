#!/usr/bin/env python3
"""Tests for grammar extension (gpu-sim-7vh)."""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_prose_numbers.py"
FIX = REPO / "tests" / "fixtures"


def run(args):
    p = subprocess.run([sys.executable, str(SCRIPT)] + args,
                       capture_output=True, text=True, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


class TestGrammar(unittest.TestCase):
    def test_fires_on_4yw_phrase(self):
        f = FIX / "fouryw_phrase.md"
        self.assertTrue(f.is_file())
        code, out = run(["--grammar", str(f)])
        self.assertEqual(code, 1, out)
        self.assertIn("3 of 6 phases", out)

    def test_honest_ratio_passes(self):
        code, out = run(["--grammar", str(FIX / "honest_ratio.md")])
        self.assertEqual(code, 0, out)

    def test_honest_count_passes(self):
        code, out = run(["--grammar", str(FIX / "honest_count.md")])
        self.assertEqual(code, 0, out)

    def test_honest_percent_passes(self):
        code, out = run(["--grammar", str(FIX / "honest_percent.md")])
        self.assertEqual(code, 0, out)

    def test_default_mode_unchanged_decimal(self):
        # default mode still catches decimals; grammar off by default
        code, out = run([str(FIX / "fouryw_phrase.md")])
        # no decimals in fixture, no artifacts issue? It cites metrics.json which resolves.
        # Should pass (no decimal claims).
        self.assertEqual(code, 0, out)


if __name__ == "__main__":
    unittest.main()
