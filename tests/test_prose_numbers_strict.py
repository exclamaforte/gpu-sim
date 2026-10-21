#!/usr/bin/env python3
"""Tests for strict proximity mode (gpu-sim-e04)."""
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


class TestStrictProximity(unittest.TestCase):
    def test_strict_fires_on_blanket_citation(self):
        f = FIX / "blanket_citation.md"
        self.assertTrue(f.is_file())
        code, out = run(["--strict", str(f)])
        self.assertEqual(code, 1, out)

    def test_strict_passes_on_per_row_citation(self):
        f = FIX / "per_row_citation.md"
        self.assertTrue(f.is_file())
        code, out = run(["--strict", str(f)])
        self.assertEqual(code, 0, out)

    def test_weak_passes_on_blanket_citation(self):
        f = FIX / "blanket_citation.md"
        code, out = run([str(f)])
        self.assertEqual(code, 0, out)


if __name__ == "__main__":
    unittest.main()
