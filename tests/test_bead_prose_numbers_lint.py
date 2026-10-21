#!/usr/bin/env python3
"""Tests for the bead prose-number lint (gpu-sim-3fm)."""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_bead_prose_numbers.py"
FIX = REPO / "tests" / "fixtures"


def run(args):
    p = subprocess.run([sys.executable, str(SCRIPT)] + args,
                       capture_output=True, text=True, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


class TestBeadProseNumbers(unittest.TestCase):
    def test_fires_on_original_bl5_description(self):
        """Criterion 2: fires on the original (rejected) bl5 description."""
        f = FIX / "bl5_rejected_description.md"
        self.assertTrue(f.is_file())
        code, out = run([str(f)])
        self.assertEqual(code, 1, out)
        # the false claim
        self.assertIn("0.67", out)

    def test_does_not_fire_on_corrected_description(self):
        """Criterion 3-ish: corrected description passes."""
        f = FIX / "bl5_corrected_description.md"
        self.assertTrue(f.is_file())
        code, out = run([str(f)])
        self.assertEqual(code, 0, out)
        self.assertIn("OK:", out)

    def test_negative_control_honest_prose_passes(self):
        f = FIX / "bead_honest_prose.md"
        self.assertTrue(f.is_file())
        code, out = run([str(f)])
        self.assertEqual(code, 0, out)

    def test_fires_when_no_artifact_cited(self):
        tmp = FIX / "NOART_tmp.md"
        try:
            tmp.write_text("<!-- bead: gpu-sim-testfixture -->\n# bare\n\nfactor is 1.23 here.\n")
            code, out = run([str(tmp)])
            self.assertEqual(code, 1, out)
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
