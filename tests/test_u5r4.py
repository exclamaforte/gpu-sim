"""Liveness guard for gpu-sim-u5r4 analysis reproducibility."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "gpu-sim-u5r4" / "run.sh"


class TestU5r4(unittest.TestCase):
    def test_analysis_reproducible(self):
        r = subprocess.run(["bash", str(RUN)], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
