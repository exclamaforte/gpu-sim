#!/usr/bin/env python3
"""Tests for 11.6 (CPU, no live-db mutation): bd-exp.sh validates args,
requires --artifact on finish, and dry-runs both commands with metadata."""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SH = str(REPO / "scripts" / "bd-exp.sh")


def run(*args):
    return subprocess.run([SH, *args], capture_output=True, text=True)


class TestBdExp(unittest.TestCase):
    def test_start_usage(self):
        r = run("start")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("usage", r.stderr)

    def test_finish_validates_outcome(self):
        r = run("finish", "x", "--outcome", "maybe", "--kept", "true",
                "--artifact", "z")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--outcome must be", r.stderr)

    def test_finish_requires_artifact(self):
        r = run("finish", "x", "--outcome", "neutral", "--kept", "false")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--artifact is required", r.stderr)

    def test_dry_run_start(self):
        r = run("start", "gpu-sim-u2o.10.4", "t", "h", "--dry-run")
        self.assertEqual(r.returncode, 0)
        self.assertIn("DRY-RUN start", r.stdout)
        self.assertIn("hypothesis", r.stdout)

    def test_dry_run_finish(self):
        r = run("finish", "demo", "--outcome", "neutral", "--kept", "false",
                "--artifact", "neg", "--spearman", "0.5", "--dry-run")
        self.assertEqual(r.returncode, 0)
        self.assertIn("DRY-RUN finish demo outcome=neutral kept=false",
                      r.stdout)
        self.assertIn("spearman", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
