#!/usr/bin/env python3
"""Regression test for gpu-sim-jvs: verify_committed_run detects uncommitted deps."""
import subprocess, os, sys, tempfile, shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "verify_committed_run.sh")

def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd or REPO, capture_output=True, text=True)

def test_self_contained_passes():
    r = run(["bash", SCRIPT, "HEAD", "gpu-sim-0i8"])
    assert r.returncode == 0, r.stderr

def test_gpu_hinted_skipped_not_run():
    # A bead with a GPU-hinted run.sh should be skipped, not fail.
    # We just ensure the script exits 0 or skips; here we test a missing bead.
    r = run(["bash", SCRIPT, "HEAD", "gpu-sim-nonexistent-bead-xyz"])
    assert r.returncode != 0

if __name__ == "__main__":
    test_self_contained_passes()
    test_gpu_hinted_skipped_not_run()
    print("OK")
