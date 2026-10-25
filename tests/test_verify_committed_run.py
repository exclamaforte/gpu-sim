#!/usr/bin/env python3
"""Regression test for gpu-sim-jvs: verify_committed_run detects uncommitted deps."""
import subprocess, os, sys, shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "verify_committed_run.sh")

def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd or REPO, capture_output=True, text=True)

def test_self_contained_passes():
    r = run(["bash", SCRIPT, "HEAD", "gpu-sim-0i8"])
    assert r.returncode == 0, r.stderr

def test_missing_bead_fails():
    r = run(["bash", SCRIPT, "HEAD", "gpu-sim-nonexistent-bead-xyz"])
    assert r.returncode != 0

def test_gpu_hinted_skips():
    branch = "test-jvs-gpuhinted"
    run(["git", "checkout", "-b", branch])
    try:
        os.makedirs(os.path.join(REPO, "experiments", "gpu-sim-gpuhint"), exist_ok=True)
        with open(os.path.join(REPO, "experiments", "gpu-sim-gpuhint", "run.sh"), "w") as f:
            f.write("#!/bin/bash\n# GPU-HINTED\nts --max-runtime 10m python3 -c 'print(1)'\n")
        run(["git", "add", "experiments/gpu-sim-gpuhint/run.sh"])
        run(["git", "commit", "-m", "test: gpu hinted"])
        r = run(["bash", SCRIPT, "HEAD", "gpu-sim-gpuhint"])
        assert r.returncode == 0, r.stderr + r.stdout
        assert "SKIP" in r.stdout
    finally:
        run(["git", "checkout", "lab/gpu-sim-jvs"])
        run(["git", "branch", "-D", branch])
        import shutil
        shutil.rmtree(os.path.join(REPO, "experiments", "gpu-sim-gpuhint"), ignore_errors=True)

def test_uncommitted_dependency_fails():
    # Create a commit whose run.sh needs an uncommitted file.
    branch = "test-jvs-uncommitted"
    run(["git", "checkout", "-b", branch])
    try:
        os.makedirs(os.path.join(REPO, "experiments", "gpu-sim-testdep"), exist_ok=True)
        with open(os.path.join(REPO, "experiments", "gpu-sim-testdep", "run.sh"), "w") as f:
            f.write("#!/bin/bash\nset -e\ncat experiments/gpu-sim-testdep/MISSING_FILE.txt\n")
        run(["git", "add", "experiments/gpu-sim-testdep/run.sh"])
        run(["git", "commit", "-m", "test: uncommitted dep"])
        # Do NOT commit MISSING_FILE.txt (uncommitted dependency)
        r = run(["bash", SCRIPT, "HEAD", "gpu-sim-testdep"])
        assert r.returncode != 0, "expected failure for uncommitted dependency"
    finally:
        run(["git", "checkout", "lab/gpu-sim-jvs"])
        run(["git", "branch", "-D", branch])
        shutil.rmtree(os.path.join(REPO, "experiments", "gpu-sim-testdep"), ignore_errors=True)

if __name__ == "__main__":
    test_self_contained_passes()
    test_missing_bead_fails()
    test_gpu_hinted_skips()
    test_uncommitted_dependency_fails()
    print("OK")
