#!/usr/bin/env python3
"""Regression test for gpu-sim-jvs: verify_committed_run detects uncommitted deps.

ISOLATION (gpu-sim-ztree): these tests used to run `git checkout -b` / `git commit` in
REPO -- the MAIN tree -- and their cleanup ended with `git checkout lab/gpu-sim-jvs`, a
branch that does not exist in the main tree. When anything failed before the cleanup
(or the cleanup itself failed, which it silently did), the main tree was left checked out
on a test branch, carrying test commits, and master had gained junk commits
("test: gpu hinted", "test: uncommitted dep"). That is how a test run moved HEAD of the
shared repository.

Every mutating case now runs inside a scratch `git worktree`, so the main tree's HEAD,
index and working copy are never touched; verify_committed_run.sh resolves its repo from
its CWD, so invoking it with cwd=<worktree> keeps everything inside the sandbox.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "verify_committed_run.sh")


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd or REPO, capture_output=True, text=True)


@pytest.fixture()
def sandbox():
    """A detached worktree of REPO. Mutations here cannot touch the main tree."""
    d = tempfile.mkdtemp(prefix="jvs-wt-", dir=os.path.join(REPO, ".lab", "tmp"))
    add = run(["git", "worktree", "add", "--detach", d, "HEAD"])
    if add.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        pytest.skip(f"cannot create scratch worktree: {add.stderr}")
    try:
        yield d
    finally:
        run(["git", "worktree", "remove", "--force", d])
        run(["git", "worktree", "prune"])
        shutil.rmtree(d, ignore_errors=True)


def test_self_contained_passes():
    r = run(["bash", SCRIPT, "HEAD", "gpu-sim-0i8"])
    assert r.returncode == 0, r.stderr


def test_missing_bead_fails():
    r = run(["bash", SCRIPT, "HEAD", "gpu-sim-nonexistent-bead-xyz"])
    assert r.returncode != 0


def test_gpu_hinted_skips(sandbox):
    bead = "gpu-sim-gpuhint"
    os.makedirs(os.path.join(sandbox, "experiments", bead), exist_ok=True)
    # a nonce keeps the fixture distinct from any copy already in history, so the
    # commit below always has something to record
    with open(os.path.join(sandbox, "experiments", bead, "run.sh"), "w") as f:
        f.write("#!/bin/bash\n# GPU-HINTED\n# fixture {}\nts --max-runtime 10m "
                "python3 -c 'print(1)'\n".format(uuid.uuid4()))
    assert run(["git", "add", f"experiments/{bead}/run.sh"], cwd=sandbox).returncode == 0
    assert run(["git", "commit", "-m", "test: gpu hinted"], cwd=sandbox).returncode == 0
    r = run(["bash", SCRIPT, "HEAD", bead], cwd=sandbox)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "SKIP" in r.stdout


def test_uncommitted_dependency_fails(sandbox):
    bead = "gpu-sim-testdep"
    os.makedirs(os.path.join(sandbox, "experiments", bead), exist_ok=True)
    with open(os.path.join(sandbox, "experiments", bead, "run.sh"), "w") as f:
        f.write("#!/bin/bash\nset -e\n# fixture {}\n"
                "cat experiments/gpu-sim-testdep/MISSING_FILE.txt\n".format(uuid.uuid4()))
    assert run(["git", "add", f"experiments/{bead}/run.sh"], cwd=sandbox).returncode == 0
    assert run(["git", "commit", "-m", "test: uncommitted dep"], cwd=sandbox).returncode == 0
    # MISSING_FILE.txt is deliberately left uncommitted
    r = run(["bash", SCRIPT, "HEAD", bead], cwd=sandbox)
    assert r.returncode != 0, "expected failure for uncommitted dependency"


if __name__ == "__main__":
    test_self_contained_passes()
    test_missing_bead_fails()
    sys.exit(0)
