"""Repo-wide pytest configuration (gpu-sim-6gu).

Registers a 'slow' marker and enforces a per-test wall-clock guard so no
test can hang the suite the way test_overfit_111 used to.
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

# gpu-sim-x1v: refuse to run tests against a stale libfakegpu.so
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.check_libfakegpu_fresh import assert_fresh


REPO = Path(__file__).resolve().parent


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True)


def pytest_configure(config):
    assert_fresh()
    # gpu-sim-ztree: a test once ran `git checkout -b` + `git commit` in the MAIN tree and
    # left the whole repository checked out on a test branch (taking master's HEAD with
    # it). Record which branch/commit the suite started on so the session can be checked
    # afterwards -- and warn loudly if the suite is launched from a detached HEAD, which is
    # how the hijack went unnoticed.
    top = _git("rev-parse", "--show-toplevel").stdout.strip()
    if top and os.path.realpath(top) == os.path.realpath(str(REPO)):
        config._start_branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        config._start_commit = _git("rev-parse", "HEAD").stdout.strip()
        if config._start_branch == "HEAD":
            print("\n[conftest] WARNING: pytest is running on a DETACHED HEAD. If the suite "
                  "finishes on a different branch than it started, a test mutated the shared "
                  "repository (see gpu-sim-ztree).", file=sys.stderr)
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")


def _handler(signum, frame):
    raise TimeoutError("per-test wall budget exceeded (120s)")


@pytest.fixture(autouse=True)
def _per_test_timeout():
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(120)
    yield
    signal.alarm(0)


def pytest_sessionfinish(session, exitstatus):
    """Fail loudly if the suite left the repository somewhere else than it found it."""
    start_branch = getattr(session.config, "_start_branch", None)
    if not start_branch:
        return                      # not the main tree (a scratch worktree): nothing to guard
    now_branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    now_commit = _git("rev-parse", "HEAD").stdout.strip()
    if now_branch != start_branch:
        session.exitstatus = 1       # make the suite FAIL, not just complain
        print(f"\n[conftest] ERROR: the suite moved HEAD from {start_branch!r} to "
              f"{now_branch!r}. Tests must not mutate the shared repository; use a scratch "
              f"git worktree (see tests/test_verify_committed_run.py).", file=sys.stderr)
