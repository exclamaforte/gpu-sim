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
REPO = Path(__file__).resolve().parent

# gpu-sim-6f9q: experiments/* contains scripts whose names match test_*.py but
# which are standalone programs (they call sys.exit at import). Keep pytest from
# collecting them; they are run directly, not via the suite.
collect_ignore_glob = ["experiments/*"]
sys.path.insert(0, str(REPO))

# gpu-sim-1upd: redirect bytecode cache to a gitignored project-local tree so
# nothing lands under experiments/ (the lab's untracked-path guard would
# otherwise flag gitignored __pycache__ dirs as untracked). Subprocesses inherit
# via os.environ.setdefault. Set before conftest's own imports for tidiness.
_PYCACHE_DIR = REPO / ".lab" / "tmp" / "pycache"
_PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
sys.pycache_prefix = str(_PYCACHE_DIR)
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(_PYCACHE_DIR))

from scripts.check_libfakegpu_fresh import assert_fresh


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
        # record the starting tree state so residue can be REPORTED at the end;
        # never restore it (gpu-sim-zsd3).
        config._dirty_at_start = _dirty_paths()
        config._start_branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        config._start_commit = _git("rev-parse", "HEAD").stdout.strip()
        if config._start_branch == "HEAD":
            print("\n[conftest] WARNING: pytest is running on a DETACHED HEAD. If the suite "
                  "finishes on a different branch than it started, a test mutated the shared "
                  "repository (see gpu-sim-ztree).", file=sys.stderr)
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "multiproc: marks tests that spawn multiple processes (skip under contention)")


def _handler(signum, frame):
    raise TimeoutError("per-test wall budget exceeded (300s)")


@pytest.fixture(autouse=True)
def _per_test_timeout():
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(300)
    yield
    signal.alarm(0)


def _dirty_paths():
    """Paths `git status --porcelain` reports, relative to the repo root."""
    out = _git("status", "--porcelain")
    paths = set()
    for line in out.stdout.splitlines():
        if len(line) < 4:
            continue
        p = line[3:].strip()
        if " -> " in p:
            p = p.split(" -> ", 1)[1]
        paths.add(p.strip('"'))
    return paths


def _report_pollution(nodeid, leaked, tracked_modified, phase):
    """Name the polluter loudly. Never repairs anything (gpu-sim-zsd3)."""
    parts = list(tracked_modified) + [p for p in leaked if p not in tracked_modified]
    print(f"\n[conftest] TREE POLLUTION ({phase}): {nodeid} left {sorted(parts)}; "
          f"NOT restored -- restoring from HEAD reverts uncommitted work, which is "
          f"how this fixture reverted a real fix mid-run (gpu-sim-zsd3). The tree is "
          f"dirty now; later failures may be consequences rather than causes.",
          file=sys.stderr)


@pytest.fixture(autouse=True)
def _tree_janitor(request):
    """DETECT -- never repair -- tracked files a test left modified (gpu-sim-zsd3).

    History matters here, because the obvious implementation is the wrong one.
    This fixture used to put the tree back: it restored tracked files from HEAD
    and unlinked plant/temp debris, to stop an interrupted plant test's payload
    (gbt_mae=999 in a metrics file, a missing fact block in model_facts.json)
    from being inherited by later tests and committed by the gate.

    That repair was itself destructive. It cannot tell a payload left by a
    crashed test from UNCOMMITTED WORK, and on 2026-11-08 it reverted a real fix
    in progress: a test then failed on the very field being corrected, and
    passed afterwards testing reverted content. The branch under review had made
    it worse -- restoring a fixed target list from pytest_configure,
    pytest_sessionfinish, SIGTERM/SIGINT handlers and atexit, so ANY exit
    (including a kill) rewrote files that a human was editing.

    Upstream plant cleanup now happens OUTSIDE the tree (the plants are created
    and removed by the harness, not by a test that can be interrupted), so the
    repair is obsolete as well as unsafe.

    Ruling: report the node id and the paths, loudly; never write, never unlink,
    never git-restore. Residue that nobody names is the failure mode this exists
    to prevent -- residue that the fixture deletes is a worse one.
    """
    if os.environ.get("LAB_NO_TREE_JANITOR"):
        yield
        return
    before = _dirty_paths()
    yield
    leaked = _dirty_paths() - before
    if not leaked:
        return
    tracked_modified = []
    for rel in sorted(leaked):
        # a modified TRACKED file can be swept into a commit: that is the
        # dangerous residue, and it fails the polluting test so it cannot be
        # inherited silently.
        if _git("ls-files", "--error-unmatch", rel).returncode == 0:
            tracked_modified.append(rel)
    _report_pollution(request.node.nodeid, leaked, tracked_modified, "test-end")
    if tracked_modified:
        pytest.fail(
            f"test left tracked file(s) modified: {sorted(tracked_modified)} "
            f"(gpu-sim-zsd3). Restore them in a finally block or use a scratch "
            f"worktree; the fixture will NOT restore them for you, because that "
            f"would also revert uncommitted work.",
            pytrace=False,
        )


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
    _report_session_pollution(session)


def _report_session_pollution(session):
    """REPORT residue at session end; never repair it (gpu-sim-zsd3).

    The old session-end hook restored HEAD content into the working tree, so a
    killed run or a SIGTERM rewrote files an author was editing. Reporting
    cannot lose work; restoring can.
    """
    before = getattr(session.config, "_dirty_at_start", None)
    if before is None:
        return
    leaked = _dirty_paths() - before
    if leaked:
        _report_pollution("<session>", leaked, [], "session-end")
