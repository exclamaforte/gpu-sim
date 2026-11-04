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
import atexit

# gpu-sim-x1v: refuse to run tests against a stale libfakegpu.so
REPO = Path(__file__).resolve().parent
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
        _restore_tree_to_head()
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


_PLANT_TARGETS = (
    "docs/model_facts.json",
    "tests/fs2y_ledger_meta.json",
    "experiments/gpu-sim-u5r4/metrics.json",
    "experiments/gpu-sim-wty2/metrics.json",
    "tests/test_vacuous_plant.py",
)


def _restore_tree_to_head():
    # Never restore in a checker child: mutation replay relies on the mutated
    # file surviving until the target test runs (gpu-sim-zsd3).
    if os.environ.get("OI4M_CHECKER_PID"):
        return
    if os.environ.get("LAB_NO_TREE_JANITOR"):
        return
    top = _git("rev-parse", "--show-toplevel").stdout.strip()
    if not top or os.path.realpath(top) != os.path.realpath(str(REPO)):
        return
    for rel in _PLANT_TARGETS:
        head = _git("show", "HEAD:" + rel)
        if head.returncode == 0:
            try:
                (REPO / rel).parent.mkdir(parents=True, exist_ok=True)
                (REPO / rel).write_text(head.stdout)
            except OSError:
                pass
    ls = _git("ls-tree", "-r", "--name-only", "HEAD", "--", "tests/fs2y_ledger.d/")
    for rel in (ls.stdout or "").splitlines():
        rel = rel.strip()
        if not rel:
            continue
        if not (REPO / rel).exists():
            head = _git("show", "HEAD:" + rel)
            if head.returncode == 0:
                try:
                    (REPO / rel).parent.mkdir(parents=True, exist_ok=True)
                    (REPO / rel).write_text(head.stdout)
                except OSError:
                    pass
    status = _git("status", "--porcelain")
    for line in (status.stdout or "").splitlines():
        if len(line) < 4:
            continue
        pp = line[3:].strip()
        if " -> " in pp:
            pp = pp.split(" -> ", 1)[1]
        pp = pp.strip('"')
        name = Path(pp).name
        if name.startswith(("plant_", "tmp")) or name.endswith((".tmp", ".pyc")):
            try:
                (REPO / pp).unlink()
            except OSError:
                pass


@pytest.fixture(autouse=True)
def _tree_janitor(request):
    """Put back any repo file a test modified but did not restore (gpu-sim-zsd3).

    The plant tests and the mutation replay deliberately corrupt a tracked file,
    run a checker, and restore it in `finally`. Under an interrupted run (OOM
    kill, timeout) the `finally` never executes, the payload survives, and every
    later test in the same session -- and the gate's residue commit -- inherits
    it. Two real examples: experiments/gpu-sim-u5r4/metrics.json left holding the
    mutation payload gbt_mae=999, and docs/model_facts.json left missing bch9's
    fact block, which the gate then committed as if the branch had dropped it.

    This janitor makes every test start from the tree state it inherited: any
    tracked file that was clean before the test and dirty after is restored from
    HEAD, and known plant/temp debris is removed. It warns loudly with the node
    id so the polluter is still visible instead of being silently papered over.
    """
    if os.environ.get("LAB_NO_TREE_JANITOR"):
        yield
        return
    root = Path(__file__).resolve().parent
    before = _dirty_paths()
    yield
    leaked = _dirty_paths() - before
    if not leaked:
        return
    restored = []
    for rel in sorted(leaked):
        if _git("ls-files", "--error-unmatch", rel).returncode == 0:
            head = _git("show", "HEAD:" + rel)
            if head.returncode == 0:
                (root / rel).write_text(head.stdout)
                restored.append(rel)
                continue
        name = Path(rel).name
        if name.startswith(("plant_", "tmp")) or name.endswith((".tmp", ".pyc")):
            try:
                (root / rel).unlink()
            except OSError:
                pass
            restored.append(rel + " (removed)")
    if restored:
        print(f"\n[conftest] tree janitor: {request.node.nodeid} left {restored}; "
              f"restored so the next test starts clean (gpu-sim-zsd3)", file=sys.stderr)


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
    _restore_tree_to_head()


def _restore_signal(signum, frame):
    _restore_tree_to_head()
    raise SystemExit(128 + signum)


for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _restore_signal)
    except (ValueError, OSError):
        pass
atexit.register(_restore_tree_to_head)
