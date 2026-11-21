"""Guard: the append-only reaper removes recorded stale paths by name (gpu-sim-k3gb).

gpu-sim-x7cy: these tests used the REPO's shared registry (.lab/tmp/appendonly-worktrees.jsonl),
and one of them asserted that file was EMPTY afterwards. That holds only on an idle box: main's
FULL gate went red on ca979688 because a concurrent agent's live run held a record in the same
file --

    assert record.read_text().strip() == "", "record not cleared"
    E  assert '{"path": ".../implementer/lab-suite-anj39pz4/appendonly-wt-wx71pjq...": 281239941}' == ''

-- so the verdict measured the OTHER AGENT'S WORK rather than the reaper. Every test here now runs
against its OWN registry via APPENDONLY_MAIN_ROOT (support the product already has), and asserts
about its OWN record. One of them asserts the property the global assertion could never state: a
FOREIGN record is left untouched.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAPER = ROOT / "scripts" / "reap_appendonly_worktrees.py"
RECORDER = ROOT / "scripts" / "appendonly_worktrees.py"


def _isolated_root():
    """A scratch main-root holding its own .lab/tmp/appendonly-worktrees.jsonl."""
    scratch = Path(tempfile.mkdtemp(prefix="k3gb-root-", dir=ROOT / ".lab" / "tmp"))
    (scratch / ".lab" / "tmp").mkdir(parents=True)
    return scratch


def _env(scratch):
    env = dict(os.environ)
    env["APPENDONLY_MAIN_ROOT"] = str(scratch)
    return env


def _registry(scratch):
    return scratch / ".lab" / "tmp" / "appendonly-worktrees.jsonl"


def _lines(scratch):
    path = _registry(scratch)
    if not path.exists():
        return []
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def test_registry_anchor_honours_env(monkeypatch):
    """The env anchor is what makes these tests isolable at all -- pin the property.

    Written as a direct property test rather than a CLI probe: the module computes its registry
    from APPENDONLY_MAIN_ROOT, so with the anchor set the path MUST land inside the scratch root
    and nowhere near the repo's real registry.
    """
    import importlib.util
    scratch = _isolated_root()
    try:
        spec = importlib.util.spec_from_file_location("only_wt_x7cy", RECORDER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        monkeypatch.setenv("APPENDONLY_MAIN_ROOT", str(scratch))
        got = Path(mod.registry_path())
        assert got == _registry(scratch), f"anchor ignored: registry_path() -> {got}"
        real = ROOT / ".lab" / "tmp" / "appendonly-worktrees.jsonl"
        assert got != real, "registry_path() still points at the repo's shared registry"
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def test_reap_removes_recorded_path():
    scratch = _isolated_root()
    wt_dir = Path(tempfile.mkdtemp(prefix="k3gb-reap-", dir=ROOT / ".lab" / "tmp"))
    wt = wt_dir / "wt"
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(wt), "HEAD"],
                   check=True, capture_output=True)
    try:
        # a FOREIGN record, live, standing in the same registry: it must survive the reap
        foreign = {"path": str(wt_dir / "foreign-wt"), "pid": os.getpid(), "starttime": None}
        _registry(scratch).write_text(json.dumps(foreign) + "\n")

        subprocess.run([sys.executable, str(RECORDER), "record", str(wt),
                        "--pid", "999999", "--repo", str(scratch)],
                       cwd=str(ROOT), check=True, capture_output=True, env=_env(scratch))
        assert len(_lines(scratch)) == 2, "expected the foreign record plus ours"

        r = subprocess.run([sys.executable, str(REAPER), "--repo", str(scratch)],
                           capture_output=True, text=True, env=_env(scratch))
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert str(wt) in r.stdout, "the reaped path was not named in the output"
        assert not wt.exists(), "recorded path not removed"

        remaining = _lines(scratch)
        assert all(json.loads(ln)["path"] != str(wt) for ln in remaining), "our record not cleared"
        assert any(json.loads(ln)["path"] == foreign["path"] for ln in remaining), (
            "a FOREIGN record was removed by a reap that did not own it")
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "prune"], capture_output=True, text=True)
        import shutil
        shutil.rmtree(wt_dir, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)


def test_live_process_not_reaped():
    scratch = _isolated_root()
    with tempfile.TemporaryDirectory(dir=ROOT / ".lab" / "tmp") as td:
        td = Path(td)
        wt = td / "live-wt"
        wt.mkdir()
        (wt / "x.txt").write_text("hi")
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            subprocess.run([sys.executable, str(RECORDER), "record", str(wt),
                            "--pid", str(proc.pid), "--repo", str(scratch)],
                           cwd=str(ROOT), check=True, capture_output=True, env=_env(scratch))
            r = subprocess.run([sys.executable, str(REAPER), "--repo", str(scratch)],
                               capture_output=True, text=True, env=_env(scratch))
            assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
            assert wt.exists(), "live-process path was reaped"
        finally:
            proc.kill()
            proc.wait()
            import shutil
            shutil.rmtree(scratch, ignore_errors=True)


def test_corrupt_registry_fails():
    scratch = _isolated_root()
    try:
        _registry(scratch).write_text("{not json\n")
        r = subprocess.run([sys.executable, str(REAPER), "--repo", str(scratch)],
                           capture_output=True, text=True, env=_env(scratch))
        assert r.returncode != 0, "corrupt registry should exit nonzero"
        assert "corrupt" in (r.stdout + r.stderr).lower()
        assert _registry(scratch).read_text() == "{not json\n", (
            "a corrupt registry was rewritten instead of being reported")
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def test_no_record_file_is_ok():
    scratch = _isolated_root()
    try:
        r = subprocess.run([sys.executable, str(REAPER), "--repo", str(scratch)],
                           capture_output=True, text=True, env=_env(scratch))
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "no recorded append-only worktrees" in (r.stdout + r.stderr).lower()
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
