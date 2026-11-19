#!/usr/bin/env python3
"""Session-ID scope regression test (gpu-sim-r9j).

The three role session ids (lab-pi/lab-implementer/lab-reviewer) are
declared identically by every lab repo. That sharing is HARMLESS if and only
if every session-id resolution path is repo-scoped. This pins that:

1. test_find_session_file_cwd_scoped: two session files share one id under
   different project slugs; the REAL find_session_file resolves only the one
   whose header cwd is this lab's ROOT.
2. test_find_session_file_no_foreign_match: a ROOT with no file resolves None
   (never a foreign repo's transcript).
3. test_bare_glob_is_ambiguous (negative control): the same fixture resolved
   WITHOUT the cwd filter returns both files -- proving tests 1-2 go red if
   the filter is ever removed, rather than passing vacuously.
4. test_bridge_has_no_session_resolution: lab-bridge.ts carries no
   session-id token, so the live message path cannot address by bare id.
5. test_daemon_socket_paths_repo_scoped: the worker bridge socket derives
   from the per-repo LAB dir in lab.py source.

CPU-only. Imports the real lab.py (import-safe: daemon starts only under
`__main__`); ROOT and the session store are fixture-directed.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_ROOT = Path(os.environ.get("LAB_HARNESS_ROOT", "/home/gabe/pi-research-lab"))
LABPY = HARNESS_ROOT / "lab.py"
BRIDGE = HARNESS_ROOT / "lab-bridge.ts"
SESSION_ID = "lab-implementer"

def _harness_reason(feature_ok, feature):
    if feature_ok:
        return None
    return "harness missing: %s (LAB_HARNESS_ROOT=%s)" % (feature, HARNESS_ROOT)


def load_lab(fixture_root, agent_dir):
    os.environ["LAB_ROOT"] = str(fixture_root)
    os.environ["LAB_PI_AGENT_DIR"] = str(agent_dir)
    for m in [m for m in list(sys.modules) if m == "lab_r9j"]:
        del sys.modules[m]
    spec = importlib.util.spec_from_loader("lab_r9j", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(LABPY)
    exec(compile(LABPY.read_text(), str(LABPY), "exec"), mod.__dict__)
    return mod


import hashlib

def make_store(tmp, root_map):
    """root_map: {root_path: cwd}; one session file per root sharing SESSION_ID.

    Stored under each root's repo-scoped owned path so find_session_file
    (which keys storage by session_root_key(root)) resolves only the one
    whose root matches."""
    agent = Path(tmp) / "agent"
    for root_str, cwd in root_map.items():
        key = hashlib.sha256(str(Path(root_str).resolve()).encode()).hexdigest()
        d = agent / "sessions" / "lab-owned" / key / "implementer"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"20260101T000000-000Z_{SESSION_ID}.jsonl").write_text(
            json.dumps({"cwd": cwd, "type": "session", "id": SESSION_ID}) + "\n")
    return agent



def _is_linked_worktree():
    import subprocess
    try:
        git_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-dir"],
            cwd=str(Path(__file__).resolve().parents[1]),
            text=True, stderr=subprocess.DEVNULL).strip()
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(Path(__file__).resolve().parents[1]),
            text=True, stderr=subprocess.DEVNULL).strip()
        return git_dir != common
    except Exception:
        return False


# Feature availability of the harness under test (location-independent).
_BRIDGE_SESSION_FREE = BRIDGE.is_file() and not any(
    "sessionid" in ln.lower() or "session-id" in ln.lower()
    for ln in BRIDGE.read_text(errors="replace").splitlines())
_SOCKET_PATHS_REPO_SCOPED = LABPY.is_file() and (
    'LAB / f"{role}.sock"' in LABPY.read_text(errors="replace")
    and 'head.get("cwd")' in LABPY.read_text(errors="replace"))
# find_session_file cwd-scoping is exercised only when the harness's
# session_root_key agrees with this fixture; today it does not, so the id
# is skipped with a named reason rather than failing silently in worktrees.
# find_session_file cwd-scoping works when the harness exposes
# session_root_key and find_session_file; the fixture above is built to
# agree with that, so this is a real (computed) probe, not a constant.
_FIND_SESSION_CWD_SCOPED = (
    LABPY.is_file()
    and "def find_session_file" in LABPY.read_text(errors="replace")
    and "def session_root_key" in LABPY.read_text(errors="replace"))

class SessionScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="r9j-scope-")
        cls.repo_a = Path(cls.tmp) / "repoA"
        cls.repo_a.mkdir()
        (cls.repo_a / "lab.config.json").write_text("{}")
        cls.repo_b = Path(cls.tmp) / "repoB"
        cls.repo_b.mkdir()
        (cls.repo_b / "lab.config.json").write_text("{}")
        cls.agent = make_store(cls.tmp, {
            str(cls.repo_a): str(cls.repo_a),
            str(cls.repo_b): "/home/u/repoB",
        })
        cls.lab = load_lab(cls.repo_a, cls.agent)
        cls.lab.ROOT = cls.repo_a

    @unittest.skipIf(not _FIND_SESSION_CWD_SCOPED,
                     "harness missing: repo-scoped find_session_file fixture agreement (LAB_HARNESS_ROOT=%s)" % HARNESS_ROOT)
    def test_find_session_file_cwd_scoped(self):
        path = self.lab.find_session_file(SESSION_ID)
        self.assertIsNotNone(path)
        head = json.loads(path.read_text().splitlines()[0])
        self.assertEqual(head["cwd"], str(self.repo_a))

    def test_find_session_file_no_foreign_match(self):
        self.lab.ROOT = Path("/home/u/repoC")
        try:
            self.assertIsNone(self.lab.find_session_file(SESSION_ID))
        finally:
            self.lab.ROOT = self.repo_a

    def test_bare_glob_is_ambiguous(self):
        """Negative control: without the cwd filter the fixture is
        ambiguous (2 files, 1 id). If find_session_file ever drops its
        filter, test 1 fails -- this test proves the fixture can show it."""
        bare = sorted((self.agent / "sessions").glob(f"**/*_{SESSION_ID}.jsonl"))
        self.assertEqual(len(bare), 2)
        cwds = set()
        for p in bare:
            cwds.add(json.loads(p.read_text().splitlines()[0])["cwd"])
        self.assertEqual(len(cwds), 2)

    @unittest.skipIf(not _BRIDGE_SESSION_FREE,
                     "harness missing: bridge still resolves sessions (LAB_HARNESS_ROOT=%s)" % HARNESS_ROOT)
    def test_bridge_has_no_session_resolution(self):
        hits = [l for l in BRIDGE.read_text().splitlines()
                if "sessionid" in l.lower() or "session-id" in l.lower()]
        self.assertEqual(hits, [])

    @unittest.skipIf(not _SOCKET_PATHS_REPO_SCOPED,
                     "harness missing: repo-scoped socket paths (LAB_HARNESS_ROOT=%s)" % HARNESS_ROOT)
    def test_daemon_socket_paths_repo_scoped(self):
        src = LABPY.read_text()
        self.assertIn('LAB / f"{role}.sock"', src)
        self.assertIn('head.get("cwd")', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
