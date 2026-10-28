import os
import subprocess
import tempfile
import json
import shutil
import io
import contextlib
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_gate_citations as cgc


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True)


def _make_repo():
    d = Path(tempfile.mkdtemp())
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "t@t.com")
    _git(d, "config", "user.name", "t")
    (d / "experiments" / "x").mkdir(parents=True)
    (d / "code.py").write_text("v1\n")
    (d / "experiments" / "x" / "metrics.json").write_text(
        json.dumps({"gate": "placeholder"}) + "\n")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "base")
    base = _git(d, "rev-parse", "HEAD").stdout.strip()
    return d, base


def _commit(repo, msg, files):
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


class TestContract(unittest.TestCase):
    def setUp(self):
        self.repo, self.base = _make_repo()
        self.head = _commit(self.repo, "head", {
            "code.py": "v2\n",
            "experiments/x/metrics.json": json.dumps({"gate": "placeholder2"}) + "\n",
        })
        self.old_repo = cgc.REPO
        cgc.REPO = self.repo

    def tearDown(self):
        cgc.REPO = self.old_repo
        shutil.rmtree(self.repo, ignore_errors=True)

    def _make_runlog(self, handle, tip, with_end=True):
        d = self.repo / ".lab" / "runs" / handle
        d.mkdir(parents=True, exist_ok=True)
        lines = [f"GATE TIP: {tip}"]
        if with_end:
            lines.append("GATE DIRTY-AT-END: abc123")
        (d / "run.log").write_text("\n".join(lines) + "\n")

    def _set_metrics(self, gate_str):
        (self.repo / "experiments" / "x" / "metrics.json").write_text(
            json.dumps({"gate": gate_str}) + "\n")

    def test_tip_equals_head_returns_zero(self):
        self._make_runlog("r1", self.head, with_end=True)
        self._set_metrics(f"scripts/gate.sh 1/1 PASS (run r1, tip {self.head})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 0)

    def test_nonexistent_tip_returns_one(self):
        self._set_metrics("scripts/gate.sh 1/1 PASS (run r1, tip 0000000000000000)")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 1)
        self.assertIn("INVALID", buf.getvalue())

    def test_ancestor_nonmetrics_returns_one(self):
        self._make_runlog("r1", self.base, with_end=True)
        self._set_metrics(f"scripts/gate.sh 1/1 PASS (run r1, tip {self.base})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 1)
        self.assertIn("real code change", buf.getvalue())

    def test_ancestor_metrics_only_returns_zero(self):
        repo2, base2 = _make_repo()
        head2 = _commit(repo2, "metrics-only", {
            "experiments/x/metrics.json": json.dumps({"gate": "x"}) + "\n",
        })
        old = cgc.REPO
        cgc.REPO = repo2
        # create runlog in repo2
        rd = repo2 / ".lab" / "runs" / "r1"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "run.log").write_text(f"GATE TIP: {base2}\nGATE DIRTY-AT-END: abc\n")
        (repo2 / "experiments" / "x" / "metrics.json").write_text(
            json.dumps({"gate": f"scripts/gate.sh 1/1 PASS (run r1, tip {base2})"}) + "\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        cgc.REPO = old
        shutil.rmtree(repo2, ignore_errors=True)
        self.assertEqual(rc, 0)

    def test_unrelated_tip_returns_zero_unverifiable(self):
        # create an orphan commit unrelated to HEAD
        import subprocess as sp
        env = {"GIT_AUTHOR_NAME":"t","GIT_AUTHOR_EMAIL":"t@t.com",
               "GIT_COMMITTER_NAME":"t","GIT_COMMITTER_EMAIL":"t@t.com"}
        # write a file and commit on a new orphan branch
        (self.repo / "orphan.txt").write_text("orphan\n")
        sp.run(["git","-C",str(self.repo),"checkout","--orphan","orph"],capture_output=True)
        sp.run(["git","-C",str(self.repo),"add","orphan.txt"],capture_output=True)
        sp.run(["git","-C",str(self.repo),"commit","-q","-m","orphan"],capture_output=True,env={**os.environ,**env})
        orphan = sp.run(["git","-C",str(self.repo),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
        sp.run(["git","-C",str(self.repo),"checkout","-q","master" if False else "main"],capture_output=True)
        # go back to original branch (likely master/main); detect default
        # find the branch we were on (first branch)
        branches = sp.run(["git","-C",str(self.repo),"branch"],capture_output=True,text=True).stdout
        # checkout the original HEAD's branch by hash
        sp.run(["git","-C",str(self.repo),"checkout","-q",self.head],capture_output=True)
        self._set_metrics(f"scripts/gate.sh 1/1 PASS (run r1, tip {orphan})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 0)
        self.assertIn("UNVERIFIABLE", buf.getvalue())

    def test_runlog_mismatch_returns_one(self):
        run_dir = self.repo / ".lab" / "runs" / "r1"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.log").write_text("GATE TIP: ffffffffffffffffffffffffffffffffffffffff\n")
        self._set_metrics(f"scripts/gate.sh 1/1 PASS (run r1, tip {self.head})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        shutil.rmtree(run_dir, ignore_errors=True)
        self.assertEqual(rc, 1)
        self.assertIn("GATE TIP", buf.getvalue())


class TestTipRegex(unittest.TestCase):
    def test_tip_regex(self):
        m = cgc.TIP_RE.search("tip abc1234")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "abc1234")


class TestContractDeclared(unittest.TestCase):
    def setUp(self):
        self.repo, self.base = _make_repo()
        self.head = _commit(self.repo, "head", {
            "code.py": "v2\n",
            "experiments/x/metrics.json": json.dumps({"gate": "placeholder2"}) + "\n",
        })
        self.old_repo = cgc.REPO
        cgc.REPO = self.repo

    def tearDown(self):
        cgc.REPO = self.old_repo
        shutil.rmtree(self.repo, ignore_errors=True)

    def _make_runlog(self, handle, tip, with_end=True):
        d = self.repo / ".lab" / "runs" / handle
        d.mkdir(parents=True, exist_ok=True)
        lines = [f"GATE TIP: {tip}"]
        if with_end:
            lines.append("GATE DIRTY-AT-END: abc123")
        (d / "run.log").write_text("\n".join(lines) + "\n")

    def _set_metrics(self, gate_str):
        (self.repo / "experiments" / "x" / "metrics.json").write_text(
            json.dumps({"gate": gate_str}) + "\n")

    def _make_runlog(self, handle, tip, with_end=True):
        d = self.repo / ".lab" / "runs" / handle
        d.mkdir(parents=True, exist_ok=True)
        lines = [f"GATE TIP: {tip}"]
        if with_end:
            lines.append("GATE DIRTY-AT-END: abc123")
        (d / "run.log").write_text("\n".join(lines) + "\n")

    def test_no_tip_fails(self):
        self._set_metrics("scripts/gate.sh 1/1 PASS (no tip here)")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 1)
        self.assertIn("INVALID", buf.getvalue())

    def test_no_handle_unverifiable(self):
        self._set_metrics(f"scripts/gate.sh 1/1 PASS (tip {self.head})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 0)
        self.assertIn("UNVERIFIABLE", buf.getvalue())
        self.assertIn("no run handle", buf.getvalue())

    def test_enoent_unverifiable(self):
        self._set_metrics(f"scripts/gate.sh 1/1 PASS (run rnope, tip {self.head})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 0)
        self.assertIn("UNVERIFIABLE", buf.getvalue())
        self.assertIn("run.log missing", buf.getvalue())

    def test_missing_dirty_end_fails(self):
        self._make_runlog("r1", self.head, with_end=False)
        self._set_metrics(f"scripts/gate.sh 1/1 PASS (run r1, tip {self.head})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 1)
        self.assertIn("lacks GATE DIRTY-AT-END", buf.getvalue())

    def test_matching_log_passes(self):
        self._make_runlog("r1", self.head, with_end=True)
        self._set_metrics(f"scripts/gate.sh 1/1 PASS (run r1, tip {self.head})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 0)

    def test_declared_unverifiable_passes(self):
        self._set_metrics(f"scripts/gate.sh 1/1 PASS (run r1, tip {self.head}); UNVERIFIABLE FOR THIS TIP: delta x")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgc.main()
        self.assertEqual(rc, 0)
        self.assertIn("UNVERIFIABLE", buf.getvalue())
