"""gpu-sim-c3hd: the scratch reaper must never delete a live test's own directory.

The pile this reaps is real (1.4 GiB of lab-suite-*, 1.8 GiB of tmp* when it was
first measured), so the classifier is the whole tool: a reaper keyed on age alone
would have deleted scratch out from under seven live processes that were running
when the first version of this file was written. Every arm below therefore states
the exact reason a directory is kept, not just that it survived.

Fixture layout (all under tmp_path, real /proc, no repo writes):

    lab-suite-old       5 days old, nobody owns it          -> DELETE
    lab-suite-fresh     brand new                           -> KEEP (age floor)
    lab-suite-live      5 days old, a child process's CWD   -> KEEP (live owner)
    lab-suite-arg       5 days old, named on a cmdline      -> KEEP (live owner)
    lab-suite-recorded  5 days old, registry entry, alive   -> KEEP (registered)
    lab-suite-deadpid   brand new, registry entry, dead pid -> DELETE (record proof)
    lab-suite-link      symlink to an outside dir, 5 days   -> KEEP (never followed)
    notes.txt           a file, 5 days old                  -> untouched
    checkout/lab-suite-x  inside a tree with .git           -> KEEP (pruned)

Dry run is the default, so the first assertion is that nothing was removed.
"""
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAPER = ROOT / "scripts" / "reap_scratch_labs.py"
OLD = 5 * 24 * 3600


def _mk(path, age=0):
    path.mkdir(parents=True, exist_ok=True)
    (path / "payload.bin").write_bytes(b"x" * 4096)
    if age:
        t = time.time() - age
        os.utime(path, (t, t))
    return path


def _run(root, *args):
    r = subprocess.run([sys.executable, str(REAPER), "--root", str(root), *args],
                       capture_output=True, text=True, timeout=120)
    return r


class TestScratchReaper(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory(dir=str(ROOT / ".lab" / "tmp"))
        self.root = Path(self._td.name) / "tmp"
        self.root.mkdir()
        self.old = _mk(self.root / "implementer" / "lab-suite-old", OLD)
        self.fresh = _mk(self.root / "implementer" / "lab-suite-fresh")
        self.live = _mk(self.root / "implementer" / "lab-suite-live", OLD)
        self.arg = _mk(self.root / "implementer" / "lab-suite-arg", OLD)
        self.recorded = _mk(self.root / "implementer" / "lab-suite-recorded", OLD)
        self.deadpid = _mk(self.root / "implementer" / "lab-suite-deadpid")
        self.outside = Path(self._td.name) / "outside"
        self.outside.mkdir()
        (self.outside / "keepme").write_text("do not delete")
        self.link = self.root / "implementer" / "lab-suite-link"
        self.link.symlink_to(self.outside)
        (self.root / "implementer" / "notes.txt").write_text("not a dir")
        t = time.time() - OLD
        os.utime(self.root / "implementer" / "notes.txt", (t, t))
        self.checkout = self.root / "reviewer" / "checkout"
        (self.checkout / ".git").mkdir(parents=True)
        _mk(self.checkout / "experiments" / "lab-suite-x", OLD)

        # a live child whose CWD is one of the scratch dirs
        self.child = subprocess.Popen(["sleep", "60"], cwd=str(self.live))
        # a live child whose ARGUMENT names another scratch dir
        self.argchild = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", str(self.arg)])
        # registry: one entry for a live pid, one for a pid that cannot be alive
        rec_ok = _run(self.root, "record", str(self.recorded),
                      "--pid", str(self.child.pid))
        self.assertEqual(rec_ok.returncode, 0, rec_ok.stderr)
        dead = subprocess.Popen(["true"])
        dead.wait()
        rec_dead = _run(self.root, "record", str(self.deadpid), "--pid", str(dead.pid))
        self.assertEqual(rec_dead.returncode, 0, rec_dead.stderr)

    def tearDown(self):
        for p in (self.child, self.argchild):
            try:
                p.kill()
                p.wait(timeout=10)
            except Exception:
                pass
        self._td.cleanup()

    # -- the arms ------------------------------------------------------------
    def test_dry_run_deletes_nothing_and_names_the_reasons(self):
        r = _run(self.root, "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        summary = json.loads(r.stdout)
        cands = {Path(c["path"]).name for c in summary["candidates"]}
        kept = {Path(c["path"]).name: c["why"] for c in summary["kept"]}
        self.assertEqual(cands, {"lab-suite-old", "lab-suite-deadpid"}, summary)
        self.assertIn("lab-suite-fresh", kept)
        self.assertIn("younger than", kept["lab-suite-fresh"])
        self.assertIn("lab-suite-live", kept)
        self.assertIn("live process", kept["lab-suite-live"])
        self.assertIn("lab-suite-arg", kept)
        self.assertIn("live process", kept["lab-suite-arg"],
                      "the cmdline-named scratch must be kept BY THE LIVE-OWNER rule")
        self.assertIn("registered owner is alive", kept["lab-suite-recorded"])
        # the dry run must have removed nothing at all
        for p in (self.old, self.fresh, self.live, self.arg, self.recorded,
                  self.deadpid, self.link):
            self.assertTrue(p.exists() or p.is_symlink(), f"{p} vanished in a dry run")
        self.assertTrue((self.checkout / "experiments" / "lab-suite-x").exists())

    def test_apply_removes_exactly_the_unowned_directories(self):
        r = _run(self.root, "--apply", "--json")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        summary = json.loads(r.stdout)
        deleted = {Path(c["path"]).name for c in summary["deleted"]}
        self.assertEqual(deleted, {"lab-suite-old", "lab-suite-deadpid"}, summary)
        self.assertFalse(self.old.exists())
        self.assertFalse(self.deadpid.exists())
        for p in (self.fresh, self.live, self.arg, self.recorded):
            self.assertTrue(p.exists(), f"{p} was deleted while it was owned")
        # a symlink is never followed: the outside dir and its file survive
        self.assertTrue(self.link.is_symlink())
        self.assertEqual((self.outside / "keepme").read_text(), "do not delete")
        # a checked-out tree is pruned, not reaped
        self.assertTrue((self.checkout / "experiments" / "lab-suite-x").exists())
        # the plain file was never a candidate
        self.assertTrue((self.root / "implementer" / "notes.txt").exists())

    def test_a_live_process_whose_cwd_is_the_PARENT_owns_a_young_scratch(self):
        """Reviewer probe: a child cwd=<root>/implementer mid-run on
        <root>/implementer/lab-suite-cwd-parent. The old classifier deleted it."""
        parent = self.root / "implementer"
        scratch = parent / "lab-suite-cwd-parent"
        scratch.mkdir()
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import time; time.sleep(60)"], cwd=str(parent))
        try:
            r = _run(self.root, "--json", "--older-than-hours", "0")
            summary = json.loads(r.stdout)
            kept = {Path(c["path"]).name: c["why"] for c in summary["kept"]}
            self.assertIn("lab-suite-cwd-parent", kept, summary)
            self.assertIn("live process", kept["lab-suite-cwd-parent"])
        finally:
            proc.kill()
            proc.wait()

    def test_a_live_process_naming_the_PARENT_on_its_cmdline_owns_a_young_scratch(self):
        """Reviewer probe: --basetemp <root>/implementer, scratch .../pytest-live-basetemp."""
        parent = self.root / "implementer"
        scratch = parent / "pytest-live-basetemp"
        scratch.mkdir()
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import time; time.sleep(60)", "--basetemp",
                                 str(parent)])
        try:
            r = _run(self.root, "--json", "--older-than-hours", "0")
            summary = json.loads(r.stdout)
            kept = {Path(c["path"]).name: c["why"] for c in summary["kept"]}
            self.assertIn("pytest-live-basetemp", kept, summary)
            self.assertIn("live process", kept["pytest-live-basetemp"])
        finally:
            proc.kill()
            proc.wait()

    def test_a_scratch_OLDER_than_the_parent_holder_is_still_a_candidate(self):
        """The parent rule must not become 'nothing is ever reaped': the fixture's own
        five-day-old siblings under the same parent stay candidates while the parent is
        held by a live process."""
        scratch = self.root / "implementer" / "lab-suite-cwd-old-sibling"
        scratch.mkdir()
        os.utime(scratch, (time.time() - 5 * 86400,) * 2)
        parent = self.root / "implementer"
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import time; time.sleep(60)"], cwd=str(parent))
        try:
            r = _run(self.root, "--json")
            summary = json.loads(r.stdout)
            cands = {Path(c["path"]).name for c in summary["candidates"]}
            self.assertIn("lab-suite-cwd-old-sibling", cands, summary)
        finally:
            proc.kill()
            proc.wait()

    def test_age_floor_is_the_only_reason_the_fresh_dir_survives(self):
        """Lowering the floor must make it a candidate: proves the floor is what
        kept it, rather than a classifier accident."""
        r = _run(self.root, "--older-than-hours", "0", "--json")
        summary = json.loads(r.stdout)
        cands = {Path(c["path"]).name for c in summary["candidates"]}
        self.assertIn("lab-suite-fresh", cands, summary)
        # ...and the live/registered ones are still kept
        self.assertNotIn("lab-suite-live", cands)
        self.assertNotIn("lab-suite-recorded", cands)

    def test_a_dead_pid_record_does_not_keep_a_fresh_leak(self):
        """The leak this bead exists for: a harness that recorded its pid died.
        Age alone would keep the directory for two more days."""
        # a fresh dir (default 48h floor would keep it for two more days)
        d = _mk(self.root / "implementer" / "lab-suite-deadpid2")
        dead = subprocess.Popen(["true"])
        dead.wait()
        _run(self.root, "record", str(d), "--pid", str(dead.pid))
        r = _run(self.root, "--older-than-hours", "168", "--json")
        cands = {Path(c["path"]).name for c in json.loads(r.stdout)["candidates"]}
        self.assertIn("lab-suite-deadpid2", cands)


if __name__ == "__main__":
    unittest.main(verbosity=2)
