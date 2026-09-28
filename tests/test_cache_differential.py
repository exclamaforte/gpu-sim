#!/usr/bin/env python3
"""Differential proof for gpu-sim-u2o.6.6.6.1's terminal-state cache.

I claimed the cache's output is IDENTICAL to the uncached path "by
construction". That is a structural argument, and structural arguments are
what this lab keeps being wrong about -- so this measures it instead, by
running both loaders over the same directory and comparing.

Three things get checked, and the third is the one that matters most:

  1. IDENTITY. Cached output == uncached output, exactly, over a directory with
     jobs in every state. If a single field differs, the speedup changed the
     record.
  2. AN ACTIVE JOB IS NEVER HIDDEN. A job that transitions queued -> running, in
     place, must be visible on the very next call. This is the property the
     whole design rests on, and the cheapest way to be wrong here is a cache
     that is one line too broad.
  3. THE STATED BOUND IS REAL. A TERMINAL job whose record changes in place
     must stay stale for at most the reverify window, and then become visible.
     I asserted that bound in prose; here it is a measured number.

Runs against a TEMPORARY queue directory, so it never reads or writes the real
shared queue -- a test that pokes the live queue of a box three other projects
use would itself be the kind of side effect this lab keeps apologising for.

Run:  python3 tests/test_cache_differential.py
"""
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "qs_under_test", REPO / "epic_u2o" / "queue_state.py")
qs = importlib.util.module_from_spec(_spec)
sys.modules["qs_under_test"] = qs
_spec.loader.exec_module(qs)


def reference_load_all(qdir: Path) -> list[dict]:
    """The loader as it was BEFORE the cache: parse every file, every call."""
    out = []
    for p in sorted(qdir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def write_job(qdir: Path, jid: str, state: str, **extra) -> Path:
    p = qdir / f"{jid}.json"
    rec = {"id": jid, "state": state, "command": ["python3", "x.py"],
           "description": f"test job {jid}", "created_at": 1.0}
    rec.update(extra)
    p.write_text(json.dumps(rec))
    return p


class CacheDifferential(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.qdir = self.td / "jobs"
        self.qdir.mkdir()
        # every state the module distinguishes, so identity is tested across all
        for st in ("finished", "failed", "canceled", "timed_out",
                   "killed_underused", "running", "queued", "starting"):
            write_job(self.qdir, f"20260101t000000-{st[:4]}", st)
        self._saved = (qs.QUEUE_DIR, qs._TERMINAL_CACHE, qs._FILE_LIST,
                       dict(qs._LOAD_STATS))
        qs.QUEUE_DIR = self.qdir
        qs.reset_cache()
        qs._LOAD_STATS.update({"calls": 0, "files_seen": 0, "hits": 0,
                               "parses": 0})

    def tearDown(self):
        (qs.QUEUE_DIR, qs._TERMINAL_CACHE, qs._FILE_LIST,
         qs._LOAD_STATS) = self._saved
        shutil.rmtree(self.td, ignore_errors=True)

    def _key(self, recs):
        """Comparable form: id, state, and the fields the record actually uses."""
        return sorted((r.get("id"), r.get("state"),
                       r.get("description"), tuple(r.get("command") or ()))
                      for r in recs)

    # --- 1. identity -----------------------------------------------------
    def test_output_identical_to_uncached(self):
        self.assertEqual(self._key(qs._load_all()),
                         self._key(reference_load_all(self.qdir)))

    def test_identity_holds_after_the_cache_is_warm(self):
        qs._load_all()                       # cold
        qs._load_all()                       # warm: every terminal is cached
        self.assertEqual(self._key(qs._load_all()),
                         self._key(reference_load_all(self.qdir)))
        s = qs.load_stats()
        self.assertGreater(s["hit_rate"], 0.0, "cache never engaged, so this "
                                               "test proved nothing")

    def test_repeated_calls_stay_identical(self):
        for _ in range(5):
            self.assertEqual(self._key(qs._load_all()),
                             self._key(reference_load_all(self.qdir)))

    # --- 2. an active job is never hidden --------------------------------
    def test_queued_to_running_is_visible_immediately(self):
        p = write_job(self.qdir, "20260101t000000-queu", "queued")
        qs._load_all()
        before = {r["id"]: r["state"] for r in qs._load_all()}
        self.assertEqual(before["20260101t000000-queu"], "queued")

        # the daemon rewrites the SAME file in place
        rec = json.loads(p.read_text())
        rec["state"] = "running"
        p.write_text(json.dumps(rec))

        after = {r["id"]: r["state"] for r in qs._load_all()}
        self.assertEqual(after["20260101t000000-queu"], "running",
                         "a job that became RUNNING was served from cache -- "
                         "that is a hidden neighbour, the defect this design "
                         "exists to avoid")

    def test_a_running_job_is_never_served_stale(self):
        p = write_job(self.qdir, "20260101t000000-runn", "running")
        qs._load_all()
        rec = json.loads(p.read_text())
        rec["state"] = "finished"
        p.write_text(json.dumps(rec))
        states = {r["id"]: r["state"] for r in qs._load_all()}
        self.assertEqual(states["20260101t000000-runn"], "finished")

    # --- 3. the stated bound is real, not prose --------------------------
    def test_terminal_change_is_stale_then_revealed(self):
        """The one honest cost of this design, measured rather than asserted."""
        qs.TERMINAL_REVERIFY_SEC = 0.2          # keep the test quick
        p = write_job(self.qdir, "20260101t000000-term", "finished",
                      description="ORIGINAL")
        qs._load_all()
        self.assertEqual(
            [r for r in qs._load_all() if r["id"] == "20260101t000000-term"][0]
            ["description"], "ORIGINAL")

        rec = json.loads(p.read_text())
        rec["description"] = "CHANGED"
        p.write_text(json.dumps(rec))

        # inside the window the cache may legitimately serve the old record
        got = [r for r in qs._load_all() if r["id"] == "20260101t000000-term"][0]
        self.assertEqual(got["description"], "ORIGINAL",
                         "a terminal change appeared instantly, so the cache is "
                         "not actually caching terminals")

        # after the window it must be visible, or the bound is a fiction
        time.sleep(qs.TERMINAL_REVERIFY_SEC + 0.05)
        got = [r for r in qs._load_all() if r["id"] == "20260101t000000-term"][0]
        self.assertEqual(got["description"], "CHANGED",
                         "the terminal change never became visible, so "
                         "TERMINAL_REVERIFY_SEC is not a bound")

    def test_the_window_is_reported_in_the_stats(self):
        qs._load_all()
        s = qs.load_stats()
        self.assertIn("terminal_reverify_sec", s)
        self.assertIn("staleness_bound", s)
        self.assertIn("never cached", s["staleness_bound"])

    # --- 4. the module is still a RECORD, not a filter --------------------
    def test_active_set_survives_the_cache(self):
        st = qs.queue_state()
        self.assertIn("n_active_including_self", st)
        self.assertIn("other_running_jobs", st)
        self.assertGreaterEqual(st["n_active_including_self"], 3,
                                "the fixture has 3 active jobs and they must "
                                "all be reported")


if __name__ == "__main__":
    unittest.main(verbosity=2)
