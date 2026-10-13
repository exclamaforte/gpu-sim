#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.6.3.1: the retry must be bounded for permanently-bad files.

The retry loop (3x50ms sleeps) exists because a partial file is a normal
transient while the daemon writes. For a PERMANENTLY malformed file -- valid
JSON followed by appended garbage, a gpqueue write bug -- the retry can never
succeed, and the bead measures 150ms of sleep per queue_state() call against
a 0.139ms device budget, breaking the differential guard on both arms.

The fix remembers files that exhaust all retries (keyed by path + mtime_ns +
size) and gives them a single parse with no sleeps on later calls, until a
rewrite heals the entry. Three properties, all CPU-only on fixture queues
(the live queue is never touched):

  1. the transient tolerance is intact: an unknown bad file still gets all 3
     retries (proven by counting sleeps, not wall time -- wall time flakes
     on a shared box, while a sleep count is deterministic);
  2. the permanent case is bounded: the second call sleeps zero times, the
     file stays absent from the records, and the skip is counted in
     provenance rather than silent;
  3. a rewrite heals: fixing the file (mtime moves) parses it again and
     forgets the entry.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o import queue_state as qs  # noqa: E402

# The bead's file, reduced: valid JSON followed by appended garbage.
GARBAGE_TAIL = '}\n069\n}\n'


def write_job(qdir: Path, jid: str, state: str) -> Path:
    p = qdir / f"{jid}.json"
    p.write_text(json.dumps({"id": jid, "state": state,
                             "command": ["python3", "x.py"],
                             "description": f"test job {jid}"}))
    return p


def write_malformed(qdir: Path, jid: str) -> Path:
    p = qdir / f"{jid}.json"
    p.write_text(json.dumps({"id": jid, "state": "finished"}) + "\n069\n}\n")
    return p


class Fixture(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.qdir = self.td / "jobs"
        self.qdir.mkdir()
        write_job(self.qdir, "20260101t000000-good", "finished")
        self.bad = write_malformed(self.qdir, "20260101t000000-bad")
        self._saved = (qs.QUEUE_DIR, qs._TERMINAL_CACHE, qs._FILE_LIST,
                       dict(qs._LOAD_STATS), dict(qs._UNPARSEABLE))
        qs.QUEUE_DIR = self.qdir
        qs.reset_cache()

    def tearDown(self):
        (qs.QUEUE_DIR, qs._TERMINAL_CACHE, qs._FILE_LIST,
         _stats, _unp) = self._saved
        qs._LOAD_STATS.clear()
        qs._LOAD_STATS.update(_stats)
        qs._UNPARSEABLE.clear()
        qs._UNPARSEABLE.update(_unp)
        shutil.rmtree(self.td, ignore_errors=True)

    def load(self):
        return qs._load_all_with_provenance()


class TestTransientToleranceIntact(Fixture):
    def test_unknown_bad_file_still_gets_all_retries(self):
        """The fix must not remove the tolerance it bounds."""
        with mock.patch("time.sleep") as slp:
            recs, prov = self.load()
        self.assertEqual(slp.call_count, 3,
                         "an unseen malformed file must still be retried 3x")
        self.assertEqual([r["id"] for r in recs], ["20260101t000000-good"])
        self.assertEqual(prov["unparseable_skipped_retry"], 0)


class TestPermanentCaseBounded(Fixture):
    def test_second_call_sleeps_zero_times(self):
        self.load()  # first call: retries, remembers
        with mock.patch("time.sleep") as slp:
            recs, prov = self.load()
        self.assertEqual(slp.call_count, 0,
                         "a remembered malformed file costs no sleeps")
        self.assertEqual([r["id"] for r in recs], ["20260101t000000-good"],
                         "the bad file stays out of the records")
        self.assertEqual(prov["unparseable_skipped_retry"], 1,
                         "the skip is counted in provenance, never silent")
        self.assertEqual(qs.load_stats()["unparseable_remembered"], 1)

    def test_rewrite_heals_the_entry(self):
        self.load()
        self.assertIn(str(self.bad), qs._UNPARSEABLE)
        write_job(self.qdir, "20260101t000000-bad", "finished")
        recs, _prov = self.load()
        self.assertIn("20260101t000000-bad", [r["id"] for r in recs],
                      "a rewritten file parses again")
        self.assertNotIn(str(self.bad), qs._UNPARSEABLE,
                         "the healed entry is forgotten")


if __name__ == "__main__":
    unittest.main(verbosity=2)
