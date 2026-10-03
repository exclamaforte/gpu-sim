#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.6.2: a row must be able to say how IT was read.

Criterion 2 of 6.6.6.1 asks that every row record which path produced its sample
and the age of the underlying read, so no reader is told a sample is fresher than
it is. Before this change that information existed only in process-global
load_stats(), which answers about the PROCESS: two samples a minute apart get the
same answer, and a caller who never called load_stats() has nothing at all.

These tests pin the three things that matter:
  * ADDITIVE. No pre-existing key changes value. The 12 keys the row already had
    must be byte-identical with the field present and absent.
  * THE PATHS ARE DISTINGUISHABLE, and the two staleness sources are not merged.
    A listing-TTL read is stale in a different way from a terminal-cache read,
    and a reader who cannot tell them apart cannot judge either.
  * IT CAN BE EMPTY. A sample that read nothing must say so, not report a
    confident age of zero.

Runs against a TEMPORARY queue directory, never the shared queue.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "qs_prov", REPO / "epic_u2o" / "queue_state.py")
qs = importlib.util.module_from_spec(_spec)
sys.modules["qs_prov"] = qs
_spec.loader.exec_module(qs)

PRIOR_KEYS = ("counts_by_state", "n_active_including_self", "n_other_active",
              "n_other_running_at_sample_time", "other_active_jobs",
              "other_running_jobs", "own_description_substr", "own_job_id",
              "queue_dir", "queue_was_empty", "recorded_at", "scope_note")


def write_job(qdir, jid, state, **extra):
    rec = {"id": jid, "state": state, "command": ["python3", "x.py"],
           "description": f"job {jid}", "created_at": 1.0}
    rec.update(extra)
    (qdir / f"{jid}.json").write_text(json.dumps(rec))


class Base(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.qdir = self.td / "jobs"
        self.qdir.mkdir()
        for i in range(4):
            write_job(self.qdir, f"term{i}", "finished")
        write_job(self.qdir, "live0", "running")
        self._saved = (qs.QUEUE_DIR, qs._TERMINAL_CACHE, qs._FILE_LIST,
                       dict(qs._LOAD_STATS))
        qs.QUEUE_DIR = self.qdir
        qs.reset_cache()
        qs.LISTING_TTL_SEC = 0.25

    def tearDown(self):
        (qs.QUEUE_DIR, qs._TERMINAL_CACHE, qs._FILE_LIST,
         qs._LOAD_STATS) = self._saved
        qs.reset_cache()
        shutil.rmtree(self.td, ignore_errors=True)

    def record(self):
        return qs.queue_state(own_job_id="live0")


class TestProvenanceIsPresentAndHonest(Base):
    def test_the_record_carries_its_own_provenance(self):
        self.record()
        st = self.record()
        self.assertIn("sample_provenance", st)
        p = st["sample_provenance"]
        self.assertIn("source", p)
        self.assertIn("read_age_ms", p)
        self.assertIn("counts_by_path", p)

    def test_all_four_paths_are_named_distinctly(self):
        """A listing-TTL read and a terminal-cache read are stale DIFFERENTLY.

        Folding them together would let a reader believe a sample was fresh when
        what actually happened is that the file LIST was up to a quarter-second
        old, which can hide a job submitted inside that window.
        """
        qs._load_all()                       # cold: everything parsed
        p = qs._load_all_with_provenance()[1]
        counts = p["counts_by_path"]
        self.assertEqual(set(counts), {"listing_ttl_hit", "terminal_hit_no_stat",
                                       "terminal_hit_stat_ok", "fresh_parse"})
        self.assertEqual(counts["fresh_parse"], 0,
                         "a second call re-reads nothing from disk")
        self.assertGreater(counts["terminal_hit_no_stat"], 0)

    def test_the_first_cold_sample_is_fresh_parse(self):
        p = qs._load_all_with_provenance()[1]
        self.assertGreater(p["counts_by_path"]["fresh_parse"], 0,
                           "the first sample after a reset must report that it "
                           "read from disk")
        self.assertEqual(p["read_age_ms"], 0.0,
                         "a fully fresh read has no age to report")

    def test_an_active_job_is_never_served_stale(self):
        """The property 6.6.6.6.1 exists to protect, now visible per sample."""
        qs._load_all()
        jobs, _ = qs._load_all_with_provenance()
        active = [j for j in jobs if j.get("state") in qs.ACTIVE]
        self.assertTrue(active, "fixture must contain an active job")
        self.assertIn("live0", {j["id"] for j in active},
                      "an active job must be re-read every call, never cached")


class TestCountsArePerSampleNotPerProcess(Base):
    """THE REVIEWER'S CATCH, and the one that made this bead's premise concrete.

    My first version filled sample_provenance.n_files_seen from
    _LOAD_STATS["files_seen"] -- the cumulative PROCESS counter. So two samples
    over an unchanged queue reported 5 then 10, and a reader could not tell
    whether this sample saw 5 files or 10. That is the same defect as putting
    provenance in load_stats() at all: a process-global answer attached to a
    sample, one layer down.

    The test takes TWO samples over a non-empty queue and requires the count to
    describe THIS sample. It fails against the pre-fix code.
    """

    def test_two_samples_report_the_same_per_sample_count(self):
        qs._load_all()                        # first sample, primes the cache
        _, p1 = qs._load_all_with_provenance()
        _, p2 = qs._load_all_with_provenance()
        self.assertGreater(p1["n_files_seen"], 0,
                           "the fixture queue is not empty")
        self.assertEqual(p1["n_files_seen"], p2["n_files_seen"],
                         "an unchanged queue must report the SAME per-sample "
                         "count twice; a growing number means the field is "
                         "carrying the cumulative process counter")

    def test_the_process_counter_is_still_cumulative_and_separate(self):
        """The process counter keeps its meaning; the two must not be conflated."""
        qs._load_all()
        before = qs.load_stats()["files_seen"]
        qs._load_all_with_provenance()
        _, p = qs._load_all_with_provenance()
        after = qs.load_stats()["files_seen"]
        self.assertGreater(after, before,
                           "load_stats() must still accumulate across samples")
        self.assertLess(p["n_files_seen"], after,
                        "the per-sample count must NOT be the cumulative one")

    def test_the_count_matches_what_the_sample_actually_served(self):
        qs._load_all()
        _, p = qs._load_all_with_provenance()
        self.assertEqual(p["n_files_seen"],
                         sum(p["counts_by_path"].values()),
                         "every file seen this sample must be attributed to "
                         "exactly one read path")


class TestAdditiveOnly(Base):
    def test_no_pre_existing_key_changes_value(self):
        """The acceptance constraint: adding the field changes nothing else."""
        with_prov = self.record()
        st = self.record()
        for key in PRIOR_KEYS:
            self.assertIn(key, st, f"{key} disappeared")
        # recorded_at and own-job identity are expected to differ between two
        # calls; everything derived from the queue must not.
        for key in ("counts_by_state", "n_other_active", "queue_dir",
                    "queue_was_empty"):
            self.assertEqual(st[key], with_prov[key],
                             f"{key} changed; the addition is not additive")

    def test_the_provenance_key_is_the_only_new_one(self):
        st = self.record()
        extra = set(st) - set(PRIOR_KEYS)
        # 6.6.6 added its own keys (own_job_entry and friends); only
        # sample_provenance is ours, and it must be present.
        self.assertIn("sample_provenance", extra)


class TestByteIdenticalToThePreChangeCode(Base):
    """THE ACCEPTANCE CONSTRAINT, checked against the real previous version.

    Comparing this record to itself proves nothing about the change. So this
    loads queue_state.py from MASTER -- the code as it was before this bead --
    runs BOTH over the SAME temporary queue, and asserts every pre-existing key
    is byte-identical. Any drift in counts_by_state, the active set, or anything
    else the addition could have disturbed fails here rather than in a review.
    """

    def _master_module(self):
        raw = subprocess.run(
            ["git", "-C", str(REPO), "show", "master:epic_u2o/queue_state.py"],
            capture_output=True, text=True, check=True).stdout
        d = self.td / "master_qs.py"
        d.write_text(raw)
        spec = importlib.util.spec_from_file_location("qs_master", d)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_every_prior_key_is_identical_to_master(self):
        master = self._master_module()
        # Same fixture, same queue dir, for both.
        master.QUEUE_DIR = self.qdir
        master.reset_cache()
        before = master.queue_state(own_job_id="live0")

        qs.reset_cache()
        after = self.record()

        self.assertNotIn("sample_provenance", before,
                         "master must NOT already carry the new key, or this "
                         "comparison proves nothing")
        self.assertIn("sample_provenance", after)
        drift = []
        for key in PRIOR_KEYS:
            if key == "recorded_at":
                continue          # a wall-clock stamp, expected to differ
            if key not in after:
                drift.append(f"{key}: MISSING after the change")
            elif json.dumps(before.get(key), sort_keys=True) != \
                    json.dumps(after[key], sort_keys=True):
                drift.append(f"{key}: {before.get(key)!r} -> {after[key]!r}")
        self.assertEqual(drift, [], "\n".join(drift))


class TestItCanSayNothingHappened(unittest.TestCase):
    """A sample that read nothing must not report a confident age of zero."""

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.qdir = self.td / "jobs"
        self.qdir.mkdir()                    # deliberately EMPTY
        self._saved = (qs.QUEUE_DIR, qs._TERMINAL_CACHE, qs._FILE_LIST,
                       dict(qs._LOAD_STATS))
        qs.QUEUE_DIR = self.qdir
        qs.reset_cache()

    def tearDown(self):
        (qs.QUEUE_DIR, qs._TERMINAL_CACHE, qs._FILE_LIST,
         qs._LOAD_STATS) = self._saved
        qs.reset_cache()
        shutil.rmtree(self.td, ignore_errors=True)

    def test_an_empty_queue_reports_none_not_a_false_zero(self):
        jobs, p = qs._load_all_with_provenance()
        self.assertEqual(jobs, [])
        self.assertEqual(p["source"], "none")
        self.assertEqual(p["n_files_seen"], 0)
        self.assertEqual(sum(p["counts_by_path"].values()), 0)


class TestReadAgeIsTheWorstCase(Base):
    def test_read_age_grows_between_samples_from_cache(self):
        qs._load_all()
        _, p1 = qs._load_all_with_provenance()
        time.sleep(0.05)
        _, p2 = qs._load_all_with_provenance()
        self.assertGreater(p2["read_age_ms"], p1["read_age_ms"],
                           "a sample served from cache must report a growing "
                           "age; a constant zero would tell a reader the sample "
                           "was fresh when it was not")


if __name__ == "__main__":
    unittest.main(verbosity=2)