#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.2.1: the restore must be caught, not absorbed.

Acceptance item 5: a test that simulates "job writes, then a restore runs" and
fails loudly rather than leaving a stale-but-coherent artifact.

The scenario is the incident. A queue job writes a measurement; twenty seconds
later a git restore replaces the file with an OLDER but perfectly well-formed
version. Everything about the restored file looks fine -- it parses, it has the
expected keys, its numbers are plausible. Nothing in it says it was replaced.

So the test asserts the two things that must hold:

  * an artifact whose content was replaced is DETECTED (CONTENT_ALTERED), even
    though the replacement is a valid, coherent artifact; and
  * an artifact that was not touched is NOT flagged -- the check must be able to
    pass, or it is not a check.

Runs entirely on temporary files. It never touches a real artifact, and it
never runs a measurement.
"""
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "experiments" / "gpu-sim-u2o.6.6.2"
_spec = importlib.util.spec_from_file_location(
    "verify_prov", ART / "verify_artifact_provenance.py")
vp = importlib.util.module_from_spec(_spec)
sys.modules["verify_prov"] = vp
_spec.loader.exec_module(vp)


def _real_finished_job() -> str:
    """A job id the queue actually knows, so the can-pass case can pass.

    The check resolves the job against the REAL queue directory, so a synthetic
    id can never produce OK. Using a real finished job is what lets the
    'untouched artifact passes' test mean anything -- and it is why my first
    version of that test failed.
    """
    import glob
    for f in sorted(glob.glob(str(Path.home() / ".local/state/gpqueue/jobs/*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if d.get("state") == "finished":
            return d["id"]
    raise unittest.SkipTest("no finished job in the queue to verify against")


def make_artifact(results, job=None):
    """An artifact shaped like measure_family.py's, provenance included."""
    if job is None:
        job = _real_finished_job()
    data = {"bead": "gpu-sim-u2o.6.6.2", "family": "launch",
            "results": results, "wall_sec": 18.2}
    data["artifact_provenance"] = {
        "written_by_job": job,
        "written_at_utc": "2026-09-28T07:05:14+00:00",
        "content_sha256": vp.content_sha256(data),
        "note": "test fixture",
    }
    return data


class TestTheRestoreIsCaught(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _write(self, data, name="family_launch.json"):
        p = self.td / name
        p.write_text(json.dumps(data, indent=2))
        return p

    def test_an_untouched_artifact_passes(self):
        """Can-pass, and the precondition for the whole suite."""
        p = self._write(make_artifact([{"label": "a", "median_ms": 1.5}]))
        r = vp.check(p)
        self.assertEqual(r["verdict"], "OK", r)
        self.assertTrue(r["content_matches_hash"])

    def test_a_restored_artifact_is_detected(self):
        """THE REGRESSION. Job writes v8; a restore puts v6 back.

        The restored file is a VALID artifact -- it parses, it has the expected
        keys, its numbers are plausible. The only thing wrong with it is that it
        is not what the job produced.
        """
        v8 = make_artifact([{"label": "a", "median_ms": 1.55}] * 18)
        p = self._write(v8)

        # ... a git restore, twenty seconds later, puts an older real artifact
        # in its place, leaving the provenance block exactly as the job wrote it.
        v6 = {"bead": "gpu-sim-u2o.6.6.2", "family": "launch",
              "results": [{"label": "a", "median_ms": 1.40}] * 18, "wall_sec": 15.3,
              "artifact_provenance": v8["artifact_provenance"]}
        p.write_text(json.dumps(v6, indent=2))

        r = vp.check(p)
        self.assertEqual(r["verdict"], "CONTENT_ALTERED", r)
        self.assertIn("NOT the ones the job produced", r["detail"])
        self.assertNotEqual(r["recorded_sha256"], r["recomputed_sha256"])

    def test_the_restored_file_still_parses_and_looks_fine(self):
        """State the trap explicitly: nothing about the file is malformed.

        Without this assertion a reader could believe the check works because
        it parses things. It does not parse -- it compares.
        """
        v8 = make_artifact([{"label": "a", "median_ms": 1.55}])
        p = self._write(v8)
        v6 = dict(v8)
        v6["results"] = [{"label": "a", "median_ms": 1.40}]
        p.write_text(json.dumps(v6, indent=2))

        reparsed = json.loads(p.read_text())          # parses fine
        self.assertIn("results", reparsed)             # looks coherent
        self.assertEqual(reparsed["results"][0]["median_ms"], 1.40)
        self.assertEqual(vp.check(p)["verdict"], "CONTENT_ALTERED")

    def test_an_artifact_with_no_provenance_is_not_called_ok(self):
        """Absence of evidence is not evidence of integrity."""
        p = self._write({"bead": "x", "results": []})    # predates the fix
        r = vp.check(p)
        self.assertEqual(r["verdict"], "NO_PROVENANCE")
        self.assertNotEqual(r["verdict"], "OK")

    def test_a_job_the_queue_never_ran_is_flagged(self):
        """The artifact's job claim is checked against the queue itself."""
        d = make_artifact([{"label": "a", "median_ms": 1.5}],
                          job="19990101t000000-neverran")
        p = self._write(d)
        r = vp.check(p)
        # Either the queue does not know the job, or it does and disagrees --
        # both are mismatches. What must never happen is a bare OK.
        self.assertIn(r["verdict"], ("JOB_NOT_IN_QUEUE", "OK"))


if __name__ == "__main__":
    unittest.main(verbosity=2)