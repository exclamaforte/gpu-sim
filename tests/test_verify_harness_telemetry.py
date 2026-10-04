#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.5, criterion 3: the sampler runs with no GPU work.

verify_harness.py records device state either side of a measurement through
read-only nvidia-smi queries. This proves, CPU-only, that the sampler runs
and returns the expected field set -- without running any measurement, queued
or otherwise. nvidia-smi --query-gpu is a driver read; it needs no CUDA
context and touches no other tenant's work.
"""
import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "verify_harness",
    REPO / "experiments" / "gpu-sim-u2o.6.1.1" / "verify_harness.py")
vh = importlib.util.module_from_spec(_spec)
sys.modules["verify_harness"] = vh
_spec.loader.exec_module(vh)


class TestSamplerRuns(unittest.TestCase):
    def test_expected_field_set(self):
        """The fields 6.6.3 needs must actually come back."""
        if shutil.which("nvidia-smi") is None:
            self.skipTest("nvidia-smi not present on this box")
        snap = vh.telemetry("before")
        self.assertEqual(snap["label"], "before")
        self.assertIn("at", snap)
        fields = snap.get("fields", {})
        if not fields:
            # No driver/GPU here: the failure must be RECORDED, not raised.
            self.assertTrue(snap.get("query_rejected") or snap.get("nvidia_smi_error"),
                            "an unsampled snapshot must say why, not go silent")
            return
        for f in ("utilization.gpu", "power.draw", "clocks.sm",
                  "temperature.gpu", "clocks_throttle_reasons.active"):
            self.assertIn(f, fields, f"sampler did not return {f}")

    def test_throttled_predicate(self):
        """A throttle bitmask is causes; the predicate reads it as such."""
        self.assertFalse(vh.throttled({}))
        self.assertFalse(vh.throttled({"fields": {}}))
        self.assertFalse(vh.throttled(
            {"fields": {"clocks_throttle_reasons.active": "0x0000000000000000"}}))
        self.assertTrue(vh.throttled(
            {"fields": {"clocks_throttle_reasons.active": "0x0000000000000004"}}))

    def test_sampling_is_read_only(self):
        """Criterion 6: no -lgc, no persistence change, no sudo in the sampler."""
        src = (REPO / "experiments" / "gpu-sim-u2o.6.1.1"
               / "verify_harness.py").read_text()
        for forbidden in ("-lgc", "-pl ", "persistence-mode -m", "sudo "):
            self.assertNotIn(forbidden, src,
                             f"sampler must stay read-only: {forbidden!r} found")

class TestAgreementIsReal(unittest.TestCase):
    """The duplication account, as executable checks, not prose.

    The README once claimed all three samplers agree; measure_family has 5
    fields, so that claim was false. What IS true: verify_harness and
    measure_noise carry the same 9-field set. Pinned here so a list edit on
either side fires loudly instead of silently forking the sets. The
    measure_family gap (missing power.draw, power.limit, utilization.gpu,
    memory.used) is asserted too -- as a DOCUMENTED gap awaiting the PI's
    schema decision, so closing it later means updating this test on
    purpose, not tripping it by accident.
    """

    def test_telemetry_set_matches_measure_noise(self):
        sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.6.6.1"))
        import measure_noise as mn
        self.assertEqual(set(vh.TELEMETRY_FIELDS), set(mn.TELEMETRY_FIELDS),
                         "verify_harness and measure_noise field sets forked")

    def test_measure_family_gap_is_documented_not_silent(self):
        sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.6.6.2"))
        import measure_family as mf
        missing = (set(vh.TELEMETRY_FIELDS) - set(mf.TELEMETRY_FIELDS))
        self.assertEqual(missing, {"power.draw", "power.limit",
                                   "utilization.gpu", "memory.used"},
                         "the measure_family gap moved: re-check the README "
                         "account and the PI schema decision")


if __name__ == "__main__":
    unittest.main(verbosity=2)
