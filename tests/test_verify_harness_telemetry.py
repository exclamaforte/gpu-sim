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


if __name__ == "__main__":
    unittest.main(verbosity=2)
