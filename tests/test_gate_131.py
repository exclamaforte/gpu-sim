#!/usr/bin/env python3
"""Tests for 13.1 (CPU): the entry gate is machine-checked. While the
transformer loses to GBT on any head-to-head triple (11.4: both), whole-
program composition is gated -- this test documents the block and FAILS
loudly the day the gate opens (forcing the design to be revisited with
real numbers instead of silently going stale)."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class TestEntryGate(unittest.TestCase):
    def test_gate_status(self):
        t = json.loads((REPO / "experiments/gpu-sim-u2o.10.4"
                        / "metrics.json").read_text())
        cells = t["cells"]
        beats = [tr for tr in ("latency->latency", "latency->memory")
                 if cells[tr]["transformer_91"] is not None
                 and cells[tr]["gbt_95"] is not None
                 and cells[tr]["transformer_91"] < cells[tr]["gbt_95"]]
        metrics = {"bead": "gpu-sim-u2o.13.1",
                   "gate": "transformer beats GBT on both head-to-head "
                           "triples + D8 multi-stream measured composition "
                           "data exists",
                   "gate_open": beats == ["latency->latency",
                                          "latency->memory"],
                   "triples_beaten": beats}
        (REPO / "experiments/gpu-sim-u2o.13.1"
         / "metrics.json").write_text(json.dumps(metrics, indent=2))
        # Gated today: assert the blocked state explicitly (a bare pass
        # would hide the reason; a future failure IS the signal).
        self.assertEqual(beats, [], "gate opened: revisit 13.1 design")


if __name__ == "__main__":
    unittest.main(verbosity=2)
