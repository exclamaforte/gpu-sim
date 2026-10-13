#!/usr/bin/env python3
"""Tests for 13.2 (CPU): spec vector content + provenance + gate."""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.gpu_spec import FIELDS, SPECS, field_docs, vectorize  # noqa: E402


class TestSpec(unittest.TestCase):
    def test_rtx5090_vector(self):
        v = vectorize("rtx5090")
        self.assertEqual(len(v), 6)
        self.assertEqual(v[0], 170.0)
        self.assertTrue(all(x > 0 for x in v))

    def test_unknown_gpu_raises(self):
        with self.assertRaises(KeyError):
            vectorize("h100")

    def test_fields_documented(self):
        docs = field_docs()
        self.assertEqual(len(docs), len(FIELDS))
        for d in docs:
            self.assertTrue(d["source"])

    def test_second_gpu_gate_closed(self):
        import json
        self.assertEqual(sorted(SPECS), ["rtx5090"])
        metrics = {"bead": "gpu-sim-u2o.13.2",
                   "spec_dim": len(FIELDS),
                   "gpus_with_specs": sorted(SPECS),
                   "gpus_with_measured_rows": ["rtx5090"],
                   "gate_open": False}
        (REPO / "experiments/gpu-sim-u2o.13.2"
         / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
