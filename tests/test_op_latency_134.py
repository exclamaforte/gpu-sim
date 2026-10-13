#!/usr/bin/env python3
"""Tests for 13.4 (CPU): table recomputes from sources (drift guard);
accessor honors table/env/fallback; harness line covered via helper."""
import json
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.op_latency import (  # noqa: E402
    LEGACY_DEFAULT_MS,
    latency_ms_for,
    latency_ms_for_spec,
    load_table,
)


class TestOpLatency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tab = load_table(REPO / "experiments/gpu-sim-u2o.13.4"
                             / "op_latency_table.json")

    def test_table_recomputes(self):
        import collections
        import statistics
        rows = []
        for f in self.tab["_meta"]["sources"]:
            rows += [json.loads(l) for l in (REPO / f).read_text()
                     .splitlines()]
        self.assertEqual(len(rows), self.tab["_meta"]["n_rows"])
        byop = collections.defaultdict(list)
        for r in rows:
            byop[r["op_type"]].append(float(r["median_runtime_ms"]))
        for op, vs in byop.items():
            self.assertAlmostEqual(self.tab["ops"][op]["median_ms"],
                                   statistics.median(vs), places=9)

    def test_known_op_from_table(self):
        self.assertAlmostEqual(latency_ms_for("relu", self.tab), 0.0041,
                               places=4)

    def test_unknown_op_falls_back(self):
        self.assertEqual(latency_ms_for("nope", self.tab),
                         LEGACY_DEFAULT_MS)

    def test_missing_table_falls_back(self):
        os.environ["GPUSIM_OP_LATENCY_JSON"] = "/tmp/134_nope.json"
        try:
            from epic_u2o import op_latency
            op_latency._cache.clear()
            self.assertEqual(latency_ms_for("relu"), LEGACY_DEFAULT_MS)
        finally:
            del os.environ["GPUSIM_OP_LATENCY_JSON"]
            op_latency._cache.clear()

    def test_spec_wiring(self):
        self.assertAlmostEqual(
            latency_ms_for_spec({"op_type": "mul"}, self.tab), 0.0038,
            places=4)
        self.assertEqual(latency_ms_for_spec({}, self.tab),
                         LEGACY_DEFAULT_MS)
        self.assertEqual(latency_ms_for_spec(None, self.tab),
                         LEGACY_DEFAULT_MS)

    def test_env_override(self):
        alt = {"ops": {"relu": {"median_ms": 9.99, "n": 1}}}
        p = Path("/tmp/134_alt.json")
        p.write_text(json.dumps(alt))
        os.environ["GPUSIM_OP_LATENCY_JSON"] = str(p)
        try:
            from epic_u2o import op_latency
            op_latency._cache.clear()
            self.assertAlmostEqual(latency_ms_for("relu"), 9.99)
        finally:
            del os.environ["GPUSIM_OP_LATENCY_JSON"]
            op_latency._cache.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
