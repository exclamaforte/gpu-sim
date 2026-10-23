import json, math, os, sys, unittest
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from epic_u2o.op_latency import latency_ms_for, latency_ms_for_spec, LEGACY_DEFAULT_MS

TABLE_PATH = REPO / "experiments" / "gpu-sim-mdz" / "calibration_fixture.json"

class Test9ei(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["GPUSIM_OP_LATENCY_JSON"] = str(TABLE_PATH)
        from epic_u2o import op_latency
        op_latency._cache.clear()
        cls.tab = op_latency.load_table()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("GPUSIM_OP_LATENCY_JSON", None)
        from epic_u2o import op_latency
        op_latency._cache.clear()

    def test_measured_size_exact(self):
        # a calibration size must return its measured median exactly
        import statistics
        # pick relu 1024? not in mdz; use 2048? mdz has 2048? Actually mdz has 2048? no. Use a known size from table.
        sizes = sorted(self.tab["ops"]["relu"]["sizes"], key=lambda x: int(x))
        sz = int(sizes[0])
        meas = self.tab["ops"]["relu"]["sizes"][str(sz)]
        self.assertEqual(latency_ms_for("relu", sz), meas)

    def test_interpolation_between(self):
        entry = self.tab["ops"]["relu"]
        sizes = sorted(entry["sizes"], key=lambda x: int(x))
        # pick midpoint between first two sizes
        x0, x1 = int(sizes[0]), int(sizes[1])
        x = int((x0 + x1) / 2)
        y0, y1 = entry["sizes"][sizes[0]], entry["sizes"][sizes[1]]
        t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
        expected = math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
        self.assertAlmostEqual(latency_ms_for("relu", x), expected, places=9)

    def test_clamp_low(self):
        entry = self.tab["ops"]["relu"]
        sizes = sorted(entry["sizes"], key=lambda x: int(x))
        first = int(sizes[0])
        self.assertEqual(latency_ms_for("relu", first // 2), entry["sizes"][sizes[0]])

    def test_clamp_high(self):
        entry = self.tab["ops"]["relu"]
        sizes = sorted(entry["sizes"], key=lambda x: int(x))
        last = int(sizes[-1])
        self.assertEqual(latency_ms_for("relu", last * 2), entry["sizes"][sizes[-1]])

    def test_unknown_op_fallback(self):
        self.assertEqual(latency_ms_for("nope", 1024), LEGACY_DEFAULT_MS)

    def test_spec_size_aware(self):
        a = latency_ms_for_spec({"op_type": "relu", "shape": [1024]})
        # 1024 not in table? maybe nearest/clamp; just ensure not legacy
        self.assertNotEqual(a, LEGACY_DEFAULT_MS)

    def test_two_sizes_differ(self):
        a = latency_ms_for("relu", 4096)
        b = latency_ms_for("relu", 1048576)
        self.assertNotEqual(a, b)

    def test_launch_bound_anchor_documented(self):
        # launch_bound relu (64) is merged into relu curve by op-name keying
        entry = self.tab["ops"]["relu"]
        self.assertIn("64", entry["sizes"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
