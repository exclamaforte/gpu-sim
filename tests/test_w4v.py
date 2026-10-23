import json, os, sys, unittest
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from epic_u2o.op_latency import latency_ms_for, LEGACY_DEFAULT_MS

class TestW4v(unittest.TestCase):
    def test_fp32_unchanged(self):
        # Pins the LEGACY 13.4 table explicitly. gpu-sim-el4 made the measured
        # table the default, so "the default" no longer means the 13.4 numbers;
        # this test's subject is that the legacy fp32 value did not move.
        from epic_u2o.op_latency import load_table
        legacy = load_table(str(REPO / "experiments" / "gpu-sim-u2o.13.4" / "op_latency_table.json"))
        self.assertAlmostEqual(latency_ms_for("relu", table=legacy), 0.0041, places=4)

    def test_fp16_within_25(self):
        os.environ["GPUSIM_OP_LATENCY_JSON"] = str(REPO / "experiments/gpu-sim-84z/mdz_calibration_fixture.json")
        from epic_u2o import op_latency
        op_latency._cache.clear()
        unseen = json.loads((REPO / "experiments/gpu-sim-84z/dtype_fixture.json").read_text())
        errs = []
        for c in unseen["cases"]:
            if c["dtype"] != "float16":
                continue
            n = c["shapes"][0]
            meas = c["device_ms_median"]
            pred = latency_ms_for(c["op"], n, dtype_bytes=2) / 1000.0  # ms? latency_ms_for returns ms? Wait table device_ms_median is ms, normalized ms. latency_ms_for returns ms. Convert to ms? Let's keep ms.
            # Actually latency_ms_for returns ms (table values are ms). meas is ms (device_ms_median). 
            pred_ms = latency_ms_for(c["op"], n, dtype_bytes=2)
            err = abs((pred_ms - meas) / meas)
            errs.append(err)
        self.assertTrue(all(e <= 0.25 for e in errs), errs)
        op_latency._cache.clear()
        del os.environ["GPUSIM_OP_LATENCY_JSON"]

if __name__ == "__main__":
    unittest.main(verbosity=2)
