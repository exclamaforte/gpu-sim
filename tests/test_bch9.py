"""gpu-sim-bch9: fabric-band robustness analysis tests."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "experiments" / "gpu-sim-bch9" / "gen_metrics.py"
METRICS = ROOT / "experiments" / "gpu-sim-bch9" / "metrics.json"


class TestBch9:
    def test_metrics_reproducible(self):
        """Re-running gen_metrics.py reproduces the committed metrics.json."""
        import tempfile, os
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        Path(tmp).unlink()
        try:
            subprocess.run(
                [sys.executable, str(GEN), tmp],
                check=True, cwd=str(ROOT), capture_output=True, text=True,
            )
            a = json.loads(METRICS.read_text())
            b = json.loads(Path(tmp).read_text())
            assert a == b, "committed metrics.json differs from freshly computed"
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_elq2_cross_check_zero_mismatches(self):
        """elq2 cross-check must report zero mismatches."""
        import tempfile, os
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        Path(tmp).unlink()
        try:
            subprocess.run(
                [sys.executable, str(GEN), tmp],
                check=True, cwd=str(ROOT), capture_output=True, text=True,
            )
            m = json.loads(Path(tmp).read_text())
            assert m["elq2_cross_check"]["mismatches"] == [], m["elq2_cross_check"]["mismatches"]
            assert m["aggregates"]["n_rows_flipping_5"] == 5
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_n_total_is_43(self):
        m = json.loads(METRICS.read_text())
        assert m["grid"]["n_total"] == 43
        assert m["grid"]["n_log_spaced"] == 41
        assert m["grid"]["n_elq2"] == 5
