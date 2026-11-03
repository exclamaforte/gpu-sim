"""Pin-coverage lint: every experiments/*/run.sh must export the five BLAS/OpenMP pins."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = ["OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"]


class TestPinCoverage(unittest.TestCase):
    def test_all_run_sh_pinned(self):
        run_shs = sorted((ROOT / "experiments").glob("*/run.sh"))
        self.assertGreater(len(run_shs), 0)
        missing = []
        for rs in run_shs:
            text = rs.read_text()
            for pin in PINS:
                if pin not in text:
                    missing.append(f"{rs.relative_to(ROOT)}: missing {pin}")
        self.assertEqual(missing, [], "\n".join(missing))

    def test_plant_fails(self):
        # Plant: remove one pin from a scratch copy and assert the lint catches it.
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp()) / "experiments"
        shutil.copytree(ROOT / "experiments" / "gpu-sim-006j", tmp / "gpu-sim-006j")
        target = tmp / "gpu-sim-006j" / "run.sh"
        text = target.read_text()
        text = text.replace("OMP_NUM_THREADS=1", "OMP_NUM_THREADS=2", 1)
        target.write_text(text)
        # Re-run the check logic against the scratch copy.
        missing = []
        for pin in PINS:
            if pin not in target.read_text():
                missing.append(f"{target}: missing {pin}")
        self.assertTrue(missing, "plant did not trigger a failure")
