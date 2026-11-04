"""Liveness guard for gpu-sim-5uu9 analysis reproducibility."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "gpu-sim-5uu9"
RUN_ANALYZE = HERE / "analyze.py"
METRICS = HERE / "metrics.json"

ENV_KEYS = ("loadavg_1_5_15", "foreign_gpu_job")


def _normalize(obj):
    out = json.loads(json.dumps(obj))
    for k in ENV_KEYS:
        out.pop(k, None)
    if isinstance(out.get("environment"), dict):
        for k in ENV_KEYS:
            out["environment"].pop(k, None)
        if not out["environment"]:
            del out["environment"]
    return out


class Test5uu9(unittest.TestCase):
    def test_analysis_reproducible(self):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        committed = json.loads(METRICS.read_text())
        backup = METRICS.read_bytes()
        scratch = ROOT / ".lab" / "tmp" / "5uu9-test-out"
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            env = dict(os.environ)
            env["U5R4_OUT"] = str(scratch)
            r = subprocess.run([sys.executable, str(RUN_ANALYZE)], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=300, env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            regenerated = json.loads(METRICS.read_text())
        finally:
            METRICS.write_bytes(backup)
        self.assertEqual(_normalize(committed), _normalize(regenerated),
                         "analyze.py output differs from committed metrics.json")


if __name__ == "__main__":
    unittest.main()
