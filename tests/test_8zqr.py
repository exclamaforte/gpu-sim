"""gpu-sim-8zqr: generalization test reproducibility."""
import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "gpu-sim-8zqr" / "run.sh"
STAGE = ROOT / "experiments" / "gpu-sim-8zqr" / "stageD_results.json"


class Test8zqr:
    def test_stageD_reproducible(self):
        """Re-running run.sh succeeds and stageD_results.json is present."""
        r = subprocess.run(["bash", str(RUN)], cwd=str(ROOT), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert STAGE.exists()
        json.loads(STAGE.read_text())
