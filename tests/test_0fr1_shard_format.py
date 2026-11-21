"""Guard: shard format check runs in --fast (gpu-sim-0fr1)."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_shard_format.py"


def test_shard_format():
    r = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
