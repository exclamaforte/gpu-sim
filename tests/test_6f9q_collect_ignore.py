"""Guard: bare pytest collects cleanly (gpu-sim-6f9q)."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_collect_only_clean():
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.returncode, r.stdout[-500:], r.stderr[-500:])
    assert "experiments/" not in (r.stdout + r.stderr), (r.stdout + r.stderr)
    # no stray test_*.py under experiments/
    stray = list((ROOT / "experiments").rglob("test_*.py"))
    assert not stray, stray
