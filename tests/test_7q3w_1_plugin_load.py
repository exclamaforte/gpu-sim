"""Guard: 7q3w.1 real libnccl loads behind the interposer (gpu-sim-7q3w.1)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNSH = ROOT / "experiments" / "gpu-sim-7q3w.1" / "run.sh"


def test_plugin_loads():
    r = subprocess.run(["bash", str(RUNSH)], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    # positive control: our plugin's constructor ran
    assert "PLUGIN_LOADED" in out, out
    assert "=== NEGATIVE" in out, out
    pos = out.split("=== NEGATIVE", 1)[0]
    neg = out.split("=== NEGATIVE", 1)[1] if "=== NEGATIVE" in out else ""
    # NCCL's OWN load line in the INFO run (not our prints)
    assert "NCCL INFO" in pos and "Loaded net plugin" in pos, out
    assert "u2o-minimal-net" in pos, out
    # negative control: the WARN run must not contain the load line at all
    assert "Loaded net plugin" not in neg, out
    # run-status: INFO run exits 0; WARN run also exits 0 (Socket fallback) but
    # its load line is absent, which is the measured refusal.
    assert "INFO_EXIT=0" in out, out
    assert "WARN_EXIT=0" in out, out
