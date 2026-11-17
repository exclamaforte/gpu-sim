"""Guard: gate.sh manifest compares independent declared vs executed sets (gpu-sim-8nv5)."""
from pathlib import Path
import subprocess, tempfile, os

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "gate.sh"


def _run_gate_text(mutated_text):
    """Write mutated gate.sh in place, run --fast, restore, return (rc, output)."""
    backup = GATE.read_text()
    GATE.write_text(mutated_text)
    try:
        r = subprocess.run(["bash", str(GATE), "--fast"],
                           capture_output=True, text=True, timeout=600, cwd=str(REPO))
        return r.returncode, r.stdout + r.stderr
    finally:
        GATE.write_text(backup)


def test_unmutated_gate_passes():
    import signal; signal.alarm(0)
    """Unmutated: gate --fast exits 0 with no MISSING/UNDECLARED."""
    r = subprocess.run(["bash", str(GATE), "--fast"],
                       capture_output=True, text=True, timeout=600, cwd=str(REPO))
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "MISSING" not in out, out
    assert "UNDECLARED" not in out, out


def test_plant_undeclared_stage_fails_manifest():
    import signal; signal.alarm(0)
    """Plant: an undeclared run_hard makes the manifest fail naming it."""
    src = GATE.read_text()
    anchor = "# gpu-sim-gman: manifest comparison (independent declared vs executed sets)."
    injected = 'run_hard "undeclared plant" true\n\n' + anchor
    assert anchor in src
    mutated = src.replace(anchor, injected, 1)
    rc, out = _run_gate_text(mutated)
    assert rc != 0, "expected failure with undeclared stage"
    assert "UNDECLARED" in out and "undeclared plant" in out, out


def test_plant_missing_stage_fails_manifest():
    import signal; signal.alarm(0)
    """Plant: deleting a declared stage's run_hard makes the manifest fail naming it."""
    src = GATE.read_text()
    anchor_line = 'run_hard "trunk refs agree" python3 scripts/check_trunk_refs.py\n'
    assert anchor_line in src
    mutated = src.replace(anchor_line, "", 1)
    rc, out = _run_gate_text(mutated)
    assert rc != 0, "expected failure with missing stage"
    assert "MISSING" in out and "trunk refs agree" in out, out
