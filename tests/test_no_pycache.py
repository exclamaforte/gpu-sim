"""Ensure the suite does not write bytecode under experiments/."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lab_config_sets_pycache_suppression():
    """Behavioural: importing a module under experiments/ leaves no .pyc there."""
    target = ROOT / "experiments" / "gpu-sim-u2o.17.3"
    assert target.is_dir(), target
    # declared one-time cleanup of pre-existing pollution (gitignored junk)
    for d in (ROOT / "experiments").rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
    for f in (ROOT / "experiments").rglob("*.pyc"):
        f.unlink(missing_ok=True)

    # The subprocess must INHERIT the suite environment (conftest sets
    # PYTHONPYCACHEPREFIX). Do not override it here -- that is the point.
    env = dict(os.environ)
    code = ("import sys; sys.path.insert(0, %r); import sim_trace_173; "
            "print('imported')" % str(target))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    # (a) no .pyc anywhere under experiments/
    stray_pc = list((ROOT / "experiments").rglob("__pycache__"))
    stray_pyc = list((ROOT / "experiments").rglob("*.pyc"))
    assert not stray_pc, f"__pycache__ created under experiments/: {stray_pc}"
    assert not stray_pyc, f".pyc created under experiments/: {stray_pyc}"

    # (b) bytecode landed under the prefix
    prefix = ROOT / ".lab" / "tmp" / "pycache"
    assert prefix.is_dir(), "pycache prefix dir missing"
    assert any(prefix.rglob("*.pyc")), "no bytecode written under prefix"

    # belt-and-braces: lab.config.json carries the env var too (not the fix)
    cfg = json.loads((ROOT / "lab.config.json").read_text())
    cmd = cfg["tests"]["command"]
    assert any("PYTHONPYCACHEPREFIX" in str(x) for x in cmd), cmd
