"""Ensure the suite does not write bytecode under experiments/."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _clean_pycache():
    for d in (ROOT / "experiments").rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
    for f in (ROOT / "experiments").rglob("*.pyc"):
        f.unlink(missing_ok=True)


def test_pycache_baseline_without_suppression_creates_pyc():
    """Negative control: without PYTHONPYCACHEPREFIX, importing a module under
    experiments/ writes a .pyc there (and unfiltered git ls-files --others
    flags it). This is the honest baseline the gate's guard must catch."""
    target = ROOT / "experiments" / "gpu-sim-u2o.17.3"
    assert target.is_dir(), target
    _clean_pycache()

    env = dict(os.environ)
    env.pop("PYTHONPYCACHEPREFIX", None)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    code = ("import sys; sys.path.insert(0, %r); import sim_trace_173; "
            "print('imported')" % str(target))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    stray_pyc = list((ROOT / "experiments").rglob("*.pyc"))
    assert stray_pyc, "baseline: expected a .pyc under experiments/ without suppression"
    # unfiltered git ls-files --others flags it (the honest query)
    ls = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--others", "--", "experiments/"],
                        capture_output=True, text=True)
    assert any(".pyc" in line for line in ls.stdout.splitlines()), \
        "baseline: unfiltered ls-files did not flag the .pyc"

    _clean_pycache()



class TestNoPycache:
    def test_lab_config_sets_pycache_suppression(self):
        """Behavioural: importing a module under experiments/ leaves no .pyc there."""
        target = ROOT / "experiments" / "gpu-sim-u2o.17.3"
        assert target.is_dir(), target
        _clean_pycache()

        # Use a temp prefix so the test does not depend on ambient state
        # (and so PYTHONDONTWRITEBYTECODE from the checker walk does not
        # suppress bytecode writing).  This tests the config plumbing.
        import tempfile
        (ROOT / ".lab" / "tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pycache_", dir=str(ROOT / ".lab" / "tmp")) as td:
            tmp_prefix = Path(td)
            env = dict(os.environ)
            env["PYTHONPYCACHEPREFIX"] = str(tmp_prefix)
            env.pop("PYTHONDONTWRITEBYTECODE", None)
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

            # (b) bytecode landed under the temp prefix
            assert any(tmp_prefix.rglob("*.pyc")), "no bytecode written under prefix"

            # belt-and-braces: lab.config.json carries the env var too (not the fix)
            cfg = json.loads((ROOT / "lab.config.json").read_text())
            cmd = cfg["tests"]["command"]
            assert any("PYTHONPYCACHEPREFIX" in str(x) for x in cmd), cmd

        _clean_pycache()
