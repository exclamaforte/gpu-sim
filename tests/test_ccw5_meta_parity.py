"""Guard: META (run.sh) and ENV_KEYS+META_KEYS (test_5uu9.py) stay equal (gpu-sim-ccw5)."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "gpu-sim-5uu9"
RUN_SH = HERE / "run.sh"
TEST_FILE = ROOT / "tests" / "test_5uu9.py"


def _parse_meta_from_runsh():
    src = RUN_SH.read_text()
    m = re.search(r'^META = \(([^)]*)\)', src, re.MULTILINE)
    assert m, "META tuple not found in run.sh"
    return tuple(re.findall(r'"([^"]*)"', m.group(1)))


def _parse_meta_keys_from_test():
    src = TEST_FILE.read_text()
    env = re.search(r'^ENV_KEYS = \(([^)]*)\)', src, re.MULTILINE)
    meta = re.search(r'^META_KEYS = \(([^)]*)\)', src, re.MULTILINE)
    assert env and meta, "ENV_KEYS/META_KEYS not found in test_5uu9.py"
    env_keys = tuple(re.findall(r'"([^"]*)"', env.group(1)))
    meta_keys = tuple(re.findall(r'"([^"]*)"', meta.group(1)))
    return env_keys + meta_keys


def test_meta_lists_equal():
    run_meta = _parse_meta_from_runsh()
    test_meta = _parse_meta_keys_from_test()
    assert sorted(run_meta) == sorted(test_meta), (
        f"META mismatch: run.sh={run_meta} vs test={test_meta}")


def test_runsh_detects_real_metric_change():
    """Driving run.sh against a metrics.json with a real value changed must fail."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        exp_copy = tmp / "gpu-sim-5uu9"
        shutil.copytree(str(HERE), str(exp_copy))
        metrics_path = exp_copy / "metrics.json"
        d = json.loads(metrics_path.read_text())
        # Bump a real metric value.
        def bump(obj):
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if k == "gbt_mae" and isinstance(v, (int, float)):
                        obj[k] = v + 0.001
                    else:
                        bump(v)
            elif isinstance(obj, list):
                for item in obj:
                    bump(item)
        bump(d)
        metrics_path.write_text(json.dumps(d, indent=2))
        # Point run.sh ROOT at the temp copy by running from a modified copy.
        runsh_copy = exp_copy / "run.sh"
        runsh_src = runsh_copy.read_text()
        # Replace ROOT resolution to use the temp experiment's parent.
        runsh_src = runsh_src.replace(
            'ROOT="$(cd "$HERE/../.." && pwd)"',
            f'ROOT="{tmp}"')
        runsh_copy.write_text(runsh_src)
        env = dict(os.environ)
        env["U5R4_OUT"] = str(tmp / ".lab" / "tmp" / "5uu9_out")
        r = subprocess.run(["bash", str(runsh_copy)], capture_output=True,
                           text=True, timeout=300, env=env, cwd=str(tmp))
        assert r.returncode != 0, (
            f"run.sh should fail on metric change but exited 0: {r.stdout[-200:]}")
