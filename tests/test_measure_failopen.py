"""Regression: measure.py must not be fail-open on missing binaries (gpu-sim-un4y)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEASURE = ROOT / "experiments" / "gpu-sim-8zqr" / "measure.py"


def _run(build_dir, out, extra=()):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, str(MEASURE), "--build-dir", str(build_dir),
         "--out", str(out), *extra],
        capture_output=True, text=True, env=env, cwd=str(ROOT))


def test_empty_build_dir_exits_nonzero_and_creates_no_file():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    (root / "bin").mkdir()
    (root / "manifest.json").write_text(json.dumps({"specs": [
        {"family": "f", "build_idx": 0, "exe": "missing.out",
         "grid": [1, 1, 1], "block": [1, 1, 1],
         "problem_size": 1024, "metadata": {}, "op_type": "gemm",
         "phys": {}}]}))
    out = root / "out.jsonl"
    r = _run(root, out)
    assert r.returncode != 0, r.stdout + r.stderr
    assert not out.exists(), "empty build must not create an output file"
    assert "SKIPPED 1/1" in r.stdout
    assert "build incomplete" in r.stdout
    td.cleanup()


def test_partial_without_allow_partial_exits_nonzero():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    (root / "bin").mkdir()
    spec = {"family": "f", "build_idx": 0, "exe": "present.out",
            "grid": [1, 1, 1], "block": [1, 1, 1],
            "problem_size": 1024, "metadata": {}, "op_type": "gemm",
            "phys": {}}
    missing = dict(spec); missing["exe"] = "missing.out"; missing["build_idx"] = 1
    (root / "manifest.json").write_text(json.dumps({"specs": [missing, spec]}))
    (root / "bin" / "present.out").write_text(
        "#!/bin/bash\nfor i in $(seq 1 12); do echo '{\"time_ms\": 1.0}'; done\n")
    os.chmod(root / "bin" / "present.out", 0o755)
    out = root / "out.jsonl"
    r = _run(root, out)
    assert r.returncode != 0
    td.cleanup()


def test_partial_with_allow_partial_exits_zero_and_writes_meta():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    (root / "bin").mkdir()
    spec = {"family": "f", "build_idx": 0, "exe": "present.out",
            "grid": [1, 1, 1], "block": [1, 1, 1],
            "problem_size": 1024, "metadata": {}, "op_type": "gemm",
            "phys": {}}
    missing = dict(spec); missing["exe"] = "missing.out"; missing["build_idx"] = 1
    (root / "manifest.json").write_text(json.dumps({"specs": [missing, spec]}))
    (root / "bin" / "present.out").write_text(
        "#!/bin/bash\nfor i in $(seq 1 12); do echo '{\"time_ms\": 1.0}'; done\n")
    os.chmod(root / "bin" / "present.out", 0o755)
    out = root / "out.jsonl"
    r = _run(root, out, ("--allow-partial",))
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    meta = json.loads((Path(str(out) + ".meta.json")).read_text())
    assert meta["skipped"] == 1
    td.cleanup()

def test_fully_populated_build_exits_zero():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    (root / "bin").mkdir()
    spec = {"family": "f", "build_idx": 0, "exe": "present.out",
            "grid": [1, 1, 1], "block": [1, 1, 1],
            "problem_size": 1024, "metadata": {}, "op_type": "gemm",
            "phys": {}}
    (root / "manifest.json").write_text(json.dumps({"specs": [spec]}))
    (root / "bin" / "present.out").write_text(
        "#!/bin/bash\nfor i in $(seq 1 12); do echo '{\"time_ms\": 1.0}'; done\n")
    os.chmod(root / "bin" / "present.out", 0o755)
    out = root / "out.jsonl"
    r = _run(root, out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    assert "DONE measured 1 new rows" in r.stdout
    td.cleanup()



def test_failing_binary_prints_honest_message():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    (root / "bin").mkdir()
    spec = {"family": "f", "build_idx": 0, "exe": "fail.out",
            "grid": [1, 1, 1], "block": [1, 1, 1],
            "problem_size": 1024, "metadata": {}, "op_type": "gemm",
            "phys": {}}
    (root / "manifest.json").write_text(json.dumps({"specs": [spec]}))
    (root / "bin" / "fail.out").write_text("#!/bin/bash\nexit 3\n")
    os.chmod(root / "bin" / "fail.out", 0o755)
    out = root / "out.jsonl"
    r = _run(root, out)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "no rows measured: 0 of 1 specs skipped (exe missing), 1 rows recorded errors" in r.stdout
    assert "build incomplete: no binaries" not in r.stdout
    assert out.exists()
    row = json.loads(out.read_text().splitlines()[0])
    assert row["error"] == "run_fail rc=3"
    assert row["target_usable"] is False
    td.cleanup()


def test_empty_manifest_exits_nonzero_and_creates_no_file():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    (root / "bin").mkdir()
    (root / "manifest.json").write_text(json.dumps({"specs": []}))
    out = root / "out.jsonl"
    r = _run(root, out)
    assert r.returncode != 0, r.stdout + r.stderr
    assert not out.exists(), "empty manifest must not create an output file"
    assert f"manifest lists no specs: {root}/manifest.json" in r.stdout
    td.cleanup()
