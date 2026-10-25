"""gpu-sim-<bug>: a run that issues no collectives must validate, not raise.

The interposer creates its trace lazily on the first traced event, so a world=1
run (no collectives) leaves no trace file. validate_trace used to open() it
unconditionally and raise FileNotFoundError, which made the FIRST documented
command -- `python3 -m tools.torch_parallel.sweep --world-sizes 1` -- report a
FAILED row on an idle GPU while the underlying train.py run succeeded.
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _summary(path, ranks=(0,)):
    with open(path, "w") as fh:
        for r in ranks:
            fh.write(json.dumps({"rank": r, "pp": 1, "mean_step_s": 0.005,
                                 "measure_start_ns": 1}) + "\n")


def test_missing_trace_validates_from_summary():
    with tempfile.TemporaryDirectory() as td:
        summary = os.path.join(td, "summary.jsonl")
        trace = os.path.join(td, "trace.jsonl")   # never created
        _summary(summary)
        assert not os.path.exists(trace)
        proc = subprocess.run(
            [sys.executable, "-m", "tools.torch_parallel.validate_trace",
             "--trace", trace, "--summary", summary],
            cwd=REPO, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "no trace file" in (proc.stderr + proc.stdout)


def test_empty_summary_still_invalid():
    with tempfile.TemporaryDirectory() as td:
        summary = os.path.join(td, "summary.jsonl")
        open(summary, "w").close()               # no ranks at all
        trace = os.path.join(td, "trace.jsonl")
        open(trace, "w").close()
        proc = subprocess.run(
            [sys.executable, "-m", "tools.torch_parallel.validate_trace",
             "--trace", trace, "--summary", summary],
            cwd=REPO, capture_output=True, text=True)
        assert proc.returncode != 0, "an empty summary must not validate"
