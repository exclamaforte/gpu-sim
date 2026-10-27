#!/usr/bin/env python3
"""scripts/validate_trace.py rejects fail-open compute records (gpu-sim-th56)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "validate_trace.py")


def run_trace(content):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(content)
    proc = subprocess.run([sys.executable, SCRIPT, path],
                          cwd=REPO, capture_output=True, text=True)
    os.unlink(path)
    return proc.returncode, proc.stderr


def rec(**kw):
    base = {"ts_ns": 1, "rank": 0, "nranks": 8, "op": "forward",
            "kind": "compute", "phase": "forward"}
    base.update(kw)
    return json.dumps(base)


class TestValidateTraceRejects(unittest.TestCase):
    def assert_rejects(self, content):
        rc, err = run_trace(content)
        self.assertNotEqual(rc, 0, msg=f"expected reject, got rc={rc}: {err}")

    def assert_accepts(self, content):
        rc, err = run_trace(content)
        self.assertEqual(rc, 0, msg=f"expected accept, got rc={rc}: {err}")

    # rejects
    def test_no_duration(self):
        self.assert_rejects(rec() + "\n")

    def test_empty_op(self):
        self.assert_rejects(rec(op="") + "\n")

    def test_whitespace_op(self):
        self.assert_rejects(rec(op="   ") + "\n")

    def test_inf_seconds(self):
        self.assert_rejects('{"ts_ns":1,"rank":0,"nranks":8,"op":"forward","kind":"compute","phase":"forward","seconds":Infinity}\n')

    def test_nan_seconds(self):
        self.assert_rejects('{"ts_ns":1,"rank":0,"nranks":8,"op":"forward","kind":"compute","phase":"forward","seconds":NaN}\n')

    def test_negative_seconds(self):
        self.assert_rejects(rec(seconds=-1) + "\n")

    def test_none_seconds(self):
        self.assert_rejects('{"ts_ns":1,"rank":0,"nranks":8,"op":"forward","kind":"compute","phase":"forward","seconds":null}\n')

    def test_missing_phase(self):
        r = {"ts_ns": 1, "rank": 0, "nranks": 8, "op": "forward", "kind": "compute", "seconds": 0.1}
        self.assert_rejects(json.dumps(r) + "\n")

    def test_whitespace_phase(self):
        self.assert_rejects(rec(phase="   ") + "\n")

    def test_negative_rank(self):
        self.assert_rejects(rec(rank=-1, seconds=0.1) + "\n")

    def test_bad_kind(self):
        self.assert_rejects(rec(kind="banana", seconds=0.1) + "\n")

    def test_ts_ns_negative(self):
        self.assert_rejects(rec(ts_ns=-5, seconds=0.1) + "\n")

    def test_ts_ns_bool(self):
        self.assert_rejects('{"ts_ns":true,"rank":0,"nranks":8,"op":"forward","kind":"compute","phase":"forward","seconds":0.1}\n')

    def test_seconds_bool(self):
        self.assert_rejects('{"ts_ns":1,"rank":0,"nranks":8,"op":"forward","kind":"compute","phase":"forward","seconds":true}\n')

    def test_collective_bad_op(self):
        r = {"ts_ns": 1, "rank": 0, "nranks": 8, "op": "optimizer", "count": 1,
             "dtype": "bfloat16", "bytes": 4, "algorithm": "ring_allreduce",
             "predicted_s": 0.1, "bottleneck_link": None, "regime": "latency"}
        self.assert_rejects(json.dumps(r) + "\n")

    def test_unrecognized_shape(self):
        self.assert_rejects('{"foo": 1}\n')

    # accepts (zero-duration controls)
    def test_seconds_zero(self):
        self.assert_accepts(rec(seconds=0) + "\n")

    def test_only_predicted_s_zero(self):
        r = {"ts_ns": 1, "rank": 0, "nranks": 8, "op": "forward", "kind": "compute", "phase": "forward", "predicted_s": 0}
        self.assert_accepts(json.dumps(r) + "\n")

    def test_both_zero(self):
        self.assert_accepts(rec(seconds=0, predicted_s=0) + "\n")

    def test_good_collective(self):
        r = {"ts_ns": 1, "rank": 0, "nranks": 8, "op": "allreduce", "count": 1,
             "dtype": "bfloat16", "bytes": 4, "algorithm": "ring_allreduce",
             "predicted_s": 0.1, "bottleneck_link": None, "regime": "latency"}
        self.assert_accepts(json.dumps(r) + "\n")

    # committed traces
    def test_cfg1_trace(self):
        path = os.path.join(REPO, "experiments", "gpu-sim-1ly7", "traces", "cfg1.jsonl")
        proc = subprocess.run([sys.executable, SCRIPT, path], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_cfg2_trace(self):
        path = os.path.join(REPO, "experiments", "gpu-sim-1ly7", "traces", "cfg2.jsonl")
        proc = subprocess.run([sys.executable, SCRIPT, path], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_legacy_trace(self):
        path = os.path.join(REPO, "experiments", "gpu-sim-5o4.7", "raw", "trace.jsonl")
        proc = subprocess.run([sys.executable, SCRIPT, path], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
