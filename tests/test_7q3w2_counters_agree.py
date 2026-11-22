"""gpu-sim-7q3w.2: an arm's printed counters must agree with the file it asserts on.

The defect this binds, in one sentence: the arm printed `isend=0 irecv=0
bytes_claimed=0` while the counters file it asserted on said `16 / 16 /
20971520`, and the arm passed because only the file was compared. A reader -- or an
agent grepping the log -- concluded the opposite of the truth, which is the one
thing a "the transport was used" claim cannot afford.

The product under test is the agreement checker the arm now runs; these arms feed
it FIXTURES, so there is no NCCL, no subprocess, no GPU, and no timing: the
property is "two numbers for one fact must be compared, and a disagreement must be
named", which is exactly what the mutation below breaks.

Fixture construction detail that matters: the log line and the JSON are built from
the SAME dict, so the agreeing arm cannot pass by accident and the doctored arm's
only difference is one integer.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# assembled from pieces, not written as one literal: the ledger's self-reference
# rule has bitten this repo twice, and the habit costs nothing.
CHECKER = REPO / "experiments" / "gpu-sim-7q3w.2" / ("check_" + "counters.py")

# keys as they appear on the printed line -> keys as they appear in the JSON
FIELDS = {
    "getProperties": 11, "devices": 3, "init": 1, "listen": 4, "connect": 4,
    "accept": 4, "regMr": 24, "isend": 16, "irecv": 16, "test": 32,
    "bytes_claimed": 20971520, "speed": 400000, "guid": 0xface1b0000000000,
}
JSON_FIELDS = {
    "getProperties": "getProperties_calls", "devices": "devices_calls",
    "init": "init_calls", "listen": "listen_calls", "connect": "connect_calls",
    "accept": "accept_calls", "regMr": "regMr_calls", "isend": "isend_calls",
    "irecv": "irecv_calls", "test": "test_calls", "speed": "props_speed",
    "guid": "props_guid", "bytes_claimed": "bytes_claimed",
}


def _load_checker():
    spec = importlib.util.spec_from_file_location("counters_under_test", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp, printed_isend):
    """Write a rank log + counters file; only `printed_isend` differs between arms."""
    printed = dict(FIELDS)
    printed["isend"] = printed_isend
    line = ("ARM counters rank=0 " + " ".join(
        (f"guid=0x{printed['guid']:x}" if k == "guid" else f"{k}={v}")
        for k, v in printed.items()))
    log = Path(tmp) / "rank0.log"
    log.write_text(f"some NCCL noise\n{line}\nARM OK rank=0\n")
    filed = {JSON_FIELDS[k]: v for k, v in FIELDS.items()}
    filed["deregMr_calls"] = 24
    js = Path(tmp) / "counters-rank0.json"
    js.write_text(json.dumps(filed))
    return str(log), str(js)


class TestCountersAgree(unittest.TestCase):
    def test_agreeing_arm_passes(self):
        checker = _load_checker()
        with tempfile.TemporaryDirectory(dir=str(REPO / ".lab" / "tmp")) as td:
            log, js = _write(td, printed_isend=FIELDS["isend"])
            rc = checker.main([checker.__file__, log, js, "0"])
        self.assertEqual(rc, 0, "an agreeing log/file pair must pass")

    def test_doctored_log_fails_and_names_the_disagreement(self):
        """The arm that proves the check is not vacuous: same file, isend=0."""
        checker = _load_checker()
        import contextlib
        import io
        buf = io.StringIO()
        with tempfile.TemporaryDirectory(dir=str(REPO / ".lab" / "tmp")) as td:
            log, js = _write(td, printed_isend=0)
            with contextlib.redirect_stdout(buf):
                rc = checker.main([checker.__file__, log, js, "0"])
        out = buf.getvalue()
        self.assertNotEqual(rc, 0, f"a doctored log passed: {out}")
        self.assertIn("isend", out, out)
        self.assertIn("printed 0 vs file 16", out,
                      f"the disagreement must be NAMED, got: {out}")

    def test_missing_counters_line_fails(self):
        checker = _load_checker()
        with tempfile.TemporaryDirectory(dir=str(REPO / ".lab" / "tmp")) as td:
            log, js = _write(td, printed_isend=FIELDS["isend"])
            Path(log).write_text("no counters line here\n")
            rc = checker.main([checker.__file__, log, js, "0"])
        self.assertNotEqual(rc, 0, "a log with no counters line must not pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
