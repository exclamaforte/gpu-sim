#!/usr/bin/env python3
"""gpu-sim-u2o.6.4: the read-only half, and the boundaries it must not cross.

Three things are asserted, and the third is the one that keeps this tool safe
on a shared box.

  1. The mutating half stays UNTOUCHED. This bead is held pending a human
     decision because `nvidia-smi -lgc/-pl` and persistence-mode setup change
     GLOBAL GPU state shared with other tenants. No code here may invoke them,
     and a test that greps for the strings is cheaper and stricter than a
     promise in prose.

  2. Throttling and boost-state are NOT conflated. A throttle bitmask is a set
     of CAUSES; clock frequency is a CONSEQUENCE. Item 4 of the bead exists
     because the 2026-09-28 run read "throttle reasons 0 -> 4, clocks spread
     0.098" as if they were one signal.

  3. The tool CANNOT RUN ANYTHING. The first draft had a `--around <command>`
     flag; it was removed because it invites GPU work OUTSIDE the queue, which
     is what the rules forbid and what would contaminate another tenant's
     measurement. A guard that can be argued out of existence by adding a flag
     is not a guard.
"""
import importlib.util
import io
import sys
import tokenize
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def executable_body(path: Path) -> str:
    """The file with comments and docstrings REMOVED.

    My first version of these checks grepped the raw source and failed on the
    module's own prose -- it explains at length that it never runs -lgc and that
    --around was REMOVED, so both strings are present precisely because the
    documentation is honest about them. A check that cannot tell a comment from
    a statement is not a check; this is the third time today that lesson has
    cost a round, after the scope lint on 6.6.6.1 and the ancestry test on
    u2o.15.
    """
    code = []
    with open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                code.append(tok.string)
    return " ".join(code)
_spec = importlib.util.spec_from_file_location(
    "gpu_telemetry", REPO / "experiments" / "gpu-sim-u2o.6.4" / "gpu_telemetry.py")
gt = importlib.util.module_from_spec(_spec)
sys.modules["gpu_telemetry"] = gt
_spec.loader.exec_module(gt)


class TestTheMutatingHalfStaysHeld(unittest.TestCase):
    """Item 1: this is a HOLD, not a preference."""

    MUTATING = ("-lgc", "-lmc", "--lock-gpu-clocks", "-pl",
                "--persistence-mode", "setpersistence", "resetpersistence")

    def test_no_mutating_flag_appears_in_the_code(self):
        src = executable_body(REPO / "experiments" / "gpu-sim-u2o.6.4"
                              / "gpu_telemetry.py")
        for flag in self.MUTATING:
            self.assertNotIn(flag, src,
                             f"{flag!r} appears in the code; the mutating half "
                             f"is HELD pending a human decision")

    def test_the_hold_is_stated_in_the_module_docstring(self):
        src = (REPO / "experiments" / "gpu-sim-u2o.6.4"
               / "gpu_telemetry.py").read_text()
        self.assertIn("HELD PENDING A HUMAN DECISION", src.upper())


class TestThrottlingIsNotBoostState(unittest.TestCase):
    """Item 4: two different signals, kept apart."""

    def test_a_throttle_bit_is_not_a_clock_change(self):
        idle = [{"clocks.sm": 2400.0, "clocks.max.sm": 3105.0,
                 "temperature.gpu": 50.0, "power.draw": 100.0,
                 "clocks_throttle_reasons.active": "0x0"}] * 3
        # IDENTICAL clocks in both arms -- only the throttle bit differs, which
        # is the whole point: same boost state, different cause.
        throttled = idle[:2] + [dict(idle[0],
                                     clocks_throttle_reasons_active=None,
                                     **{"clocks_throttle_reasons.active": "0x4"})]
        a = gt.analyse(idle)
        b = gt.analyse(throttled)
        self.assertEqual(a["throttle_active_samples"], 0)
        self.assertGreater(b["throttle_active_samples"], 0)
        # Same clock, different throttle state: the two are independent signals.
        self.assertAlmostEqual(a["clocks_sm_spread_relative"],
                               b["clocks_sm_spread_relative"], places=6)

    def test_an_active_throttle_bit_changes_the_verdict(self):
        s = gt.analyse([{"clocks.sm": 2400.0, "clocks.max.sm": 3105.0,
                         "clocks_throttle_reasons.active": "0x1"}])
        self.assertTrue(s["throttle_ever_active"])
        self.assertIn("Throttling WAS active", s["THROTTLING_VS_BOOST"]["verdict"])

    def test_an_idle_run_says_so_rather_than_claiming_no_throttle_ever(self):
        s = gt.analyse([{"clocks.sm": 2400.0, "clocks.max.sm": 3105.0,
                         "clocks_throttle_reasons.active": "0x0"}] * 3)
        self.assertFalse(s["throttle_ever_active"])
        self.assertIn("NOT the same signal", s["THROTTLING_VS_BOOST"]["verdict"])

    def test_a_single_sample_yields_no_spread_rather_than_a_fake_zero(self):
        s = gt.analyse([{"clocks.sm": 2400.0, "clocks.max.sm": 3105.0,
                         "clocks_throttle_reasons.active": "0x0"}])
        self.assertIsNone(s["clocks_sm_spread_relative"],
                          "one sample cannot establish a spread")


class TestTheToolRunsNothing(unittest.TestCase):
    """The flag that was removed must stay removed."""

    def test_there_is_no_around_flag(self):
        src = executable_body(REPO / "experiments" / "gpu-sim-u2o.6.4"
                              / "gpu_telemetry.py")
        self.assertNotIn("--around", src)
        self.assertNotIn("Popen", src,
                         "the tool must not be able to launch anything")

    def test_every_record_is_marked_diagnosis_not_measurement(self):
        """Item 3: telemetry is never citable as a measurement."""
        self.assertIn("NOT a measurement", gt.DIAGNOSIS_NOT_MEASUREMENT)
        self.assertIn("never be cited", gt.DIAGNOSIS_NOT_MEASUREMENT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
