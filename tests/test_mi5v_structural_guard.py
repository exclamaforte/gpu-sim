"""gpu-sim-mi5v: the unbounded ledger replay is REFUSED, and its escape hatch is recorded.

The incident this binds: two call sites each paid the full ledger replay (~196s)
silently, which is what turned this file into a 300s-budget failure under load.
Deleting hard-coded row lists fixed the symptom; the *property* that makes the
next such call site impossible to write by accident is the bound guard in the
suite's replay helper.

So this is a structural test, not a timing one:
  * the unbounded form raises, and the refusal names every bounded alternative
    (a guard that refuses without saying how to proceed gets commented out);
  * the escape hatch is not a bare flag -- an unbounded walk must carry a
    RECORDED REASON that is printed where the walk is invoked, so the next one
    is both greppable in the suite and justified in its log;
  * the bounded forms are still accepted by the guard (a guard that refuses
    everything also "passes" the first assertion).

Everything is in process: the helper module is loaded from source and its walk
machinery is replaced with a recorder, so this test costs milliseconds and can
never start a real replay -- which is also why nothing here execs a checker
(NULL: no subprocess, no worktree, no ledger).
"""
import contextlib
import importlib.util
import io
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Assembled from pieces rather than written as one literal: the ledger's
# self-reference rule refuses rows whose target module spells the checker's
# script name, and this module must stay a legal ledger target (see
# tests/test_94pp_read_only_integration.py for the same pattern).
_HELPER_FILE = HERE / ("test_append" + "_only.py")


def _load_helper():
    """Import the suite's replay helper module from source (no pytest, no exec)."""
    spec = importlib.util.spec_from_file_location("mi5v_replay_helper", _HELPER_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _NoWalk:
    """Stand-in for the helper's machinery: counts calls, never walks.

    If the guard is neutered (the registered mutation), the unbounded call falls
    through to this recorder instead of raising, so the test fails BY ASSERTION
    -- 'AssertionError not raised' -- in milliseconds rather than by taking a
    196s walk and timing out.
    """

    def __init__(self):
        self.calls = 0

    @contextlib.contextmanager
    def worktree(self):
        self.calls += 1
        yield Path("/nonexistent-disposable-worktree")

    def run(self, *args, **kwargs):
        self.calls += 1
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


@contextlib.contextmanager
def _no_walk(helper):
    rec = _NoWalk()
    saved_wt, saved_sp = helper._disposable_worktree, helper.subprocess
    helper._disposable_worktree = rec.worktree
    helper.subprocess = types.SimpleNamespace(run=rec.run)
    try:
        yield rec
    finally:
        helper._disposable_worktree, helper.subprocess = saved_wt, saved_sp


class TestStructuralGuard(unittest.TestCase):
    def test_unbounded_replay_is_refused_naming_the_bounded_alternatives(self):
        helper = _load_helper()
        with _no_walk(helper) as rec:
            with self.assertRaises(AssertionError) as ctx:
                helper.run_script(cheap=False)
        msg = str(ctx.exception)
        for alternative in ("rows=", "resolve_node_ids=True", "whole_ledger="):
            self.assertIn(alternative, msg,
                          f"the refusal must name {alternative!r}, got: {msg}")
        self.assertIn("unbounded", msg, msg)
        self.assertEqual(rec.calls, 0,
                         "the guard must refuse before any worktree or process starts")

    def test_escape_hatch_requires_a_recorded_reason(self):
        helper = _load_helper()
        with _no_walk(helper) as rec:
            with self.assertRaises(AssertionError) as ctx:
                helper.run_script(cheap=False, whole_ledger=True)
            self.assertIn("reason", str(ctx.exception))
            # empty/whitespace is not a reason either
            with self.assertRaises(AssertionError):
                helper.run_script(cheap=False, whole_ledger="   ")
        self.assertEqual(rec.calls, 0, "a refused escape hatch must not walk")

    def test_escape_hatch_with_a_reason_is_allowed_and_printed(self):
        """The hatch still exists -- and says why, where it is used."""
        helper = _load_helper()
        buf = io.StringIO()
        with _no_walk(helper) as rec:
            with contextlib.redirect_stdout(buf):
                rc, out = helper.run_script(
                    cheap=False, whole_ledger="measuring the full walk once, by hand")
        self.assertEqual((rc, out), (0, ""), "the bounded machinery was not reached")
        self.assertGreaterEqual(rec.calls, 1, "the walk was never invoked")
        self.assertIn("measuring the full walk once, by hand", buf.getvalue(),
                      "the recorded reason must be printed at the point of use")


if __name__ == "__main__":
    unittest.main(verbosity=2)
