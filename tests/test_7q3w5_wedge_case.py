"""Bind gpu-sim-7q3w.5's wedge predicate to the synthetic shapes it claims.

The p2p control's negative result is a statement about the SOFTWARE, not about
an exit code: 'the sender never saw the peer's connection record, so it never
posted the data WR and the receiver got nothing'.  capture.py::wedge_case is
where that claim lives, so it is tested here against four synthetic logs:

  C-shaped      no progress thread          -> True
  D-shaped      FAKEIB_LAZY_PUMP (pre-fix)  -> True   (same signature as C)
  TIMEOUT-ONLY  rc != 0, ranks at 124, it POLLED, but there are no ring
                counter lines                -> False  (a bare timeout is not
                                                       evidence of a wedge)
  B-shaped      the passing p2p arm          -> False

The TIMEOUT-ONLY case is the one that matters: the mutation registered as
7q3w5_wedge_case reduces the predicate to bool(ev["wedged"]), which would make
that case True, so this test -- not the mutation harness -- is what catches a
timeout being passed off as a measured wedge.
"""

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "experiments" / "gpu-sim-7q3w.5" / "capture.py"


def load_capture():
    spec = importlib.util.spec_from_file_location("capture_7q3w5", CAPTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(**overrides):
    """A C-shaped wedge: rank1 done and waiting, rank0 blind to the wire."""
    ev = {
        "rc": 1,
        "rank_rc": {"0": "124", "1": "124"},
        "rank1_posts_conn": True,
        "rank0_sees_inbound": False,
        "rank0_posts_payload": False,
        "rank1_gets_payload": False,
        "rank1_p2p_line": False,
        "wedged": True,
        "wedge_last_rings": (1, 0),
    }
    ev.update(overrides)
    return ev


class TestWedgeCase(unittest.TestCase):
    def setUp(self):
        self.wedge_case = load_capture().wedge_case

    def test_C_shaped_log_is_a_wedge(self):
        self.assertTrue(self.wedge_case(evidence()))

    def test_D_shaped_log_is_a_wedge(self):
        # D is the same arm with the pre-fix LAZY start: the signature is the
        # same 64-byte-only ring state, so the predicate must accept it.
        self.assertTrue(self.wedge_case(evidence()))

    def test_timeout_only_log_is_not_a_wedge(self):
        # Killed at the rank timeout after polling, but no ring counters: this
        # is an absence of evidence, not the measured signature.  The mutation
        # (bool(ev['wedged'])) turns exactly this case True.
        self.assertFalse(self.wedge_case(evidence(wedge_last_rings=None)))

    def test_b_shaped_log_is_not_a_wedge(self):
        ev = evidence(rc=0, rank_rc={"0": "0", "1": "0"},
                      rank0_sees_inbound=True, rank0_posts_payload=True,
                      rank1_gets_payload=True, rank1_p2p_line=True,
                      wedged=False, wedge_last_rings=None)
        self.assertFalse(self.wedge_case(ev))

    def test_nonzero_exit_alone_is_not_a_wedge(self):
        ev = evidence(wedged=False, wedge_last_rings=None)
        self.assertFalse(self.wedge_case(ev))

    def test_consumed_nothing_is_required(self):
        # rank1 must have consumed zero records (in=0); out=1 is its own
        # connection info.  A ring state of (1, 1) is NOT this wedge.
        self.assertFalse(self.wedge_case(evidence(wedge_last_rings=(1, 1))))


if __name__ == "__main__":
    unittest.main()
