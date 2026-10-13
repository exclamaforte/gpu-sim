#!/usr/bin/env python3
"""gpu-sim-u2o.6.2: the ruled-ladder guard.

The bead's ruling (owner-accepted risk profile): SOFT_RESET is the ONE
automated attempt; HARD_RESET/DRIVER_RELOAD/SYSTEM_REBOOT are MANUAL-ONLY.
`auto_recover()` once escalated through all four -- this fails if that
returns. On SOFT_RESET failure: surface flag, no escalation.

Every test mocks subprocess: even a read-only nvidia-smi is fine, but a
reset pulse must NEVER fire from a CPU test on the desktop GPU.

Run:  CUDA_VISIBLE_DEVICES= python3 tests/test_recovery_ladder.py
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.gpu_recovery import (GPURecovery, GPUHealth, RecoveryAction,
                                   RecoveryResult)


def _rec(tmp, **kw):
    kw.setdefault("checkpoint_dir", str(tmp / "ckpt"))
    return GPURecovery(**kw)


class TestRuledLadder(unittest.TestCase):
    def test_healthy_means_none_no_attempt(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = _rec(Path(td))
            with mock.patch.object(r, "check_gpu_health",
                                   return_value=GPUHealth.HEALTHY), \
                 mock.patch.object(r, "attempt_recovery") as att:
                out = r.auto_recover()
        self.assertTrue(out.success)
        self.assertEqual(out.action, RecoveryAction.NONE)
        att.assert_not_called()

    def test_hung_attempts_soft_reset_only_then_surfaces(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = _rec(Path(td))
            seen = []

            def fake_attempt(action):
                seen.append(action)
                return RecoveryResult(success=False, action=action,
                                      message="nope", duration_sec=0.0)

            with mock.patch.object(r, "check_gpu_health",
                                   return_value=GPUHealth.HUNG), \
                 mock.patch.object(r, "attempt_recovery",
                                   side_effect=fake_attempt):
                out = r.auto_recover()
                self.assertEqual(seen, [RecoveryAction.SOFT_RESET])
                self.assertFalse(out.success)
                self.assertIn("surface", out.message.lower())
                self.assertTrue((Path(td) / "ckpt" / "SURFACE_TO_OWNER.txt").exists())

    def test_no_automatic_hard_reload_or_reboot_in_ladder(self):
        import ast
        src = (REPO / "epic_u2o" / "gpu_recovery.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "auto_recover":
                body = ast.dump(node)
        for banned in ("HARD_RESET", "DRIVER_RELOAD", "SYSTEM_REBOOT"):
            self.assertNotIn(banned, body,
                             f"auto_recover still reaches {banned}")

    def test_soft_reset_declines_on_foreign_holders(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = _rec(Path(td), own_pids={1234})
            with mock.patch.object(r, "compute_app_pids",
                                   return_value=[(1234, "job"), (9999, "gnome")]), \
                 mock.patch("subprocess.run") as sp:
                out = r._soft_reset()
        self.assertFalse(out.success)
        self.assertIn("DECLINED", out.message)
        sp.assert_not_called()  # no reset pulse fired

    def test_soft_reset_fires_only_with_own_holders(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = _rec(Path(td), own_pids={1234})
            ok = mock.Mock(returncode=0, stdout="", stderr="")
            with mock.patch.object(r, "compute_app_pids",
                                   return_value=[(1234, "job")]), \
                 mock.patch.object(r, "check_gpu_health",
                                   return_value=GPUHealth.HEALTHY), \
                 mock.patch("subprocess.run", return_value=ok) as sp:
                out = r._soft_reset()
        self.assertTrue(out.success)
        fired = [c for c in sp.call_args_list if "-r" in str(c)]
        self.assertEqual(len(fired), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
