#!/usr/bin/env python3
"""gpu-sim-u2o.6.3: the static VRAM cap must be IN preflight_check, and stay there.

THE INCIDENT THIS EXISTS FOR. 6.3's first commit (2dd3074) put the static VRAM
budget check inside `preflight_check`. Its own subsequent rebase DELETED the hunk,
and the cherry-pick onto master reproduced the loss rather than preventing it.
Nothing failed: the limit table still said "VRAM cap", the docs still described
it, and the suite stayed green -- while `preflight_check` no longer consulted
the budget at all. A pathological problem_size would have gone straight at the
shared GPU.

That is the same shape as the guard gpu-sim-u2o.6.3.1 was filed for: a table
describing code that no longer behaves that way, with nothing comparing the two.

So this is a structural guard. It reads the SOURCE of `preflight_check` and
fails if the budget is not consulted in that function. Source rather than
behaviour on purpose: the failure mode is DELETION by a rebase, and a behavioural
test that constructs a harness can be skipped or made to pass for unrelated
reasons, whereas the hunk simply either is in the function or is not.
"""
import ast
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "epic_u2o" / "safe_harness.py"


def preflight_source() -> str:
    tree = ast.parse(HARNESS.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "preflight_check":
            return ast.get_source_segment(HARNESS.read_text(), node) or ""
    raise AssertionError("preflight_check not found")


class TestTheVramHunkIsPresent(unittest.TestCase):
    def test_preflight_check_consults_the_vram_budget(self):
        src = preflight_source()
        self.assertIn("check_vram_budget", src,
                      "preflight_check no longer consults the static VRAM "
                      "budget. 6.3's first commit had this hunk and its own "
                      "rebase deleted it; the limit table and the docs still "
                      "describe a check the code does not perform.")

    def test_it_records_the_verdict_for_the_record(self):
        self.assertIn("last_vram_check", preflight_source(),
                      "the budget verdict is not recorded, so a later reader "
                      "cannot tell whether the check ran")

    def test_the_limit_table_and_the_code_agree(self):
        """The guard gpu-sim-u2o.6.3.1 could not make, because it lives on a
        different branch: that the TABLE claims a VRAM cap and the CODE has one.

        Merging a correct table beside a preflight that lost its budget step is
        exactly what the reviewer blocked 6.3.1 over. Asserted here so the two
        cannot drift apart silently on either branch.
        """
        sys.path.insert(0, str(REPO))
        try:
            from epic_u2o.rlimit_cuda_gate import LIMIT_TABLE
        except ModuleNotFoundError:
            self.skipTest(
                "rlimit_cuda_gate is not on this branch. It lives on "
                "gpu-sim-u2o.6.3.1, which is BLOCKED on this hunk. Once 6.3 "
                "merges, this test runs on 6.3.1 and is the one that keeps the "
                "table and the code from drifting apart again.")
        vram_entries = [e for e in LIMIT_TABLE
                        if "VRAM" in e.get("limit", "") or "AS" in e.get("limit", "")]
        if not vram_entries:
            self.skipTest("no VRAM/AS entry in the limit table")
        self.assertTrue(any(e.get("applied", True) for e in vram_entries),
                        "the table claims no VRAM limit is applied")
        self.assertIn("check_vram_budget", preflight_source(),
                      "the table claims a VRAM limit is applied but "
                      "preflight_check does not apply one")


if __name__ == "__main__":
    unittest.main(verbosity=2)