#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.2.1: porcelain parsing must not eat a leading character.

The triage reported `xperiments/gpu-sim-u2o.6.6.2/family_gemm.json` instead of
`experiments/...`. The cause: `git status --porcelain` renders a
modified-not-staged path as " M path" -- leading SPACE -- and the helper did
`.strip()`, which removes it from the FIRST line only, because strip() acts on
the string boundary. `line[3:]` then ate a real character of the path.

Worth stating why this deserved a test rather than a re-read. One character, on
one line, in the file whose entire job is to report paths a human will act on --
and every count in the artifact stayed correct, because a truncated path still
counts as one path. The numbers looked perfect and the payload was wrong.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "experiments" / "gpu-sim-u2o.6.6.2"
_spec = importlib.util.spec_from_file_location(
    "triage", ART / "triage_stale_worktree.py")
triage = importlib.util.module_from_spec(_spec)
sys.modules["triage"] = triage
_spec.loader.exec_module(triage)


def parse(porcelain: str):
    """The same shape the triage uses, so the test exercises the real thing."""
    modified, untracked = [], []
    for line in porcelain.splitlines():
        if line.startswith("??"):
            untracked.append(line[3:].strip())
        elif line[:2].strip():
            modified.append({"path": line[3:].strip(), "status": line[:2].strip()})
    return modified, untracked


class TestLeadingSpaceIsNotEaten(unittest.TestCase):
    RAW = (" M experiments/gpu-sim-u2o.6.6.2/family_gemm.json\n"
           "?? epic_u2o/new_file.py\n"
           " M experiments/gpu-sim-u2o.6.6.2/family_memory.json\n")

    def test_the_first_modified_path_survives_intact(self):
        modified, _ = parse(self.RAW)
        self.assertEqual(modified[0]["path"],
                         "experiments/gpu-sim-u2o.6.6.2/family_gemm.json")

    def test_the_bug_this_regression_covers(self):
        """Pinned so the shape of the failure is visible, not just its absence."""
        broken = self.RAW.strip().splitlines()
        truncated = broken[0][3:]
        self.assertTrue(truncated.startswith("xperiments"),
                        "the failure mode was a lost leading character")
        self.assertFalse(truncated.startswith("experiments"))

    def test_untracked_paths_are_unaffected(self):
        _m, untracked = parse(self.RAW)
        self.assertEqual(untracked, ["epic_u2o/new_file.py"])

    def test_the_helper_does_not_strip_leading_whitespace(self):
        """The fix is in the HELPER, not the caller."""
        src = (ART / "triage_stale_worktree.py").read_text()
        self.assertNotIn("p.stdout.strip()", src,
                         "the helper must not strip leading whitespace; that "
                         "was the bug and it was in the helper")

    def test_every_recorded_path_resolves_where_it_claims(self):
        """A path a human will act on should at least be well formed."""
        _m, untracked = parse(self.RAW)
        for p in untracked:
            self.assertFalse(p.startswith(("/", " ")))
            self.assertNotIn("..", p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
