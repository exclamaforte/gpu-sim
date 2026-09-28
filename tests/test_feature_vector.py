#!/usr/bin/env python3
"""Tests for the numeric feature vector. gpu-sim-u2o.8.3.

The bead's audit note said this code "has never been populated from any real
measured example". The first thing found was more basic than that: the module
did not IMPORT. `from tokenizer import ...` instead of
`epic_u2o.tokenizer`, so epic_u2o/features.py could never have been run by
anything, and the note understated how dead the code was.

Every test below corresponds to a way the vector could be populated and still
be worthless, and each was a real behaviour before this branch.

Run:  python3 tests/test_feature_vector.py
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.features import (  # noqa: E402
    LaunchConfig, extract_features, extract_launch_config,
)

SOURCE = """
__global__ void saxpy_kernel(const float *a, float *b, const float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * s + b[i];
}
"""


class TestModuleIsImportable(unittest.TestCase):
    def test_features_imports_as_a_package_module(self):
        """The defect that meant this code had never run at all."""
        r = subprocess.run(
            [sys.executable, "-c", "import epic_u2o.features"],
            cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_no_top_level_absolute_import_of_a_sibling(self):
        src = (REPO / "epic_u2o" / "features.py").read_text()
        for line in src.splitlines():
            if line.startswith(("from ", "import ")):
                self.assertTrue(
                    line.startswith("from epic_u2o.") or
                    line.startswith(("import ", "from __future__")) or
                    line.split()[1].split(".")[0] in
                    ("re", "math", "dataclasses", "typing", "sys", "os"),
                    f"unpackage-relative import: {line!r}")


class TestVectorSeesWhatDeterminesRuntime(unittest.TestCase):
    def test_problem_size_is_in_the_vector(self):
        a = extract_features(SOURCE, LaunchConfig(grid_x=64, block_x=64),
                             problem_size=4096)
        b = extract_features(SOURCE, LaunchConfig(grid_x=64, block_x=64),
                             problem_size=262144)
        self.assertNotEqual(a.to_vector(), b.to_vector(),
                            "the vector cannot distinguish two sizes of the "
                            "same kernel at the same launch")

    def test_launch_config_is_in_the_vector(self):
        a = extract_features(SOURCE, LaunchConfig(grid_x=64, block_x=64),
                             problem_size=4096)
        b = extract_features(SOURCE, LaunchConfig(grid_x=1, block_x=64),
                             problem_size=4096)
        self.assertNotEqual(a.to_vector(), b.to_vector())

    def test_vector_dim_matches_the_vector(self):
        """Was 26 while to_vector() returned 27: the declared dim disagreed
        with the vector it describes, so anything sized from it dropped a
        feature silently."""
        f = extract_features(SOURCE, LaunchConfig(grid_x=8, block_x=64),
                             problem_size=1024)
        self.assertEqual(f.vector_dim, len(f.to_vector()))

    def test_problem_size_defaults_to_zero_and_is_visible(self):
        """Omission must be visible in the emitted vector, not invisible."""
        f = extract_features(SOURCE, LaunchConfig(grid_x=8, block_x=64))
        self.assertEqual(f.problem_size, 0)
        self.assertEqual(f.to_vector()[0], 0.0)


class TestLaunchConfigIsNotSilentlyGuessed(unittest.TestCase):
    def test_kernel_definition_has_no_launch_site(self):
        """A __global__ DEFINITION contains no <<< >>>; the launch is a host
        decision. So this path is taken for every real kernel, and it used to
        return a plausible-looking 1-block default with no signal at all."""
        lc = extract_launch_config(SOURCE)
        self.assertTrue(lc.is_default_heuristic)

    def test_explicit_config_is_not_flagged_default(self):
        self.assertFalse(LaunchConfig(grid_x=64).is_default_heuristic)

    def test_default_and_measured_differ_for_real_sources(self):
        naive = extract_features(SOURCE).to_vector()
        measured = extract_features(
            SOURCE, LaunchConfig(grid_x=256, block_x=256),
            problem_size=1 << 18).to_vector()
        self.assertNotEqual(naive, measured)


class TestPopulatedFromRealRows(unittest.TestCase):
    """The bead's actual ask: populated from real measured examples."""

    def rows(self):
        p = REPO / "experiments" / "gpu-sim-u2o.8.3" / "features.jsonl"
        if not p.exists():
            raise unittest.SkipTest("run populate_feature_vector.py first")
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    def test_every_measured_row_is_populated(self):
        src = subprocess.run(
            ["git", "show",
             "lab/gpu-sim-u2o.7.1:experiments/gpu-sim-u2o.7.1/measured.jsonl"],
            cwd=REPO, capture_output=True, text=True)
        if src.returncode != 0:
            self.skipTest("7.1 dataset not on this ref")
        measured = {json.loads(l)["kernel_hash"]
                    for l in src.stdout.splitlines() if l.strip()}
        self.assertEqual({r["kernel_hash"] for r in self.rows()}, measured)

    def test_populated_vector_separates_far_more_than_the_broken_one(self):
        feats = self.rows()
        distinct = {tuple(f["vector"]) for f in feats}
        self.assertGreater(len(distinct), 100,
                           "the vector still collapses; the launch config and "
                           "problem size are not reaching it")

    def test_every_row_carries_its_measured_label(self):
        for f in self.rows():
            self.assertIn("measured_median_ms", f)
            self.assertGreater(f["measured_median_ms"], 0.0)

    def test_problem_size_in_vector_matches_the_row(self):
        for f in self.rows():
            self.assertEqual(f["problem_size_in_vector"], f["problem_size"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
