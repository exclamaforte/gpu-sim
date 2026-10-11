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
                    # `from epic_u2o import hw_5090`: package-relative, the
                    # g8x single-sourcing (the defect was a BARE module).
                    line.startswith("from epic_u2o import ") or
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

    def measured_hashes(self):
        out = set()
        for c in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
                  "experiments/gpu-sim-e5q/measured.jsonl"):
            p = REPO / c
            if not p.exists():
                raise unittest.SkipTest(f"corpus missing: {c}")
            out |= {json.loads(l)["kernel_hash"]
                    for l in p.read_text().splitlines() if l.strip()}
        return out

    def test_every_measured_row_is_populated(self):
        self.assertEqual({r["kernel_hash"] for r in self.rows()},
                         self.measured_hashes())
        self.assertEqual(len(self.rows()), 456)

    def test_no_row_ships_zero_bytes(self):
        """Item 4, spirit: a 456/456 run with bytes_moved == 0 proves nothing."""
        zeroed = [r["kernel_hash"] for r in self.rows()
                  if r["bytes_moved"] == 0]
        self.assertEqual(zeroed, [])

    def test_no_two_runtimes_share_a_vector(self):
        """Item 5, unit mirror of the populate gate (full gate is the driver)."""
        groups = {}
        for r in self.rows():
            groups.setdefault(tuple(r["vector"]), set()).add(
                r["measured_median_ms"])
        bad = {v: t for v, t in groups.items() if len(t) > 1}
        self.assertEqual(bad, {})

    def test_every_row_carries_its_measured_label(self):
        for f in self.rows():
            self.assertIn("measured_median_ms", f)
            self.assertGreater(f["measured_median_ms"], 0.0)

    def test_problem_size_in_vector_matches_the_row(self):
        for f in self.rows():
            self.assertEqual(f["vector"][0], float(f["problem_size"]))

    def test_scale_hand_computation_on_the_real_corpus(self):
        """Item 9 on measured rows: full-cover scale at n=2^22 == 3*4*n."""
        hand = [f for f in self.rows() if f["op_type"] == "scale"
                and f["problem_size"] == 1 << 22 and f["covers"] == 1 << 22]
        self.assertTrue(hand, "no full-cover scale row at n=2^22")
        h = hand[0]
        n = 1 << 22
        self.assertEqual(h["reads"], 2 * n)
        self.assertEqual(h["writes"], 1 * n)
        self.assertEqual(h["bytes_moved"], 3 * 4 * n)


class TestStagesTwoThroughFive(unittest.TestCase):
    """Item 10: the read/write computation is not a stage-1 special case.

    One generated kernel per stage (reductions, tiled/GMMA, divergence,
    atomics); each must yield NONZERO plausible global traffic through the
    SAME code path as the corpus -- no per-stage branches exist to test
    beyond this, which is the point."""

    def feat(self, source):
        return extract_features(
            source, LaunchConfig(grid_x=64, block_x=256),
            problem_size=1 << 20)

    def one_of(self, module, kernel_name):
        for k in module.generate_kernels(max_kernels=300):
            if k.kernel_name == kernel_name:
                return k.source
        raise unittest.SkipTest(f"{kernel_name} not generated")

    def test_reduction_reads_and_writes(self):
        from epic_u2o import generator_stage2 as s2
        f = self.feat(self.one_of(s2, "reduce_sum_block"))
        self.assertGreater(f.estimated_global_reads, 0)
        self.assertGreater(f.estimated_global_writes, 0)
        self.assertGreater(f.estimated_shared_reads, 0)
        self.assertTrue(f.uses_shared_memory)
        self.assertTrue(f.has_loops)

    def test_tiled_reads_and_writes(self):
        from epic_u2o import generator_stage3 as s3
        names = {k.kernel_name for k in s3.generate_kernels(max_kernels=300)}
        target = sorted(names)[0]
        f = self.feat(self.one_of(s3, target))
        self.assertGreater(f.estimated_global_reads, 0)
        self.assertGreater(f.estimated_global_writes, 0)
        self.assertGreater(f.estimated_bytes_moved, 0)

    def test_divergence_reads_and_writes(self):
        from epic_u2o import generator_stage4 as s4
        f = self.feat(self.one_of(s4, "gemm_naive"))
        self.assertGreater(f.estimated_global_reads, 0)
        self.assertGreater(f.estimated_global_writes, 0)
        self.assertTrue(f.has_loops)

    def test_atomic_reads_and_writes(self):
        from epic_u2o import generator_stage5 as s5
        f = self.feat(self.one_of(s5, "histogram_global"))
        self.assertGreater(f.estimated_global_reads, 0)
        self.assertGreater(f.estimated_global_writes, 0)
        self.assertGreater(f.num_call_sites, 0)


class TestAccessCounting(unittest.TestCase):
    """D5: the counters that shipped as permanent zeros."""

    def feat(self, src, n=4096, grid=64, block=64):
        return extract_features(
            src, LaunchConfig(grid_x=grid, block_x=block), problem_size=n)

    def test_saxpy_hand_computation(self):
        """Item 9, unit form: x[i] + y[i] + s = 3 reads, b[i] = 1 write."""
        saxpy = self.feat(SOURCE, n=4096)
        self.assertEqual(saxpy.covers, 4096)
        self.assertEqual(saxpy.estimated_global_reads, 3 * 4096)
        self.assertEqual(saxpy.estimated_global_writes, 1 * 4096)
        self.assertEqual(saxpy.estimated_bytes_moved, 4 * 4 * 4096)

    def test_operator_mix_separates_add_mul_div(self):
        # Every kernel carries +1 add / +1 mul from the index expression
        # (blockIdx.x * blockDim.x + threadIdx.x); the SPLIT still separates
        # the compute op, which is what differs across add/mul/div.
        SIG = lambda f: (f.estimated_add_ops, f.estimated_mul_ops,
                         f.estimated_div_ops)
        base = self.feat(SOURCE)
        self.assertEqual(SIG(base), (2 * 4096, 2 * 4096, 0))
        add = self.feat(SOURCE.replace("a[i] * s + b[i]", "a[i] + b[i]"))
        mul = self.feat(SOURCE.replace("a[i] * s + b[i]", "a[i] * b[i]"))
        div = self.feat(SOURCE.replace("a[i] * s + b[i]", "a[i] / b[i]"))
        self.assertEqual(SIG(add), (2 * 4096, 1 * 4096, 0))
        self.assertEqual(SIG(mul), (1 * 4096, 2 * 4096, 0))
        self.assertEqual(SIG(div), (1 * 4096, 1 * 4096, 1 * 4096))
        self.assertEqual(len({SIG(add), SIG(mul), SIG(div), SIG(base)}), 4)

    def test_call_sites_counted(self):
        f = self.feat(SOURCE)  # saxpy: no calls
        self.assertEqual(f.num_call_sites, 0)
        relu = self.feat(SOURCE.replace("b[i] = a[i] * s + b[i]",
                                         "b[i] = fmaxf(a[i], 0.0f)"))
        self.assertEqual(relu.num_call_sites, 4096)

    def test_covers_scales_over_covered_launches(self):
        # 64 threads launched, n = 4096: covers = 64, not 4096.
        f = self.feat(SOURCE, n=4096, grid=1, block=64)
        self.assertEqual(f.covers, 64)
        self.assertEqual(f.estimated_bytes_moved, 4 * 4 * 64)

    def test_normalized_length_matches_vector(self):
        from epic_u2o.features import extract_normalized_features
        f = self.feat(SOURCE)
        self.assertEqual(len(extract_normalized_features(SOURCE, None)),
                         len(f.to_vector()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
