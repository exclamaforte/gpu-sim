#!/usr/bin/env python3
"""gpu-sim-u2o.11.3: dimension scalars reach the launch call.

Without this, every stage-4 driver launched with M=N=K=1 (int default) and
measured the launcher, not the GEMM -- and every 2D driver launched with
rows=cols=1. Stage-1 drivers must be byte-identical to before (no colliding
metadata keys there).

s4 re-run: GEMM pointer buffers are arity-sized (A: M*K, B: K*N, C: M*N).
The generic N-per-buffer rule filled gigabytes on the host while the timed
kernel ran microseconds (5.5% GPU util, killed by the queue's 10% gate).
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.kernel_driver import synthesize  # noqa: E402
from epic_u2o.generator_stage1 import generate_kernels as gen1  # noqa: E402
import epic_u2o.generator_stage4 as s4  # noqa: E402


def _timed_call(prog):
    """The timed launch line (8-space indent), not the warmup loop."""
    calls = [l for l in prog.splitlines()
             if l.startswith("        ") and "<<<" in l]
    assert len(calls) == 1, calls
    return calls[0].rstrip().rstrip(";")


def _spec(**kw):
    d = {"source": "__global__ void k(float *o, int n) { o[0] = n; }",
         "kernel_name": "k", "grid": [1, 1, 1], "block": [32, 1, 1],
         "args": [{"name": "o", "type": "float*", "ptr": True},
                  {"name": "n", "type": "int", "ptr": False}],
         "problem_size": 1024, "shared_mem_bytes": 0, "metadata": {}}
    d.update(kw)
    return d


class TestDimScalars(unittest.TestCase):
    def test_metadata_dims_win_over_int_default(self):
        prog = synthesize(_spec(
            args=[{"name": "M", "type": "int", "ptr": False},
                  {"name": "K", "type": "int", "ptr": False}],
            metadata={"M": 512, "K": 128, "generator": "x"}))
        self.assertTrue(_timed_call(prog).endswith("(512, 128)"),
                        _timed_call(prog))

    def test_metadata_dims_win_over_size_names(self):
        """GEMM's N is a dimension, not the element count."""
        prog = synthesize(_spec(
            args=[{"name": "M", "type": "int", "ptr": False},
                  {"name": "N", "type": "int", "ptr": False},
                  {"name": "K", "type": "int", "ptr": False}],
            metadata={"M": 64, "N": 128, "K": 256}))
        self.assertTrue(_timed_call(prog).endswith("(64, 128, 256)"),
                        _timed_call(prog))

    def test_size_names_still_map_to_N(self):
        prog = synthesize(_spec(metadata={"M": 512}))
        self.assertTrue(_timed_call(prog).endswith("(d_o, N)"),
                        _timed_call(prog))

    def test_stage1_driver_unchanged(self):
        s = next(gen1(ops=["scale"], grid_sweeps=[[4, 1, 1]],
                      block_sweeps=[[256, 1, 1]], problem_sizes=[4096],
                      max_kernels=1)).to_dict()
        prog = synthesize(s)
        self.assertNotIn("dims", prog)
        self.assertIn("const int N = 4096;", prog)

    def test_real_stage4_spec_carries_dims_to_launch(self):
        specs = [s for s in s4.generate_kernels(max_kernels=100000)
                 if s.op_type == "gemm_naive"][:2]
        self.assertTrue(specs)
        for s in specs:
            m = s.to_dict()["metadata"]
            prog = synthesize(s.to_dict())
            tail = f", {m['M']}, {m['N']}, {m['K']})"
            self.assertTrue(_timed_call(prog).endswith(tail),
                            _timed_call(prog))

    def test_real_2d_spec_carries_rows_cols(self):
        import epic_u2o.generator_stage2 as s2
        s = next(s for s in s2.generate_kernels(max_kernels=100000)
                 if s.op_type == "stencil_2d_5pt")
        m = s.to_dict()["metadata"]
        prog = synthesize(s.to_dict())
        tail = f", {m['rows']}, {m['cols']})"
        self.assertTrue(_timed_call(prog).endswith(tail),
                        _timed_call(prog))


class TestGemmBufferCounts(unittest.TestCase):
    def _gemm(self, m=512, n=512, k=256):
        s = next(s for s in s4.generate_kernels(max_kernels=100000)
                 if s.op_type == "gemm_tiled" and s.metadata["M"] == m
                 and s.metadata["N"] == n and s.metadata["K"] == k)
        return s.to_dict()

    def test_buffers_are_arity_sized(self):
        prog = synthesize(self._gemm())
        # A: 512*256, B: 256*512, C: 512*512 -- none of them M*N*K.
        for frag in ("131072 * sizeof(float)",  # A and B
                     "262144 * sizeof(float)"):  # C
            self.assertIn(frag, prog)
        self.assertNotIn("134217728 * sizeof", prog)  # M*N*K must be gone

    def test_fills_match_mallocs(self):
        prog = synthesize(self._gemm())
        self.assertIn("for (int i = 0; i < 131072; i++)", prog)

    def test_missing_dims_fall_back_to_N(self):
        d = self._gemm()
        d["metadata"] = {"generator": "stage4_gemm"}  # dims stripped
        prog = synthesize(d)
        self.assertIn(f"{d['problem_size']} * sizeof(float)", prog)

    def test_nongemm_buffers_stay_N(self):
        s = next(gen1(ops=["scale"], grid_sweeps=[[4, 1, 1]],
                      block_sweeps=[[256, 1, 1]], problem_sizes=[4096],
                      max_kernels=1)).to_dict()
        prog = synthesize(s)
        self.assertIn("4096 * sizeof(float)", prog)


if __name__ == "__main__":
    unittest.main(verbosity=2)
