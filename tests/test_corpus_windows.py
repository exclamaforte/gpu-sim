#!/usr/bin/env python3
"""gpu-sim-u2o.7.6: the miner's window defect and the idiom list stay pinned.

The reviewer's item: the miner has a documented 60-line window defect, the
whole-file idiom counts are the load-bearing numbers, and neither is asserted
anywhere. A change to the window width, the idiom list or the root set would
move the headline silently -- and the failure this bead exists to have caught
is the one with no guard against it returning.

Two tests, both CPU-only, both on fixture trees (the real corpus at
/tmp/cutlass is gone; fixtures make the property independent of it):

  1. test_whole_file_coverage_catches_what_the_window_misses: a fixture
     kernel using mma.sync and atomicAdd ONLY past line 60. The project's
     real window extractor (epic_u2o/corpus_miner) must MISS them -- that is
     the documented defect, reproduced -- while this bead's whole-file
     file_level_coverage must FIND them. If either side changes, one of the
     two assertions fires.
  2. test_load_bearing_idioms_are_in_the_list: the IDIOMS entries the gap
     analysis stands on (tensor-core wmma/mma.sync, atomicAdd, __shfl) must
     be present, and gap() must derive n_gaps from the measured sets. An
     edited list or a hand-typed count fails here instead of moving the
     headline silently.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.7.6"))

import coverage_vs_synthetic as cvs  # noqa: E402
from epic_u2o.corpus_miner import extract_kernels_from_file  # noqa: E402


def fixture_tree(names_to_bodies: dict) -> Path:
    import tempfile
    td = Path(tempfile.mkdtemp(prefix="corpus_fix_"))
    for name, body in names_to_bodies.items():
        (td / name).write_text(body)
    return td


def late_idiom_kernel() -> str:
    lines = ["// fixture: idioms live ONLY past line 60",
             "__global__ void late_kernel(float *p) {",
             "  int i = blockIdx.x * blockDim.x + threadIdx.x;"]
    lines += [f"  float t{j} = p[i] + {j}.0f;" for j in range(60)]
    lines += ["  __shared__ float tile[256];",
              "  tile[threadIdx.x] = t0;",
              "  __syncthreads();",
              "  atomicAdd(&p[i], tile[threadIdx.x]);",
              "  p[i] = t1; // mma.sync used below",
              "  // wmma fragment and mma.sync would be instantiated here",
              "  asm(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\");",
              "}"]
    return "\n".join(lines) + "\n"


class TestWindowDefectStaysPinned(unittest.TestCase):
    def test_whole_file_coverage_catches_what_the_window_misses(self):
        td = fixture_tree({"late.cu": late_idiom_kernel()})
        try:
            files = [td / "late.cu"]
            whole = cvs.file_level_coverage(files)
            self.assertIn("atomics", whole["capabilities_present"],
                          "whole-file coverage must find the late atomicAdd")
            self.assertIn("tensor cores", whole["capabilities_present"],
                          "whole-file coverage must find the late mma.sync")
            windowed = list(extract_kernels_from_file(
                td / "late.cu", provenance="fixture"))
            self.assertTrue(windowed, "fixture must yield at least one kernel")
            window_text = "\n".join(k.source for k in windowed)
            self.assertNotIn("atomicAdd", window_text,
                             "the 60-line window must MISS the late idiom -- "
                             "that miss is the documented defect this pins")
            self.assertNotIn("mma.sync", window_text,
                             "the 60-line window must MISS the late mma.sync")
        finally:
            import shutil
            shutil.rmtree(td, ignore_errors=True)


class TestIdiomListStaysPinned(unittest.TestCase):
    def test_load_bearing_idioms_are_listed(self):
        flat = [i for idioms in cvs.IDIOMS.values() for i in idioms]
        for must in ("wmma", "mma.sync", "atomicAdd", "__shfl",
                     "__shared__", "__syncthreads", "cudaStream"):
            self.assertIn(must, flat,
                          f"{must} removed from IDIOMS moves the headline silently")

    def test_gap_count_is_derived_not_typed(self):
        real = {"capabilities_present": ["tensor cores", "atomics"],
                "capabilities": ["tensor cores", "atomics"]}
        synth = {"capabilities": ["atomics"]}
        g = cvs.gap(real, synth)
        self.assertEqual(g["n_gaps"], 1)
        self.assertEqual(g["real_code_uses_synthetic_lacks"], ["tensor cores"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
