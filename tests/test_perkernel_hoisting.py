#!/usr/bin/env python3
"""Pin the per-kernel hoisting, because a silent regex failure is invisible.

I hypothesised that the geometry hoisting was failing SILENTLY -- a regex that
matched nothing would leave `const int N` in place, the runtime globals would
never be set, and the kernels would run against the wrong size. That is exactly
the defect class this project has catalogued all night, and it was the most
plausible explanation for `loop exited 3`.

CHECKED IT. It was wrong: all three regexes match exactly once and leave zero
declarations behind. So the hypothesis is REFUTED and the real cause of the
failure is still unknown.

Which is why this test exists. The hypothesis was cheap to check and would have
been expensive to act on -- so the check is now a test, and the refutation is
recorded here rather than left in a shell history.

Run:  python3 tests/test_perkernel_hoisting.py
"""
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.kernel_driver import synthesize  # noqa: E402

HOIST = (
    (r"(\s*)const int N = \d+;", r"\1N = 0;"),
    (r"(\s*)dim3 grid\(([^)]*)\);", r"\1grid = dim3(0, 0, 0);"),
    (r"(\s*)dim3 block\(([^)]*)\);", r"\1block = dim3(0, 0, 0);"),
)

SPEC = {
    "op_type": "saxpy", "kernel_name": "saxpy_kernel",
    "source": ("__global__ void saxpy_kernel(float alpha, const float* a, "
               "const float* b, float* y, int n){ int i=blockIdx.x*blockDim.x"
               "+threadIdx.x; if(i<n) y[i]=alpha*a[i]+b[i];}"),
    "problem_size": 4096, "grid": [64, 1, 1], "block": [64, 1, 1],
    "args": [{"name": "a", "type": "const float*"},
             {"name": "b", "type": "const float*"},
             {"name": "y", "type": "float*"},
             {"name": "n", "type": "int"}],
}


class TestHoistingMatches(unittest.TestCase):
    def source(self):
        return synthesize(SPEC)

    def test_each_pattern_matches_exactly_once(self):
        src = self.source()
        for pat, _ in HOIST:
            self.assertEqual(len(re.findall(pat, src)), 1,
                             f"{pat!r} matched {len(re.findall(pat, src))} "
                             f"times; a silent zero-match is the failure mode "
                             f"this test exists to prevent")

    def test_no_declaration_survives_hoisting(self):
        out = self.source()
        for pat, rep in HOIST:
            out = re.sub(pat, rep, out)
        for decl in (r"\bconst int N\b", r"\bdim3 grid\s*\(", r"\bdim3 block\s*\("):
            self.assertIsNone(re.search(decl, out),
                              f"{decl!r} survived the hoisting, so the runtime "
                              f"globals would never be set and the kernel would "
                              f"run against its baked-in size")

    def test_the_assignment_targets_the_runtime_global(self):
        out = self.source()
        for pat, rep in HOIST:
            out = re.sub(pat, rep, out)
        self.assertRegex(out, r"(^|\s)N = 0;")
        self.assertRegex(out, r"(^|\s)grid = dim3\(0, 0, 0\);")
        self.assertRegex(out, r"(^|\s)block = dim3\(0, 0, 0\);")

    def test_the_kernel_body_is_untouched(self):
        """The confound is in the BODY and the ALLOCATIONS, so hoisting must
        not touch them. If a future edit rewrites the body, this experiment is
        measuring something else again."""
        out = self.source()
        for pat, rep in HOIST:
            out = re.sub(pat, rep, out)
        self.assertIn("__global__ void saxpy_kernel", out)
        self.assertEqual(out.count("cudaMalloc"), self.source().count("cudaMalloc"))


class TestTheDriverNamesItsVictim(unittest.TestCase):
    def test_dispatch_failure_is_reported_with_the_spec_index(self):
        """`loop exited 3` alone made 264 specs indistinguishable."""
        tpl = (REPO / "experiments" / "gpu-sim-u2o.6.7.1"
               / "perkernel_loop.cu.in").read_text()
        self.assertIn("DISPATCH_FAIL spec=", tpl)
        self.assertIn("FAILED %%d %%d", tpl)

    def test_the_driver_is_a_file_not_a_generated_string(self):
        """Five build iterations were escaping bugs in a Python string holding
        C. Keeping the driver in a .cu file removes the whole class, and this
        test stops someone putting it back."""
        p = (REPO / "experiments" / "gpu-sim-u2o.6.7.1"
             / "perkernel_confound.py").read_text()
        self.assertNotIn("DISPATCH_LOOP = r", p,
                         "the driver template is back inside a Python string; "
                         "five iterations were lost to escaping that way")
        self.assertIn("perkernel_loop.cu.in", p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
