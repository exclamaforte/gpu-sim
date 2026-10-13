#!/usr/bin/env python3
"""OOB guard regression sweep (gpu-sim-f6y).

The driver preflight (epic_u2o/kernel_driver.py: check_subscripts_bounded)
must refuse provably-out-of-bounds kernels on CPU and must never refuse a
correct one. Both directions are pinned here:

1. test_broken_templates_refused (negative control, item 6): the exact
   pre-fix segmented_read / strided_write sources -- kept as literals so
   the test survives the generator fix -- are refused with messages naming
   the buffer, subscript, reachable max, and capacity.
2. test_manifest_all_pass: every one of the 2322 manifest occurrences
   synthesizes under the CURRENT (fixed) templates. Zero refusals allowed.
3. test_fixed_cells_pass: the 80 occurrences under the 18 missing hashes
   (the f6y measurement set) all synthesize.
4. test_unchecked_passthrough: data-dependent families (gather, scatter,
   random) synthesize WITHOUT refusal and report unchecked notes -- the
   guard's honest scope boundary.
5. test_guard_rules: synthetic kernels pinning each narrowing rule
   (bare `if (i<n)` over-covered, `&&`-condition, monotone inversion,
   pre-increment loop var) so the rules survive template refactors.

CPU-only. No GPU, no queue, no network.
"""

import json
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from epic_u2o.kernel_driver import synthesize  # noqa: E402

MANIFEST = os.path.join(
    REPO, "experiments", "gpu-sim-u2o.11.3", "manifest.json")
MEASURE_SUMMARY = os.path.join(
    REPO, "experiments", "gpu-sim-u2o.11.3", "measure_summary.json")

# Exact pre-fix sources (git HEAD before the f6y template fix). Pinned as
# literals: the negative control must not depend on the generator it tests.
BROKEN_SEGMENTED = """
__global__ void segmented_read(const float* __restrict__ in, float* __restrict__ out, int n, int segment_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int segment = i / segment_size;
        int offset = i % segment_size;
        out[i] = in[segment * n + offset];
    }
}
"""

BROKEN_STRIDED_WRITE = """
__global__ void strided_write(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i * stride] = in[i];
}
"""

PTR_ARGS = [
    {"name": "in", "type": "const float* __restrict__"},
    {"name": "out", "type": "float* __restrict__"},
]
INT_ARG = {"name": "segment_size", "type": "int"}
STRIDE_ARG = {"name": "stride", "type": "int"}
N_ARG = {"name": "n", "type": "int"}


def _spec(source, op, n, grid, block, meta):
    return {
        "source": source,
        "kernel_name": op,
        "op_type": op,
        "grid": grid,
        "block": block,
        "args": PTR_ARGS + [N_ARG]
        + ([INT_ARG] if "segment_size" in meta else [STRIDE_ARG]),
        "problem_size": n,
        "metadata": dict(meta),
        "shared_mem_bytes": 0,
    }


def _load_generators():
    import epic_u2o.generator_stage1 as s1
    import epic_u2o.generator_stage2 as s2
    import epic_u2o.generator_stage3 as s3
    import epic_u2o.generator_stage4 as s4
    import epic_u2o.generator_stage5 as s5
    by_op = {}
    for mod in (s1, s2, s3, s4, s5):
        for s in mod.generate_kernels(max_kernels=100000):
            by_op.setdefault(s.op_type, []).append(s.to_dict())
    return by_op


class TestOOBGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MANIFEST) as f:
            cls.manifest = json.load(f)["specs"]
        with open(MEASURE_SUMMARY) as f:
            cls.missing_hashes = set(
                json.load(f)["missing_hashes"])
        cls.by_op = _load_generators()

    def _manifest_spec(self, m):
        d0 = self.by_op[m["op_type"]][0]
        d = dict(d0)
        d.update(grid=m["grid"], block=m["block"],
                 problem_size=m["problem_size"])
        md = dict(d.get("metadata") or {})
        md.update(m.get("metadata") or {})
        d["metadata"] = md
        return d

    def test_broken_templates_refused(self):
        """Negative control: the pre-fix sources MUST be refused on CPU."""
        with self.assertRaises(ValueError) as cm:
            synthesize(_spec(BROKEN_SEGMENTED, "segmented_read", 262144,
                             [256, 1, 1], [1024, 1, 1],
                             {"segment_size": 16384}))
        msg = str(cm.exception)
        self.assertIn("oob-guard", msg)
        self.assertIn("in[segment * n + offset]", msg)
        self.assertIn("3948543", msg)  # reachable max, not a vague warning
        self.assertIn("262144", msg)  # capacity
        with self.assertRaises(ValueError) as cm2:
            synthesize(_spec(BROKEN_STRIDED_WRITE, "strided_write", 262144,
                             [256, 1, 1], [1024, 1, 1], {"stride": 8}))
        msg2 = str(cm2.exception)
        self.assertIn("oob-guard", msg2)
        self.assertIn("out[i * stride]", msg2)
        self.assertIn("2097144", msg2)

    def test_broken_safe_variants_pass(self):
        """segment_size/stride values that keep every access in bounds pass:
        refusal is about proven overflow, not the template name."""
        synthesize(_spec(BROKEN_SEGMENTED, "segmented_read", 128,
                         [1, 1, 1], [128, 1, 1], {"segment_size": 256}))
        synthesize(_spec(BROKEN_STRIDED_WRITE, "strided_write", 4096,
                         [4, 1, 1], [1024, 1, 1], {"stride": 1}))

    def test_manifest_all_pass(self):
        """All 2322 manifest occurrences synthesize under current templates."""
        refused = []
        for m in self.manifest:
            if m["op_type"] not in self.by_op:
                continue
            try:
                synthesize(self._manifest_spec(m))
            except ValueError as e:
                refused.append((m["op_type"], m["problem_size"],
                                str(e)[:80]))
        self.assertEqual(refused, [],
                         f"{len(refused)} manifest cells refused: "
                         f"{refused[:5]}")

    def test_fixed_cells_pass(self):
        """The 80 occurrences under the 18 missing hashes synthesize."""
        cells = [m for m in self.manifest
                 if m["kernel_hash"] in self.missing_hashes]
        self.assertEqual(len(cells), 80)
        for m in cells:
            synthesize(self._manifest_spec(m))

    def test_unchecked_passthrough(self):
        """Data-dependent families pass with unchecked notes, never refuse."""
        from epic_u2o.kernel_driver import check_subscripts_bounded
        import epic_u2o.generator_stage3 as s3
        gen = {s.op_type: s.to_dict()
               for s in s3.generate_kernels(max_kernels=100000)}
        for op in ("gather_read", "scatter_write", "random_access"):
            d = gen[op]
            n = 262144
            threads = 256 * 1024
            caps = {a["name"]: n for a in d["args"]
                    if "*" in a.get("type", "")}
            dims = {k: v for k, v in (d.get("metadata") or {}).items()
                    if type(v) is int}
            checked, unchecked = check_subscripts_bounded(
                d["source"], kernel_name=op, args=d["args"],
                capacities=caps, scalars=dict(dims, n=n),
                n=n, threads=threads)
            self.assertTrue(unchecked,
                            f"{op}: expected unchecked notes, got none")

    def test_guard_rules(self):
        """Synthetic kernels pinning each narrowing rule."""
        wrap = ("__global__ void k(const float* __restrict__ in, "
                "float* __restrict__ out, int n, int stride) {\n"
                "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n%s\n}\n")
        base = {"kernel_name": "k", "op_type": "k", "problem_size": 1048576,
                "grid": [4096, 1, 1], "block": [1024, 1, 1],
                "args": PTR_ARGS + [N_ARG, STRIDE_ARG],
                "metadata": {"stride": 8}, "shared_mem_bytes": 0}
        # bare guard, 4x over-covered: in-bounds only via the guard.
        d = dict(base, source=wrap % "    if (i < n) out[i] = in[i];")
        synthesize(d)
        # &&-condition read: covered left-to-right past the &&.
        d = dict(base, source=wrap % (
            "    if (i < n && in[i] > 0.0f) out[i] = in[i];"))
        synthesize(d)
        # monotone inversion: i*stride<n bounds i itself.
        d = dict(base, source=wrap % (
            "    if (i * stride < n) out[i] = in[i * stride];"))
        synthesize(d)
        # pre-increment loop var is unknown, never const-0: passthrough.
        src = ("__global__ void k(const float* __restrict__ in, "
               "float* __restrict__ out, int n, int stride) {\n"
               "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
               "    if (i < n) {\n"
               "        float acc = 0.0f;\n"
               "        for (int k = 0; k < 8; ++k) acc += in[(i + k) % n];\n"
               "        out[i] = acc;\n    }\n}\n")
        synthesize(dict(base, source=src))
        # genuinely OOB even with the bare guard: must refuse.
        d = dict(base, source=wrap % "    if (i < n) out[i * stride] = in[i];")
        with self.assertRaises(ValueError):
            synthesize(d)


if __name__ == "__main__":
    unittest.main()
