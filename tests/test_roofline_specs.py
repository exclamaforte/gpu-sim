"""gpu-sim-g8x: the baseline's hardware constants are measured, single-sourced,
and the register bound can bind. CPU-only (stdlib only; numpy not required).
"""
import ast
import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))  # features.py imports bare `tokenizer`
from epic_u2o import hw_5090  # noqa: E402
from epic_u2o.roofline import (  # noqa: E402
    KernelFeatures as RKF, SPECS, calculate_occupancy)
from epic_u2o.features import RTX_5090_SPECS  # noqa: E402
from epic_u2o.generator_stage1 import kernel_hash_for  # noqa: E402

MEGAKERN = Path("/home/gabe/Documents/megakern/megakern/hw/rtx5090.json")


def fg_defines():
    out = {}
    for m in re.finditer(r"#define\s+(FG_\w+)\s+(\d+)", (REPO / "src/fakegpu_cuda.c").read_text()):
        out[m.group(1)] = int(m.group(2))
    return out


def heavy_regs_features(regs_per_thread=200, block=256):
    return RKF(grid_x=4, grid_y=1, grid_z=1, block_x=block, block_y=1,
               block_z=1, shared_mem_bytes=0,
               estimated_registers_per_thread=regs_per_thread,
               num_args=3, num_pointer_args=2, num_scalar_args=1,
               total_arg_bytes=24, estimated_global_reads=4096,
               estimated_global_writes=4096, estimated_shared_reads=0,
               estimated_shared_writes=0, estimated_bytes_moved=8192,
               estimated_flops=1024, estimated_int_ops=0, has_loops=False,
               has_conditionals=True, max_loop_depth=0, has_syncthreads=False,
               num_syncthreads=0, uses_shared_memory=False,
               uses_constant_memory=False, uses_vector_types=False)


class TestMeasuredConstants(unittest.TestCase):
    def test_u2o19_defines_match(self):
        d = fg_defines()
        self.assertEqual(hw_5090.MAX_THREADS_PER_SM, d["FG_MAX_THREADS_PER_SM"])
        self.assertEqual(hw_5090.SHARED_MEM_PER_SM_BYTES, d["FG_SHMEM_PER_SM"])
        self.assertEqual(hw_5090.L2_CACHE_BYTES, d["FG_L2_BYTES"])
        self.assertEqual(hw_5090.MAX_SMEM_PER_BLOCK_OPTIN, d["FG_OPTIN_PER_BLOCK"])

    def test_megakern_json_matches(self):
        m = json.loads(MEGAKERN.read_text())
        self.assertTrue(m["measured"])
        self.assertEqual(hw_5090.SHARED_MEM_PER_SM_BYTES, m["smem_per_sm"])
        self.assertEqual(hw_5090.L2_CACHE_BYTES, m["l2_bytes"])
        self.assertEqual(hw_5090.REGISTERS_PER_SM, m["regs_per_sm"])
        self.assertEqual(hw_5090.MAX_SM_CLOCK_MHZ, m["max_sm_clock_mhz"])
        self.assertEqual(hw_5090.MEASURED_READ_BW_GBS, m["peak_read_GBps"])
        self.assertEqual(hw_5090.SM_COUNT, m["sm_count"])
        self.assertEqual(hw_5090.MAX_SMEM_PER_BLOCK_OPTIN,
                         m["max_smem_per_block_optin"])

    def test_specs_use_measured_values(self):
        self.assertEqual(SPECS.max_threads_per_sm, 1536)
        self.assertEqual(SPECS.shared_mem_per_sm_bytes, 102400)
        self.assertEqual(SPECS.l2_cache_bytes, 100663296)
        self.assertEqual(SPECS.regs_per_sm, 65536)
        self.assertEqual(SPECS.memory_bandwidth_gbps, 1688.592)
        self.assertEqual(SPECS.boost_clock_mhz, 2422)
        self.assertNotEqual(SPECS.peak_fp32_tflops, 0)  # present but labelled

    def test_features_dict_single_sourced(self):
        self.assertEqual(RTX_5090_SPECS["max_shared_mem_per_block"], 49152)
        self.assertEqual(RTX_5090_SPECS["memory_bandwidth_gbps"], 1688.592)
        self.assertEqual(RTX_5090_SPECS["num_sms"], 170)


class TestRegisterBoundBinds(unittest.TestCase):
    def test_register_heavy_kernel_is_throttled(self):
        # 200 regs/thread x 256 threads = 51200 regs/block; file is 65536,
        # so exactly 1 block fits: occupancy = 8 warps / 64 = 0.125.
        # Under the old (255*2048)//regs formula this bound 10 blocks and
        # never fired; threads alone would allow 6 blocks (0.75).
        occ = calculate_occupancy(heavy_regs_features())
        self.assertAlmostEqual(occ, 0.125)
        self.assertLess(occ, 0.75)

    def test_occupancy_ignores_peak_flops(self):
        a = calculate_occupancy(heavy_regs_features())
        old_peak, SPECS.peak_fp32_tflops = SPECS.peak_fp32_tflops, 1.0
        try:
            self.assertEqual(calculate_occupancy(heavy_regs_features()), a)
        finally:
            SPECS.peak_fp32_tflops = old_peak


class TestOneKernelHash(unittest.TestCase):
    def test_no_second_definition(self):
        files = [REPO / f"epic_u2o/generator_stage{n}.py" for n in (1, 2, 3, 4, 5)]
        files.append(REPO / "epic_u2o/corpus_miner.py")
        found = []
        for f in files:
            tree = ast.parse(f.read_text())
            found += [(str(f), n.name) for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name == "kernel_hash"]
        self.assertEqual(found, [])

    def test_canonical_hash_matches_both_corpora(self):
        from epic_u2o.generator_stage1 import generate_kernels
        for corpus in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
                       "experiments/gpu-sim-e5q/measured.jsonl"):
            rows = [json.loads(l) for l in (REPO / corpus).read_text().splitlines()]
            for r in rows[::57]:
                g, b = r["launch_config"]["grid"], r["launch_config"]["block"]
                ss = generate_kernels(ops=[r["op_type"]], grid_sweeps=[g],
                                      block_sweeps=[b],
                                      problem_sizes=[r["problem_size"]])
                c = [s for s in ss if s.kernel_name == r["kernel_name"]][0]
                self.assertEqual(
                    kernel_hash_for(c.source, c.grid, c.block, r["problem_size"]),
                    r["kernel_hash"], (corpus, r["kernel_hash"]))


if __name__ == "__main__":
    unittest.main()
