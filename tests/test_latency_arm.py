"""gpu-sim-zhq: the latency arm is the measured floor, nothing invented.

- calculate_latency_time returns hw_5090.LAUNCH_FLOOR_US for ANY launch
  config (the grid-scaled sync term and the wave term are deleted).
- No invented constants remain in the latency path (AST: the function body
  may reference only hw_5090.LAUNCH_FLOOR_US and the ms conversion).
- The floor reconciles with 6.6.1's measured sub-2^18 floors.
CPU-only.
"""
import ast
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))
from epic_u2o import hw_5090  # noqa: E402
from epic_u2o.roofline import KernelFeatures, calculate_latency_time  # noqa: E402


def feats(grid=32768, block=1024):
    return KernelFeatures(grid_x=grid, grid_y=1, grid_z=1, block_x=block,
                          block_y=1, block_z=1, shared_mem_bytes=0,
                          estimated_registers_per_thread=24, num_args=3,
                          num_pointer_args=2, num_scalar_args=1,
                          total_arg_bytes=24, estimated_global_reads=4096,
                          estimated_global_writes=4096, estimated_shared_reads=0,
                          estimated_shared_writes=0, estimated_bytes_moved=8192,
                          estimated_flops=1024, estimated_int_ops=0,
                          has_loops=False, has_conditionals=True,
                          max_loop_depth=0, has_syncthreads=False,
                          num_syncthreads=0, uses_shared_memory=False,
                          uses_constant_memory=False, uses_vector_types=False)


class TestLatencyFloor(unittest.TestCase):
    def test_latency_is_the_measured_floor(self):
        self.assertAlmostEqual(calculate_latency_time(feats()), hw_5090.LAUNCH_FLOOR_US / 1000.0)
        # grid-independent: the deleted sync term scaled with grid
        self.assertAlmostEqual(calculate_latency_time(feats(grid=1)), hw_5090.LAUNCH_FLOOR_US / 1000.0)
        self.assertAlmostEqual(calculate_latency_time(feats(grid=32768)),
                               calculate_latency_time(feats(grid=1)))

    def test_floor_matches_hw_module(self):
        self.assertEqual(hw_5090.LAUNCH_FLOOR_US, 1.3652799999999998)

    def test_no_invented_constants_in_latency_path(self):
        src = Path("epic_u2o/roofline.py").read_text() \
            if Path("epic_u2o/roofline.py").exists() \
            else (REPO / "epic_u2o/roofline.py").read_text()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "calculate_latency_time")
        nums = [n.value for n in ast.walk(fn)
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]
        self.assertEqual(sorted(set(nums)), [1000.0],
                         f"invented numbers in latency path: {sorted(set(nums))}")

    def test_floor_reconciles_with_661(self):
        m = json.loads((REPO / "experiments/gpu-sim-u2o.6.6.1/metrics.json").read_text())
        small = [m["noise_floor"]["reported_time_decomposition"][k]["irreducible_launch_us"]
                 for k in ("1024", "16384", "262144")]
        self.assertEqual(hw_5090.LAUNCH_FLOOR_US, max(small))


if __name__ == "__main__":
    unittest.main()
