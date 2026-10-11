"""gpu-sim-46p: the memory arm takes the achieved bandwidth flat.

- memory_time is occupancy-INDEPENDENT (same bytes, different occupancy ->
  identical memory time; compute still derates).
- THEORETICAL_READ_BW_GBS pinned to the json's theoretical figure, and
  MEASURED below it (the two labels never silently interchange).
CPU-only.
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))
from epic_u2o import hw_5090  # noqa: E402
from epic_u2o.roofline import (  # noqa: E402
    KernelFeatures, calculate_compute_time, calculate_memory_time,
    calculate_occupancy)

MEGAKERN = Path("/home/gabe/Documents/megakern/megakern/hw/rtx5090.json")


def feats(regs, bytes_moved=50_000_000):
    return KernelFeatures(
        grid_x=65536, grid_y=1, grid_z=1, block_x=64, block_y=1, block_z=1,
        shared_mem_bytes=0, estimated_registers_per_thread=regs,
        num_args=4, num_pointer_args=2, num_scalar_args=2, total_arg_bytes=24,
        estimated_global_reads=2 * 4194304, estimated_global_writes=4194304,
        estimated_shared_reads=0, estimated_shared_writes=0,
        estimated_bytes_moved=bytes_moved, estimated_flops=4194304,
        estimated_int_ops=0, has_loops=False, has_conditionals=True,
        max_loop_depth=0, has_syncthreads=False, num_syncthreads=0,
        uses_shared_memory=False, uses_constant_memory=False,
        uses_vector_types=False)


class TestFlatMemoryArm(unittest.TestCase):
    def test_memory_time_ignores_occupancy(self):
        lo, hi = feats(regs=24), feats(regs=200)
        self.assertNotEqual(calculate_occupancy(lo), calculate_occupancy(hi))
        self.assertAlmostEqual(
            calculate_memory_time(lo, calculate_occupancy(lo)),
            calculate_memory_time(hi, calculate_occupancy(hi)))
        # ... while the compute arm still derates (occupancy has a job).
        self.assertNotEqual(
            calculate_compute_time(lo, calculate_occupancy(lo)),
            calculate_compute_time(hi, calculate_occupancy(hi)))

    def test_memory_time_is_bytes_over_achieved(self):
        f = feats(regs=24)
        self.assertAlmostEqual(
            calculate_memory_time(f, 0.6842),
            50_000_000 / (1688.592 * 1e9) * 1000.0)

    def test_theory_and_achieved_carried_separately(self):
        m = json.loads(MEGAKERN.read_text())
        self.assertEqual(hw_5090.THEORETICAL_READ_BW_GBS,
                         m["theoretical_read_bw_GBs"])
        self.assertEqual(hw_5090.MEASURED_READ_BW_GBS, m["peak_read_GBps"])
        self.assertLess(hw_5090.MEASURED_READ_BW_GBS,
                        hw_5090.THEORETICAL_READ_BW_GBS)


if __name__ == "__main__":
    unittest.main()
