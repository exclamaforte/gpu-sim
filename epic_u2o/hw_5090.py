#!/usr/bin/env python3
"""RTX 5090 hardware constants, SINGLE-SOURCED with provenance (gpu-sim-g8x).

Every number the Phase-2 baseline treats as a hardware fact lives here, once,
with the measurement that produced it. Import this; do not re-type literals
(the roofline used 2048/228KB/64MB/1792/2520 from no source, all optimistically
wrong; features.py carried its own copies).

Source classes:
  MEASURED/OWN  - gpu-sim's own nvcc probes on this 5090. u2o.19 closed the
                  struct+attribute values as FG_* defines in src/fakegpu_cuda.c
                  (probes ts 20260928t105453-2dc7af7c, 20261010t211802-ff147780,
                  20261010t213835-4d4bdf3f; the last reads struct==attribute
                  to the byte on all four channels).
  MEASURED/XLAB - megakern's independent pass on the same silicon:
                  megakern/hw/rtx5090.json, measured=true, source_bead
                  megakern-37e.2, ts 20261003t213327-db6b7f3f.
  ARCH          - CUDA architecture invariants (same on every GPU of the arch;
                  wrong only after a toolkit/arch change, which is a re-probe).
  VENDOR        - vendor spec-sheet figure with NO in-tree measurement.
                  Labelled as such; never allowed into the occupancy path
                  (occupancy must be built from measured quantities only).

Do NOT tune these to make a baseline fit: documented bias -> undocumented
bias is a downgrade (g8x hard rule).
"""

# ---- measured: gpu-sim's own probes (u2o.19 FG_* defines) + megakern ------
MAX_THREADS_PER_SM = 1536       # OWN (FG_MAX_THREADS_PER_SM; attr readthrough
                                # 1536 agrees; no megakern field exists)
SHARED_MEM_PER_SM_BYTES = 102400  # OWN (FG_SHMEM_PER_SM) + XLAB smem_per_sm
L2_CACHE_BYTES = 100663296      # OWN (FG_L2_BYTES) + XLAB l2_bytes (96 MiB)
REGISTERS_PER_SM = 65536        # XLAB regs_per_sm + OWN regsPerBlock/SM 65536
MAX_SMEM_PER_BLOCK_OPTIN = 101376  # OWN probe + XLAB max_smem_per_block_optin
MAX_SM_CLOCK_MHZ = 2422         # XLAB max_sm_clock_mhz
MEASURED_READ_BW_GBS = 1688.592  # XLAB peak_read_GBps (achievable, not the
                                # 1792.0 theoretical figure also in that file)
SM_COUNT = 170                  # XLAB sm_count + OWN multiProcessorCount
SHARED_MEM_PER_BLOCK = 49152    # OWN sharedMemPerBlock
MAX_THREADS_PER_BLOCK = 1024    # OWN maxThreadsPerBlock

# ---- architectural invariants (CUDA, not card-specific) --------------------
WARP_SIZE = 32                  # ARCH
MAX_REGISTERS_PER_THREAD = 255  # ARCH (per-thread cap, not a file size)
MAX_BLOCKS_PER_SM = 32          # ARCH
MAX_WARPS_PER_SM = 64           # ARCH

# ---- vendor figures: NO in-tree measurement; kept out of occupancy ---------
# peak_fp32_tflops ships 82.6 with no source anywhere. It is retained ONLY as
# a labelled vendor figure for the compute ceiling, because deleting it would
# delete the compute path, and re-sourcing it is a measurement bead, not this
# one. It must not enter calculate_occupancy (a test pins that).
PEAK_FP32_TFLOPS = 82.6         # VENDOR, UNSOURCED IN-TREE
PEAK_FP16_TFLOPS = 165.2        # VENDOR, UNSOURCED IN-TREE
PEAK_INT32_TOPS = 82.6          # VENDOR, UNSOURCED IN-TREE
BASE_CLOCK_MHZ = 2235           # VENDOR
L1_CACHE_KB_PER_SM = 128        # VENDOR/ARCH

# ---- launch floor: MEASURED, this lab, this path (gpu-sim-zhq) -----------
# gpu-sim-u2o.6.6.1 decomposed reported time on this exact harness path:
# irreducible_launch_us = 1.308/1.330/1.365 at n = 1K/16K/256K (ts final
# job 20260927t162535-cb00ea78, metrics.json noise_floor). Rule for the
# single constant: the max over the n-range the corpus's launch-bound rows
# occupy (all <= 2^18 < 262144). NOT tuned to the baseline error.
# Context (not used): at n = 4M/64M the 'irreducible' term itself grows
# (8.1/31.0 us) -- that is real work entering, past launch_bound_crossover_n
# = 67108864. megakern's end-to-end stream launch (2.8455 us) is a different
# quantity (submit-to-start incl. queueing), not this floor.
LAUNCH_FLOOR_US = 1.3652799999999998  # raw 6.6.1 value at n=256K, unrounded