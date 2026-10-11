#!/usr/bin/env python3
"""Analytic roofline model baseline for CUDA runtime prediction.

Classic max(compute_time, memory_time) roofline estimate from RTX 5090's
peak FLOPs/bandwidth spec sheet and the static feature vector.
Zero learned parameters - the floor every learned model must beat.
"""

from dataclasses import dataclass
from typing import Optional
import math

from epic_u2o import hw_5090


@dataclass
class RTX5090Specs:
    """RTX 5090 (Blackwell) specifications."""
    # Compute (vendor figures where marked; measured quantities live in
    # epic_u2o.hw_5090 and are imported as defaults below)
    peak_fp32_tflops: float = hw_5090.PEAK_FP32_TFLOPS  # VENDOR, UNSOURCED IN-TREE -- never an occupancy input
    # (8.3 D2: the sparsity-variant literals that shadowed the two lines
    # above are deleted -- a third copy of two constants with no reader.)
    
    # Memory
    # Memory (measured achievable quantities, not vendor theory)
    memory_bandwidth_gbps: float = hw_5090.MEASURED_READ_BW_GBS  # 1688.592, megakern-37e.2 peak_read (theory 1792)
    l2_cache_bytes: int = hw_5090.L2_CACHE_BYTES  # 100663296 (96 MiB), own probe + megakern-37e.2
    l1_cache_kb_per_sm: float = hw_5090.L1_CACHE_KB_PER_SM
    
    # Architecture
    # Architecture (measured quantities imported; the rest are arch invariants)
    num_sms: int = hw_5090.SM_COUNT  # 170, megakern-37e.2 + own multiProcessorCount
    warp_size: int = hw_5090.WARP_SIZE  # ARCH
    max_threads_per_sm: int = hw_5090.MAX_THREADS_PER_SM  # 1536, own probe + attr readthrough (was 2048)
    max_blocks_per_sm: int = hw_5090.MAX_BLOCKS_PER_SM  # ARCH
    max_warps_per_sm: int = hw_5090.MAX_WARPS_PER_SM  # ARCH
    max_registers_per_thread: int = hw_5090.MAX_REGISTERS_PER_THREAD  # ARCH cap (not a file size)
    regs_per_sm: int = hw_5090.REGISTERS_PER_SM  # 65536, megakern-37e.2 + own probe (the real file size)
    shared_mem_per_sm_bytes: int = hw_5090.SHARED_MEM_PER_SM_BYTES  # 102400, own probe + megakern-37e.2 (was 228KB)
    
    # Clock
    # Clock (measured max, not the vendor boost figure)
    base_clock_mhz: float = hw_5090.BASE_CLOCK_MHZ  # VENDOR
    boost_clock_mhz: float = hw_5090.MAX_SM_CLOCK_MHZ  # 2422, megakern-37e.2 (was vendor 2520)


# Global specs instance
SPECS = RTX5090Specs()


# gpu-sim-u2o.8.3 D2: the hand-copied KernelFeatures mirror that stood here is
# DELETED. This module imports the struct from features.py -- one definition,
# not two. (Verified the acceptance way: by deleting this copy and watching
# the suite pass, not by asserting the copies match.)
from epic_u2o.features import KernelFeatures  # noqa: E402
@dataclass
class RooflinePrediction:
    """Roofline model prediction result."""
    compute_time_ms: float
    memory_time_ms: float
    predicted_time_ms: float
    bottleneck: str  # 'compute', 'memory', 'latency'
    achieved_occupancy: float
    arithmetic_intensity: float  # FLOP/byte
    compute_ceiling_tflops: float
    memory_ceiling_gbps: float


def calculate_occupancy(features: KernelFeatures) -> float:
    """Calculate theoretical occupancy based on register and shared memory usage."""
    specs = SPECS
    
    threads_per_block = features.block_x * features.block_y * features.block_z
    
    # Register limit
    regs_per_thread = min(features.estimated_registers_per_thread, specs.max_registers_per_thread)
    regs_per_block = regs_per_thread * threads_per_block
    # Register limit. gpu-sim-g8x: the old formula multiplied the per-thread
    # CAP (255) by a thread count to fake a file size (255*2048 = 522240 -- a
    # number with no hardware meaning, ~7.96x the real file, so this bound
    # never fired). The bound is register FILE / registers per block.
    max_blocks_regs = specs.regs_per_sm // regs_per_block if regs_per_block > 0 else specs.max_blocks_per_sm
    
    # Shared memory limit
    shared_mem_per_block = features.shared_mem_bytes
    max_blocks_smem = specs.shared_mem_per_sm_bytes // shared_mem_per_block if shared_mem_per_block > 0 else specs.max_blocks_per_sm
    
    # Thread limit
    max_blocks_threads = specs.max_threads_per_sm // threads_per_block
    
    # Block limit
    max_blocks = min(max_blocks_regs, max_blocks_smem, max_blocks_threads, specs.max_blocks_per_sm)
    max_blocks = max(1, max_blocks)
    
    # Active warps per SM
    warps_per_block = threads_per_block / specs.warp_size
    active_warps = max_blocks * warps_per_block
    
    occupancy = min(1.0, active_warps / specs.max_warps_per_sm)
    return occupancy


def calculate_arithmetic_intensity(features: KernelFeatures) -> float:
    """Calculate arithmetic intensity (FLOP/byte)."""
    total_flops = features.estimated_flops
    total_bytes = features.estimated_bytes_moved
    
    if total_bytes == 0:
        return float('inf')
    
    return total_flops / total_bytes


def calculate_compute_time(features: KernelFeatures, occupancy: float) -> float:
    """Calculate compute-limited time in milliseconds."""
    specs = SPECS
    
    total_flops = features.estimated_flops
    if total_flops == 0:
        return 0.0
    
    # Effective compute throughput considering occupancy
    effective_tflops = specs.peak_fp32_tflops * occupancy * 1e12  # FLOP/s
    
    # Time = FLOPs / throughput
    time_s = total_flops / effective_tflops
    return time_s * 1000  # ms


def calculate_memory_time(features: KernelFeatures, occupancy: float) -> float:
    """Calculate memory-limited time in milliseconds."""
    specs = SPECS
    
    total_bytes = features.estimated_bytes_moved
    if total_bytes == 0:
        return 0.0
    
    # Effective memory bandwidth considering occupancy
    effective_bw = specs.memory_bandwidth_gbps * occupancy * 1e9  # Bytes/s
    
    # Time = Bytes / bandwidth
    time_s = total_bytes / effective_bw
    return time_s * 1000  # ms


def calculate_latency_time(features: KernelFeatures) -> float:
    """Latency floor: the measured irreducible dispatch, nothing else."""
    # gpu-sim-zhq: the old arm was three invented constants (launch 10.0,
    # sync grid*0.01, wave (waves-1)*5.0). The sync term alone predicted
    # 0.363 ms for rows measuring 0.022-0.035 ms -- the whole 772% error.
    # No measurement sources a per-block or per-wave cost, so both terms are
    # DELETED, not re-guessed (bead item 3). What remains is the one measured
    # quantity: 6.6.1's irreducible dispatch floor on this path.
    # `features` is unused and kept for API compatibility.
    _ = features
    return hw_5090.LAUNCH_FLOOR_US / 1000.0  # ms


def roofline_predict(features: KernelFeatures) -> RooflinePrediction:
    """Main roofline prediction function."""
    # Calculate occupancy
    occupancy = calculate_occupancy(features)
    
    # Calculate arithmetic intensity
    ai = calculate_arithmetic_intensity(features)
    
    # Calculate compute and memory times
    compute_time = calculate_compute_time(features, occupancy)
    memory_time = calculate_memory_time(features, occupancy)
    latency_time = calculate_latency_time(features)
    
    # Roofline: max of compute, memory, latency
    predicted_time = max(compute_time, memory_time, latency_time)
    
    # Determine bottleneck
    times = {'compute': compute_time, 'memory': memory_time, 'latency': latency_time}
    bottleneck = max(times, key=times.get)
    
    # Ceilings
    compute_ceiling = SPECS.peak_fp32_tflops * occupancy
    memory_ceiling = SPECS.memory_bandwidth_gbps * occupancy
    
    return RooflinePrediction(
        compute_time_ms=compute_time,
        memory_time_ms=memory_time,
        predicted_time_ms=predicted_time,
        bottleneck=bottleneck,
        achieved_occupancy=occupancy,
        arithmetic_intensity=ai,
        compute_ceiling_tflops=compute_ceiling,
        memory_ceiling_gbps=memory_ceiling,
    )


def roofline_predict_batch(features_list: list[KernelFeatures]) -> list[RooflinePrediction]:
    """Predict for a batch of features."""
    return [roofline_predict(f) for f in features_list]


def evaluate_roofline(
    features_list: list[KernelFeatures],
    actual_times_ms: list[float],
) -> dict:
    """Evaluate roofline model against ground truth."""
    predictions = roofline_predict_batch(features_list)
    
    pred_times = [p.predicted_time_ms for p in predictions]
    
    # Metrics
    import numpy as np
    
    pred = np.array(pred_times)
    actual = np.array(actual_times_ms)
    
    # Avoid division by zero
    mask = actual > 0
    pred = pred[mask]
    actual = actual[mask]
    
    if len(actual) == 0:
        return {'error': 'No valid samples'}
    
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    mse = np.mean((actual - pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - pred))
    
    # Log-scale metrics
    log_actual = np.log(actual + 1e-6)
    log_pred = np.log(pred + 1e-6)
    rmsle = np.sqrt(np.mean((log_actual - log_pred) ** 2))
    
    # R-squared
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Bottleneck breakdown
    bottlenecks = [p.bottleneck for p in predictions]
    bottleneck_counts = {b: bottlenecks.count(b) for b in set(bottlenecks)}
    
    return {
        'mape': mape,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'rmsle': rmsle,
        'r2': r2,
        'bottleneck_distribution': bottleneck_counts,
        'mean_occupancy': np.mean([p.achieved_occupancy for p in predictions]),
        'mean_arithmetic_intensity': np.mean([p.arithmetic_intensity for p in predictions if p.arithmetic_intensity != float('inf')]),
    }


# Simple interface using feature dict
def roofline_from_dict(feature_dict: dict) -> RooflinePrediction:
    """Create prediction from feature dictionary."""
    features = KernelFeatures(**feature_dict)
    return roofline_predict(features)


if __name__ == '__main__':
    # Demo with synthetic features
    test_features = KernelFeatures(
        grid_x=4, grid_y=1, grid_z=1,
        block_x=256, block_y=1, block_z=1,
        shared_mem_bytes=0,
        estimated_registers_per_thread=24,
        num_args=3, num_pointer_args=2, num_scalar_args=1,
        total_arg_bytes=24,
        estimated_global_reads=1024*4,  # 4KB reads
        estimated_global_writes=1024*4,  # 4KB writes
        estimated_shared_reads=0,
        estimated_shared_writes=0,
        estimated_bytes_moved=8192,
        estimated_flops=1024,  # Simple elementwise
        estimated_int_ops=0,
        has_loops=False,
        has_conditionals=True,
        max_loop_depth=0,
        has_syncthreads=False,
        num_syncthreads=0,
        uses_shared_memory=False,
        uses_constant_memory=False,
        uses_vector_types=False,
    )
    
    pred = roofline_predict(test_features)
    print(f"Roofline Prediction:")
    print(f"  Predicted time: {pred.predicted_time_ms:.3f} ms")
    print(f"  Compute time: {pred.compute_time_ms:.3f} ms")
    print(f"  Memory time: {pred.memory_time_ms:.3f} ms")
    print(f"  Latency time: {pred.predicted_time_ms - max(pred.compute_time_ms, pred.memory_time_ms):.3f} ms")
    print(f"  Bottleneck: {pred.bottleneck}")
    print(f"  Occupancy: {pred.achieved_occupancy:.2%}")
    print(f"  Arithmetic intensity: {pred.arithmetic_intensity:.3f} FLOP/byte")
    print(f"  Compute ceiling: {pred.compute_ceiling_tflops:.2f} TFLOP/s")
    print(f"  Memory ceiling: {pred.memory_ceiling_gbps:.2f} GB/s")