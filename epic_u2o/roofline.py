#!/usr/bin/env python3
"""Analytic roofline model baseline for CUDA runtime prediction.

Classic max(compute_time, memory_time) roofline estimate from RTX 5090's
peak FLOPs/bandwidth spec sheet and the static feature vector.
Zero learned parameters - the floor every learned model must beat.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class RTX5090Specs:
    """RTX 5090 (Blackwell) specifications."""
    # Compute
    peak_fp32_tflops: float = 82.6  # TFLOP/s
    peak_fp16_tflops: float = 165.2  # TFLOP/s (with sparsity)
    peak_int32_tiops: float = 82.6  # TIOP/s
    
    # Memory
    memory_bandwidth_gbps: float = 1792  # GB/s
    l2_cache_mb: float = 64  # MB
    l1_cache_kb_per_sm: float = 128  # KB per SM
    
    # Architecture
    num_sms: int = 170
    warp_size: int = 32
    max_threads_per_sm: int = 2048
    max_blocks_per_sm: int = 32
    max_warps_per_sm: int = 64
    max_registers_per_thread: int = 255
    shared_mem_per_sm_kb: float = 228  # KB
    
    # Clock
    base_clock_mhz: float = 2235
    boost_clock_mhz: float = 2520


# Global specs instance
SPECS = RTX5090Specs()


@dataclass
class KernelFeatures:
    """Static features extracted from kernel (matches features.py)."""
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    shared_mem_bytes: int
    estimated_registers_per_thread: int
    num_args: int
    num_pointer_args: int
    num_scalar_args: int
    total_arg_bytes: int
    estimated_global_reads: int
    estimated_global_writes: int
    estimated_shared_reads: int
    estimated_shared_writes: int
    estimated_bytes_moved: int
    estimated_flops: int
    estimated_int_ops: int
    has_loops: bool
    has_conditionals: bool
    max_loop_depth: int
    has_syncthreads: bool
    num_syncthreads: int
    uses_shared_memory: bool
    uses_constant_memory: bool
    uses_vector_types: bool


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
    max_blocks_regs = (specs.max_registers_per_thread * specs.max_threads_per_sm) // regs_per_block if regs_per_block > 0 else specs.max_blocks_per_sm
    
    # Shared memory limit
    shared_mem_per_block = features.shared_mem_bytes
    max_blocks_smem = (specs.shared_mem_per_sm_kb * 1024) // shared_mem_per_block if shared_mem_per_block > 0 else specs.max_blocks_per_sm
    
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
    """Calculate latency-limited time (launch overhead, synchronization)."""
    # Rough estimate: launch overhead + grid synchronization
    grid_size = features.grid_x * features.grid_y * features.grid_z
    
    # Base launch overhead ~5-10 microseconds
    launch_overhead_us = 10.0
    
    # Grid synchronization scales with grid size
    sync_overhead_us = grid_size * 0.01  # ~0.01 us per block
    
    # Wave count effect
    specs = SPECS
    max_blocks_concurrent = specs.num_sms * specs.max_blocks_per_sm
    waves = max(1, grid_size / max_blocks_concurrent)
    wave_overhead_us = (waves - 1) * 5.0  # ~5 us per additional wave
    
    total_us = launch_overhead_us + sync_overhead_us + wave_overhead_us
    return total_us / 1000.0  # ms


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