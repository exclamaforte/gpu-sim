# Ground-Truth Measurement Spec (v1)

Parent: gpu-sim-u2o.5.4
Reference: real RTX 5090 GPU measurements.

---

## 1. Measurement Protocol

- **Timing method**: `cudaEventElapsedTime()` for kernel execution; `std::chrono::high_resolution_clock` for host-side overhead.
- **Repetition count**: 30 runs per kernel (to model noise); discard outliers beyond 2σ.
- **Clock-locking**: Lock GPU clock to base frequency (`nvidia-smi -lgc`) before measurement to reduce thermal variance.
- **Warm-up**: 5 warm-up runs before recording.

---

## 2. Ground-Truth Dataset Format

Each row = one measurement event:
- `kernel_hash`: SHA-256 of kernel source
- `launch_config`: JSON `{grid:[x,y,z], block:[x,y,z], shared_mem_bytes}`
- `execution_time_ms`: mean of 30 runs (after outlier removal)
- `noise_std_ms`: standard deviation of 30 runs
- `device_prop_snapshot`: `cudaGetDeviceProperties()` output at measurement time

---

## 3. Safety (Safe Execution Epic u2o.6)

Every measurement uses process isolation (`subprocess` with timeout/watchdog), resource limits (`ulimit -v` for VRAM cap), and recovery procedure (GPU reset on hang).
