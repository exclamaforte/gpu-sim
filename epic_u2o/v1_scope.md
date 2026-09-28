# Epic u2o — v1 Scope: Kernel / Program Space & Rejection Rules

Status: IN PROGRESS (bead u2o.5.1)
Parent epic: gpu-sim-u2o (CUDA runtime prediction: learned performance model for RTX 5090)

---

## 1. v1 Prediction Scope

Single CUDA kernel launch runtime (not whole-program wall-clock). The model predicts `execution_time_ms` given:
- CUDA source code (or PTX) of the kernel
- Launch configuration (`gridDim`, `blockDim`, `sharedMem`, `stream`)
- Device spec (RTX 5090 fixed for v1)

---

## 2. Allowed Constructs (v1 Kernel Space)

The following CUDA constructs are **allowed** in v1 kernels:

### Kernel Definition
- Single `__global__` kernel per compilation unit
- Function parameters: pointers to global memory (`__restrict__` preferred), scalar values (int, float, double)
- Return type: `void`

### Memory Access
- Coalesced global memory reads/writes (linear, strided, or 2D patterns)
- Shared memory (`__shared__`) for intra-block communication
- Constant memory (`__constant__`) for read-only data
- Register usage for local variables

### Execution Configuration
- 1D, 2D, or 3D grid and block dimensions
- Dynamic shared memory allocation at launch (`<<<grid, block, sharedMemBytes>>>`)
- Single stream (default stream or explicit non-default stream)
- Synchronous execution (model predicts kernel runtime only)

### Control Flow
- Simple loops with compile-time or launch-time bounds
- Uniform conditional branches (warp-uniform or block-uniform)
- `if/else` on threadIdx/blockIdx for boundary checks

### Math Operations
- Standard arithmetic (+, -, *, /, %)
- Transcendental functions (sin, cos, exp, log, sqrt, rsqrt)
- Integer bitwise operations
- Vector types (float2, float4, int2, int4, etc.)

### Synchronization
- `__syncthreads()` for block-level synchronization
- No warp-level primitives (shuffle, vote, match) in v1

---

## 3. Disallowed Constructs (v1 Rejection Rules)

The following are **rejected** and will fail pre-flight validation:

| Category | Disallowed Items |
|----------|------------------|
| **Multi-kernel / Multi-stream** | Multiple `__global__` kernels, `cudaStream_t` arrays, `cudaLaunchKernel`, `cudaGraph` |
| **Dynamic Parallelism** | `__global__` kernels launching other kernels (`<<<>>>` inside device code) |
| **Cooperative Groups** | `cooperative_groups::thread_block`, `grid_group`, `thread_block_tile`, `tiled_partition`, `sync()` |
| **Host Callbacks** | `cudaStreamAddCallback`, `cudaHostFn_t` |
| **Cluster / Multi-GPU** | `cudaLaunchCooperativeKernel`, `cudaLaunchCooperativeKernelMultiDevice` |
| **Persistent Thread Blocks** | Infinite loops with `__syncthreads()` waiting for host signals |
| **Texture / Surface Memory** | `texture<>`, `surface<>`, `cudaBindTexture`, `cudaBindSurface` |
| **Unified Memory** | `cudaMallocManaged`, `__managed__`, `cudaMemAdvise` |
| **Peer Access / IPC** | `cudaDeviceEnablePeerAccess`, `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle` |
| **Profiler / Debug APIs** | `cupti`, `nvtx`, `cudaProfilerStart/Stop`, `cudaGdbAttach` |
| **PTX / Inline Assembly** | `asm()`, `.ptx` files as input (v1 uses CUDA source only) |
| **Templates / C++ Features** | Template kernels, virtual functions, exceptions, RTTI, STL in device code |
| **Variable-Length Arrays** | Dynamic stack arrays in device code |
| **Recursion** | Recursive `__device__` or `__global__` functions |

---

## 4. Max Resource Bounds (RTX 5090 / CC 10.0 / v1 Limits)

| Resource | v1 Limit | Notes |
|----------|----------|-------|
| **Max threads per block** | 1024 | Hardware limit |
| **Max blockDim.x** | 1024 | |
| **Max blockDim.y** | 1024 | |
| **Max blockDim.z** | 64 | |
| **Max gridDim.x** | 2^31 - 1 | |
| **Max gridDim.y** | 65535 | |
| **Max gridDim.z** | 65535 | |
| **Max shared memory per block** | 228 KB (233472 bytes) | RTX 5090: 228 KB per SM; v1 caps at 128 KB for safety |
| **Max registers per thread** | 255 | Hardware limit; v1 warns at 128 |
| **Max blocks per SM** | 32 | |
| **Max warps per SM** | 64 | |
| **Max resident threads per SM** | 2048 | |
| **VRAM pool (fakegpu)** | 24 GB | `FAKEGPU_DEVICE_MEM_MB=24576` |
| **Runtime ceiling per kernel** | 5 seconds | Watchdog timeout; kernels exceeding this are killed |
| **Kernel launch overhead** | Not modeled | v1 predicts only `execution_time_ms` from `cudaEventElapsedTime()` |

---

## 5. Rejection Rules (Pre-Flight Structural Validator using fakegpu)

Before any generated program reaches the real GPU, it must pass `gpu-sim` structural checks (no ground-truth timing):

1. **Compilation**: Kernel must compile with `nvcc` (syntax / link errors rejected)
2. **Launch Config Limits**: Launch config must respect device limits (max threads/block, max grid dims, max shared mem)
3. **VRAM Check**: Memory access patterns must not exceed device VRAM (fake pool size check via `FAKEGPU_DEVICE_MEM_MB`)
4. **Execution Success**: Kernel must complete without `LD_PRELOAD` interposer errors (`cudaSuccess` required for all calls)
5. **Timeout**: Kernel must complete within 5-second watchdog window
6. **No Disallowed Features**: Static analysis rejects any disallowed constructs from Section 3

---

## 6. Program Space (v1 Generators)

- **Stage-1** (`gpu-sim-u2o.7.1`): Parameterized elementwise / map kernels with grid/block sweep
- **Stage-2** (`gpu-sim-u2o.7.2`): Reductions + shared-memory patterns
- **Stage-3** (future): Tensor cores, warp specialization, persistent kernels

---

## 7. Tracking Protocol

Every data-generation / measurement run creates a bead (`bd create`) so runs are never lost.

---

## 8. Acceptance Criteria Checklist

- [x] Written spec doc lists allowed constructs (Section 2)
- [x] Written spec doc lists disallowed constructs (Section 3)
- [x] Max resource bounds documented (Section 4: grid size, shared mem, registers, runtime ceiling)
- [ ] Spec reviewed and approved by team
- [ ] Pre-flight validator implemented against these rules (separate bead: gpu-sim-u2o.7.1)

(End of file)