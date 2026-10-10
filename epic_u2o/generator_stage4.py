#!/usr/bin/env python3
"""Stage-4 generator: tiled GEMM and tensor-core (WMMA/MMA) kernel family.

Compute-bound family with tile-size/warp-tiling variants, including
tensor-core paths, covering the compute-bound end of the roofline where
occupancy and register pressure dominate over memory pattern.
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Iterator, Optional
from pathlib import Path



from epic_u2o.kernel_names import declared_kernel_names


def _kernel_function_name(source: str) -> str:
    """The __global__ function actually declared in `source` (gpu-sim-9ti).

    This generator used to set kernel_name=f"{op}_kernel" while its templates
    declare `__global__ void {op}(...)`, so every synthesized driver called a
    kernel that did not exist and failed to COMPILE. Deriving the name from
    the source makes the two impossible to disagree.

    The implementation is imported UNCONDITIONALLY from epic_u2o.kernel_names
    rather than duplicated behind an ImportError fallback. The earlier
    fallback re-implemented the pattern in a weaker form (it missed
    extern "C", __device__-marked and non-void returns), which is a silent
    semantic difference in a defensive path that will drift -- and
    kernel_driver.py hard-requires epic_u2o.kernel_names anyway, so the
    standalone case it protected is close to vestigial (reviewer, gpu-sim-9ti).
    """
    names = declared_kernel_names(source)
    if not names:
        raise ValueError("kernel template declares no __global__ function; "
                         "a spec built from it could never be launched")
    return names[0]


@dataclass
class KernelSpec:
    """Specification for a generated kernel."""
    kernel_name: str
    source: str
    grid: list[int]
    block: list[int]
    shared_mem_bytes: int
    args: list[dict]
    problem_size: int
    op_type: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


GEMM_TEMPLATES = {
    'gemm_naive': '''
__global__ void gemm_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
''',
    'gemm_tiled': '''
__global__ void gemm_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        if (row < M && t * TILE + threadIdx.x < K) As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else As[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && t * TILE + threadIdx.y < K) Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
''',
    'gemm_wmma_fp16': '''
#include <mma.h>
using namespace nvcuda;
__global__ void gemm_wmma_fp16(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int laneId = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    // One 16x16x16 WMMA tile per warp (simplified placeholder layout)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    for (int k = 0; k < K; k += 16) {
        int aRow = warpM * 16 + laneId / 4;
        int aCol = k + (laneId % 4) * 4;
        int bRow = k + laneId / 4;
        int bCol = warpN * 16 + (laneId % 4) * 4;
        wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}
''',
}

TILE_SIZES = [16, 32]
PROBLEM_SIZES_GEMM = [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 128)]
BLOCK_DIMS_GEMM = [(16, 16)]

GEMM_ARGS_FP32 = [
    {'name': 'A', 'type': 'const float*', 'ptr': True},
    {'name': 'B', 'type': 'const float*', 'ptr': True},
    {'name': 'C', 'type': 'float*', 'ptr': True},
    {'name': 'M', 'type': 'int', 'ptr': False},
    {'name': 'N', 'type': 'int', 'ptr': False},
    {'name': 'K', 'type': 'int', 'ptr': False},
]

GEMM_ARGS_FP16 = [
    {'name': 'A', 'type': 'const half*', 'ptr': True},
    {'name': 'B', 'type': 'const half*', 'ptr': True},
    {'name': 'C', 'type': 'float*', 'ptr': True},
    {'name': 'M', 'type': 'int', 'ptr': False},
    {'name': 'N', 'type': 'int', 'ptr': False},
    {'name': 'K', 'type': 'int', 'ptr': False},
]


def generate_kernels(max_kernels: int = 300, ops: Optional[list] = None) -> Iterator[KernelSpec]:
    ops = ops or list(GEMM_TEMPLATES.keys())
    count = 0
    for op in ops:
        tmpl = GEMM_TEMPLATES[op]
        is_tiled = op == 'gemm_tiled'
        is_wmma = op == 'gemm_wmma_fp16'
        tiles = TILE_SIZES if is_tiled else [0]
        for tile in tiles:
            src = tmpl.replace('TILE', str(tile)) if is_tiled else tmpl
            for (M, N, K) in PROBLEM_SIZES_GEMM:
                for (bx, by) in BLOCK_DIMS_GEMM:
                    gx = (N + bx - 1) // bx
                    gy = (M + by - 1) // by
                    if is_tiled:
                        # Grid must align to TILE for the simplified tiled kernel
                        gx = (N + tile - 1) // tile
                        gy = (M + tile - 1) // tile
                    smem = 2 * tile * tile * 4 if is_tiled else 0
                    if smem > 48 * 1024:
                        continue
                    yield KernelSpec(
                        kernel_name=_kernel_function_name(src.strip()),
                        source=src.strip(),
                        grid=[gx, gy, 1],
                        block=[bx, by, 1],
                        shared_mem_bytes=smem,
                        args=GEMM_ARGS_FP16 if is_wmma else GEMM_ARGS_FP32,
                        problem_size=M * N * K,
                        op_type=op,
                        metadata={
                            'generator': 'stage4_gemm',
                            'op_type': op,
                            'tile': tile,
                            'uses_shared_memory': is_tiled,
                            'uses_tensor_cores': is_wmma,
                            'compute_bound': True,
                            'M': M, 'N': N, 'K': K,
                        },
                    )
                    count += 1
                    if count >= max_kernels:
                        return


def generate_dataset(output_dir, max_kernels: int = 300,
                     split_ratio=(0.7, 0.15, 0.15)) -> dict:
    import random
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels = list(generate_kernels(max_kernels=max_kernels))
    random.seed(42)
    random.shuffle(kernels)
    n = len(kernels)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    splits = {'train': kernels[:n_train], 'val': kernels[n_train:n_train + n_val],
              'test': kernels[n_train + n_val:]}
    for name, ks in splits.items():
        with open(output_dir / f'{name}.jsonl', 'w') as f:
            for k in ks:
                f.write(json.dumps(k.to_dict()) + '\n')
    meta = {'total_kernels': len(kernels),
            'splits': {k: len(v) for k, v in splits.items()},
            'templates': list(GEMM_TEMPLATES.keys()),
            'generator_version': 'stage4_v1'}
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Generated {len(kernels)} kernels:")
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")
    return meta


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'runs/stage4_dataset'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    generate_dataset(out, max_kernels=n)
