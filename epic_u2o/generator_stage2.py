#!/usr/bin/env python3
"""Stage-2 generator: reductions + shared-memory patterns.

Generates tree/warp-shuffle reductions and shared-memory tiling kernels,
including deliberately-inserted bank-conflict patterns for dataset coverage.
"""

import itertools
import json
from dataclasses import dataclass, asdict
from typing import Iterator, Optional
from pathlib import Path
import hashlib



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


# Reduction templates
REDUCE_TEMPLATES = {
    'reduce_sum_block': '''
__global__ void reduce_sum_block(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
''',
    'reduce_sum_warp_shuffle': '''
__global__ void reduce_sum_warp_shuffle(const float* __restrict__ in, float* __restrict__ out, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float val = (i < n) ? in[i] : 0.0f;
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    if (tid % 32 == 0) {
        atomicAdd(out, val);
    }
}
''',
    'reduce_max_block': '''
__global__ void reduce_max_block(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = (i < n) ? in[i] : -__FLT_MAX__;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
''',
    'reduce_min_block': '''
__global__ void reduce_min_block(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = (i < n) ? in[i] : __FLT_MAX__;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
''',
}

# Shared memory patterns
SHARED_MEM_TEMPLATES = {
    'shared_mem_transpose': '''
__global__ void shared_mem_transpose(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float tile[32][32];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.y * 32 + tx;
    int out_y = blockIdx.x * 32 + ty;
    
    if (out_x < rows && out_y < cols) {
        out[out_y * rows + out_x] = tile[tx][ty];
    }
}
''',
    'shared_mem_bank_conflict_1way': '''
__global__ void shared_mem_bank_conflict_1way(float* out, int n) {
    __shared__ float sdata[32 * 33];  // 33 elements per warp = 1-way bank conflict
    
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    
    // Each thread writes to consecutive banks (no conflict)
    sdata[warp * 33 + lane] = lane * 1.0f;
    __syncthreads();
    
    // Each thread reads from next element (1-way conflict within warp)
    out[blockIdx.x * blockDim.x + tid] = sdata[warp * 33 + (lane + 1) % 32];
}
''',
    'shared_mem_bank_conflict_2way': '''
__global__ void shared_mem_bank_conflict_2way(float* out, int n) {
    __shared__ float sdata[32 * 34];  // 34 elements per warp = 2-way bank conflict
    
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    
    sdata[warp * 34 + lane] = lane * 1.0f;
    __syncthreads();
    
    // 2-way bank conflict
    out[blockIdx.x * blockDim.x + tid] = sdata[warp * 34 + (lane + 2) % 32];
}
''',
    'shared_mem_bank_conflict_32way': '''
__global__ void shared_mem_bank_conflict_32way(float* out, int n) {
    __shared__ float sdata[32];
    
    int tid = threadIdx.x;
    int lane = tid % 32;
    
    // All threads in warp write to same bank (32-way conflict)
    if (lane == 0) sdata[0] = 1.0f;
    __syncthreads();
    
    // All threads read from same bank
    out[blockIdx.x * blockDim.x + tid] = sdata[0];
}
''',
    'stencil_2d_5pt': '''
__global__ void stencil_2d_5pt(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float tile[34][34];  // Halo for 5-point stencil
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * 32 + tx;
    int y = blockIdx.y * 32 + ty;
    
    // Load with halo
    if (x < cols && y < rows) {
        tile[ty + 1][tx + 1] = in[y * cols + x];
    }
    
    // Load halo
    if (tx == 0 && x > 0) tile[ty + 1][0] = in[y * cols + x - 1];
    if (tx == 31 && x < cols - 1) tile[ty + 1][33] = in[y * cols + x + 1];
    if (ty == 0 && y > 0) tile[0][tx + 1] = in[(y - 1) * cols + x];
    if (ty == 31 && y < rows - 1) tile[33][tx + 1] = in[(y + 1) * cols + x];
    
    __syncthreads();
    
    // 5-point stencil
    if (x < cols && y < rows) {
        float center = tile[ty + 1][tx + 1];
        float sum = center * 4.0f - tile[ty][tx + 1] - tile[ty + 2][tx + 1] - tile[ty + 1][tx] - tile[ty + 1][tx + 2];
        out[y * cols + x] = sum;
    }
}
''',
    'shared_mem_reduction_tiled': '''
__global__ void shared_mem_reduction_tiled(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x * 2 + tid;
    
    // Load two elements per thread
    float val1 = (i < n) ? in[i] : 0.0f;
    float val2 = (i + blockDim.x < n) ? in[i + blockDim.x] : 0.0f;
    float sum = val1 + val2;
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[bid] = sdata[0];
}
''',
}

# Combined templates
ALL_TEMPLATES = {**REDUCE_TEMPLATES, **SHARED_MEM_TEMPLATES}


def get_template_args(op_name: str) -> list[dict]:
    """Get argument specification for a template."""
    if op_name.startswith('reduce_') or op_name.startswith('shared_mem_reduction'):
        return [
            {'name': 'in', 'type': 'const float*', 'ptr': True},
            {'name': 'out', 'type': 'float*', 'ptr': True},
            {'name': 'n', 'type': 'int', 'ptr': False},
        ]
    elif op_name == 'stencil_2d_5pt':
        return [
            {'name': 'in', 'type': 'const float*', 'ptr': True},
            {'name': 'out', 'type': 'float*', 'ptr': True},
            {'name': 'rows', 'type': 'int', 'ptr': False},
            {'name': 'cols', 'type': 'int', 'ptr': False},
        ]
    elif op_name.startswith('shared_mem_transpose'):
        return [
            {'name': 'in', 'type': 'const float*', 'ptr': True},
            {'name': 'out', 'type': 'float*', 'ptr': True},
            {'name': 'rows', 'type': 'int', 'ptr': False},
            {'name': 'cols', 'type': 'int', 'ptr': False},
        ]
    else:  # bank conflict patterns
        return [
            {'name': 'out', 'type': 'float*', 'ptr': True},
            {'name': 'n', 'type': 'int', 'ptr': False},
        ]


def get_shared_mem(op_name: str, block_size: int) -> int:
    """Calculate shared memory bytes for a template."""
    if op_name in ('reduce_sum_block', 'reduce_max_block', 'reduce_min_block'):
        return block_size * 4
    elif op_name == 'shared_mem_reduction_tiled':
        return block_size * 4
    elif op_name == 'shared_mem_transpose':
        return 32 * 32 * 4
    elif op_name.startswith('shared_mem_bank_conflict'):
        if '1way' in op_name:
            return 32 * 33 * 4
        elif '2way' in op_name:
            return 32 * 34 * 4
        elif '32way' in op_name:
            return 32 * 4
    elif op_name == 'stencil_2d_5pt':
        return 34 * 34 * 4
    return 0


# Sweep configurations
BLOCK_SIZES_REDUCE = [32, 64, 128, 256, 512, 1024]
BLOCK_SIZES_SHARED = [256, 512, 1024]
BLOCK_DIMS_2D = [(16, 16), (16, 32), (32, 16), (32, 32)]
GRID_SIZES_1D = [1, 2, 4, 8, 16, 32, 64, 128, 256]
GRID_SIZES_2D = [(1, 1), (2, 2), (4, 4), (8, 8)]
PROBLEM_SIZES_REDUCE = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
PROBLEM_SIZES_2D = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]


def generate_kernels(
    max_kernels: int = 500,
    ops: Optional[list[str]] = None,
) -> Iterator[KernelSpec]:
    """Generate Stage-2 kernels."""
    ops = ops or list(ALL_TEMPLATES.keys())
    count = 0
    
    for op_name in ops:
        template = ALL_TEMPLATES[op_name]
        args = get_template_args(op_name)
        
        if op_name.startswith('reduce_') or op_name == 'shared_mem_reduction_tiled':
            # 1D reduction sweeps
            for block_size in BLOCK_SIZES_REDUCE:
                if block_size > 1024:
                    continue
                for grid_size in GRID_SIZES_1D:
                    for n in PROBLEM_SIZES_REDUCE:
                        total_threads = grid_size * block_size
                        if total_threads < n // 4:
                            continue
                        
                        shared_mem = get_shared_mem(op_name, block_size)
                        if shared_mem > 48 * 1024:  # 48KB limit
                            continue
                        
                        metadata = {
                            'generator': 'stage2_reduction_shared',
                            'op_type': op_name,
                            'uses_shared_memory': True,
                            'uses_warp_shuffle': 'warp_shuffle' in op_name,
                            'bank_conflict_pattern': None,
                            'arithmetic_intensity': 0.5,
                        }
                        
                        if 'bank_conflict' in op_name:
                            if '1way' in op_name:
                                metadata['bank_conflict_pattern'] = '1way'
                            elif '2way' in op_name:
                                metadata['bank_conflict_pattern'] = '2way'
                            elif '32way' in op_name:
                                metadata['bank_conflict_pattern'] = '32way'
                        
                        yield KernelSpec(
                            kernel_name=_kernel_function_name(template.strip()),
                            source=template.strip(),
                            grid=[grid_size, 1, 1],
                            block=[block_size, 1, 1],
                            shared_mem_bytes=shared_mem,
                            args=args,
                            problem_size=n,
                            op_type=op_name,
                            metadata=metadata,
                        )
                        count += 1
                        if count >= max_kernels:
                            return
        
        elif op_name == 'stencil_2d_5pt':
            for (bx, by) in BLOCK_DIMS_2D:
                for (gx, gy) in GRID_SIZES_2D:
                    for (rows, cols) in PROBLEM_SIZES_2D:
                        shared_mem = get_shared_mem(op_name, bx * by)
                        if shared_mem > 48 * 1024:
                            continue
                        
                        yield KernelSpec(
                            kernel_name=_kernel_function_name(template.strip()),
                            source=template.strip(),
                            grid=[gx, gy, 1],
                            block=[bx, by, 1],
                            shared_mem_bytes=shared_mem,
                            args=args,
                            problem_size=rows * cols,
                            op_type=op_name,
                            metadata={
                                'generator': 'stage2_reduction_shared',
                                'op_type': op_name,
                                'uses_shared_memory': True,
                                'stencil_pattern': '5pt',
                                # u2o.11.3: dims for the driver (rows/cols args
                                # would otherwise synthesize as 1 -> vacuous).
                                'rows': rows,
                                'cols': cols,
                            },
                        )
                        count += 1
                        if count >= max_kernels:
                            return
        
        elif op_name == 'shared_mem_transpose':
            for (bx, by) in [(32, 32)]:
                for (gx, gy) in GRID_SIZES_2D:
                    for (rows, cols) in PROBLEM_SIZES_2D:
                        if rows % 32 != 0 or cols % 32 != 0:
                            continue
                        
                        shared_mem = get_shared_mem(op_name, bx * by)
                        
                        yield KernelSpec(
                            kernel_name=_kernel_function_name(template.strip()),
                            source=template.strip(),
                            grid=[gx, gy, 1],
                            block=[bx, by, 1],
                            shared_mem_bytes=shared_mem,
                            args=args,
                            problem_size=rows * cols,
                            op_type=op_name,
                            metadata={
                                'generator': 'stage2_reduction_shared',
                                'op_type': op_name,
                                'uses_shared_memory': True,
                                'access_pattern': 'transpose',
                                # u2o.11.3: dims for the driver (see stencil site).
                                'rows': rows,
                                'cols': cols,
                            },
                        )
                        count += 1
                        if count >= max_kernels:
                            return
        
        elif op_name.startswith('shared_mem_bank_conflict'):
            for block_size in BLOCK_SIZES_SHARED:
                for grid_size in GRID_SIZES_1D:
                    shared_mem = get_shared_mem(op_name, block_size)
                    if shared_mem > 48 * 1024:
                        continue
                    
                    yield KernelSpec(
                        kernel_name=_kernel_function_name(template.strip()),
                        source=template.strip(),
                        grid=[grid_size, 1, 1],
                        block=[block_size, 1, 1],
                        shared_mem_bytes=shared_mem,
                        args=args,
                        problem_size=block_size * grid_size,
                        op_type=op_name,
                        metadata={
                            'generator': 'stage2_reduction_shared',
                            'op_type': op_name,
                            'uses_shared_memory': True,
                            'bank_conflict_pattern': op_name.split('_')[-1],
                        },
                    )
                    count += 1
                    if count >= max_kernels:
                        return


def generate_dataset(
    output_dir: str | Path,
    max_kernels: int = 500,
    split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict:
    """Generate full dataset with train/val/test splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    kernels = list(generate_kernels(max_kernels=max_kernels))
    
    # Split by kernel hash to prevent template leakage
    import random
    random.seed(42)
    random.shuffle(kernels)
    
    n = len(kernels)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    
    splits = {
        'train': kernels[:n_train],
        'val': kernels[n_train:n_train + n_val],
        'test': kernels[n_train + n_val:],
    }
    
    # Save each split
    for split_name, split_kernels in splits.items():
        split_file = output_dir / f'{split_name}.jsonl'
        with open(split_file, 'w') as f:
            for kernel in split_kernels:
                f.write(json.dumps(kernel.to_dict()) + '\n')
    
    # Save metadata
    metadata = {
        'total_kernels': len(kernels),
        'splits': {k: len(v) for k, v in splits.items()},
        'templates': list(ALL_TEMPLATES.keys()),
        'split_ratio': split_ratio,
        'generator_version': 'stage2_v1',
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {len(kernels)} kernels:")
    for split_name, split_kernels in splits.items():
        print(f"  {split_name}: {len(split_kernels)}")
    
    return metadata


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = 'runs/stage2_dataset'
    
    if len(sys.argv) > 2:
        max_kernels = int(sys.argv[2])
    else:
        max_kernels = 500
    
    generate_dataset(output_dir, max_kernels=max_kernels)
