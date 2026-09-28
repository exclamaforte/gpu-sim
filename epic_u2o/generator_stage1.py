#!/usr/bin/env python3
"""Stage-1 generator: parameterized elementwise/map kernels with grid/block sweep.

Generates a family of elementwise kernels (scale, add, saxpy, relu, etc.)
with swept grid/block dimensions and data sizes for pipeline validation.
"""

import itertools
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Iterator
from pathlib import Path


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
    
    def kernel_hash(self) -> str:
        """SHA-256 hash of kernel source for deduplication."""
        return hashlib.sha256(self.source.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return asdict(self)


# Elementwise operation templates
OP_TEMPLATES = {
    'scale': '''
__global__ void scale_kernel(const float* __restrict__ in, float* __restrict__ out, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * alpha;
}
''',
    'add': '''
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
''',
    'saxpy': '''
__global__ void saxpy_kernel(float alpha, const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + y[i];
}
''',
    'relu': '''
__global__ void relu_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(in[i], 0.0f);
}
''',
    'mul': '''
__global__ void mul_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}
''',
    'div': '''
__global__ void div_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}
''',
    'copy': '''
__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}
''',
    'triad': '''
__global__ void triad_kernel(float alpha, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = alpha * a[i] + b[i];
}
''',
}


# Grid/block sweep configurations
GRID_SWEEPS = [
    [1, 1, 1], [2, 1, 1], [4, 1, 1], [8, 1, 1], [16, 1, 1],
    [32, 1, 1], [64, 1, 1], [128, 1, 1], [256, 1, 1],
    [1, 2, 1], [1, 4, 1], [1, 8, 1],
    [2, 2, 1], [4, 4, 1], [8, 8, 1], [16, 16, 1],
]

BLOCK_SWEEPS = [
    [32, 1, 1], [64, 1, 1], [128, 1, 1], [256, 1, 1], [512, 1, 1], [1024, 1, 1],
    [16, 16, 1], [16, 32, 1], [32, 16, 1], [32, 32, 1],
]

PROBLEM_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


def generate_kernels(
    ops: list[str] | None = None,
    grid_sweeps: list[list[int]] | None = None,
    block_sweeps: list[list[int]] | None = None,
    problem_sizes: list[int] | None = None,
    max_kernels: int | None = None,
) -> Iterator[KernelSpec]:
    """Generate parameterized elementwise kernels."""
    ops = ops or list(OP_TEMPLATES.keys())
    grid_sweeps = grid_sweeps or GRID_SWEEPS
    block_sweeps = block_sweeps or BLOCK_SWEEPS
    problem_sizes = problem_sizes or PROBLEM_SIZES
    
    count = 0
    for op_name in ops:
        template = OP_TEMPLATES[op_name]
        kernel_name = f"{op_name}_kernel"
        
        # Argument specification based on op type
        if op_name == 'scale':
            args = [
                {'name': 'in', 'type': 'const float*', 'ptr': True},
                {'name': 'out', 'type': 'float*', 'ptr': True},
                {'name': 'alpha', 'type': 'float', 'ptr': False},
                {'name': 'n', 'type': 'int', 'ptr': False},
            ]
        elif op_name == 'add':
            args = [
                {'name': 'a', 'type': 'const float*', 'ptr': True},
                {'name': 'b', 'type': 'const float*', 'ptr': True},
                {'name': 'out', 'type': 'float*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False},
            ]
        elif op_name == 'saxpy':
            args = [
                {'name': 'alpha', 'type': 'float', 'ptr': False},
                {'name': 'x', 'type': 'const float*', 'ptr': True},
                {'name': 'y', 'type': 'float*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False},
            ]
        elif op_name == 'relu':
            args = [
                {'name': 'in', 'type': 'const float*', 'ptr': True},
                {'name': 'out', 'type': 'float*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False},
            ]
        elif op_name == 'mul':
            args = [
                {'name': 'a', 'type': 'const float*', 'ptr': True},
                {'name': 'b', 'type': 'const float*', 'ptr': True},
                {'name': 'out', 'type': 'float*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False},
            ]
        elif op_name == 'div':
            args = [
                {'name': 'a', 'type': 'const float*', 'ptr': True},
                {'name': 'b', 'type': 'const float*', 'ptr': True},
                {'name': 'out', 'type': 'float*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False},
            ]
        elif op_name == 'copy':
            args = [
                {'name': 'in', 'type': 'const float*', 'ptr': True},
                {'name': 'out', 'type': 'float*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False},
            ]
        elif op_name == 'triad':
            args = [
                {'name': 'alpha', 'type': 'float', 'ptr': False},
                {'name': 'a', 'type': 'const float*', 'ptr': True},
                {'name': 'b', 'type': 'const float*', 'ptr': True},
                {'name': 'out', 'type': 'float*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False},
            ]
        else:
            args = []
        
        for grid in grid_sweeps:
            for block in block_sweeps:
                # Skip invalid combinations
                threads_per_block = block[0] * block[1] * block[2]
                if threads_per_block > 1024:
                    continue
                if threads_per_block < 32:
                    continue
                
                for n in problem_sizes:
                    # Ensure grid covers problem size
                    total_threads = grid[0] * grid[1] * grid[2] * threads_per_block
                    if total_threads < n // 4:  # Allow some underutilization
                        continue
                    
                    metadata = {
                        'generator': 'stage1_elementwise',
                        'op_type': op_name,
                        'bytes_per_thread': 4,  # float
                        'arithmetic_intensity': 1.0,  # FLOP/byte
                        'expected_runtime_model': 'memory_bound',
                    }
                    
                    yield KernelSpec(
                        kernel_name=kernel_name,
                        source=template.strip(),
                        grid=grid,
                        block=block,
                        shared_mem_bytes=0,
                        args=args,
                        problem_size=n,
                        op_type=op_name,
                        metadata=metadata,
                    )
                    
                    count += 1
                    if max_kernels and count >= max_kernels:
                        return


def generate_dataset(
    output_dir: str | Path,
    max_kernels: int = 1000,
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
        'ops': list(OP_TEMPLATES.keys()),
        'grid_sweeps': len(GRID_SWEEPS),
        'block_sweeps': len(BLOCK_SWEEPS),
        'problem_sizes': len(PROBLEM_SIZES),
        'split_ratio': split_ratio,
        'generator_version': 'stage1_v1',
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
        output_dir = 'runs/stage1_dataset'
    
    if len(sys.argv) > 2:
        max_kernels = int(sys.argv[2])
    else:
        max_kernels = 1000
    
    generate_dataset(output_dir, max_kernels=max_kernels)