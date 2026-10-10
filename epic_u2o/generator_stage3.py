#!/usr/bin/env python3
"""Stage-3 generator: coalesced vs strided vs random memory-access families.

Parameterizes access stride/pattern explicitly so the dataset spans the full
memory-bound spectrum (coalesced through pathologically strided/random).
"""

import json
import random
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


# Memory access pattern templates
MEMORY_TEMPLATES = {
    'coalesced_read': '''
__global__ void coalesced_read(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}
''',
    'coalesced_write': '''
__global__ void coalesced_write(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}
''',
    'strided_read': '''
__global__ void strided_read(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * stride < n) out[i] = in[i * stride];
}
''',
    'strided_write': '''
__global__ void strided_write(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i * stride] = in[i];
}
''',
    'strided_read_write': '''
__global__ void strided_read_write(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * stride < n) out[i * stride] = in[i * stride];
}
''',
    'segmented_read': '''
__global__ void segmented_read(const float* __restrict__ in, float* __restrict__ out, int n, int segment_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int segment = i / segment_size;
        int offset = i % segment_size;
        out[i] = in[segment * n + offset];
    }
}
''',
    'interleaved_read': '''
__global__ void interleaved_read(const float* __restrict__ in, float* __restrict__ out, int n, int interleave) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int source_idx = (i % interleave) * (n / interleave) + (i / interleave);
        out[i] = in[source_idx];
    }
}
''',
    'gather_read': '''
__global__ void gather_read(const float* __restrict__ in, float* __restrict__ out, const int* __restrict__ indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[indices[i]];
}
''',
    'scatter_write': '''
__global__ void scatter_write(const float* __restrict__ in, float* __restrict__ out, const int* __restrict__ indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[indices[i]] = in[i];
}
''',
    'random_access': '''
__global__ void random_access(const float* __restrict__ in, float* __restrict__ out, const unsigned int* __restrict__ seeds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int seed = seeds[i];
        // Simple LCG for pseudo-random access
        seed = seed * 1664525u + 1013904223u;
        int idx = seed % n;
        out[i] = in[idx];
    }
}
''',
    'transpose_2d': '''
__global__ void transpose_2d(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        out[x * rows + y] = in[y * cols + x];
    }
}
''',
    'diagonal_access': '''
__global__ void diagonal_access(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows && x == y) {
        out[y * cols + x] = in[y * cols + x] * 2.0f;
    }
}
''',
    'block_cyclic': '''
__global__ void block_cyclic(const float* __restrict__ in, float* __restrict__ out, int n, int block_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int cyclic_idx = (i % block_size) * (n / block_size) + (i / block_size);
        out[i] = in[cyclic_idx];
    }
}
''',
}

# Stride configurations
STRIDES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
INTERLEAVES = [2, 4, 8, 16, 32]
SEGMENT_SIZES = [256, 512, 1024, 4096, 16384]
BLOCK_SIZES_CYCLIC = [64, 128, 256, 512, 1024]

# Block/grid sweeps
BLOCK_1D = [64, 128, 256, 512, 1024]
BLOCK_2D = [(16, 16), (16, 32), (32, 16), (32, 32)]
GRID_1D = [1, 2, 4, 8, 16, 32, 64, 128, 256]
GRID_2D = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
PROBLEM_SIZES_1D = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
PROBLEM_SIZES_2D = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]


def get_args(op_name: str) -> list[dict]:
    """Get argument spec for an op."""
    base_args = [
        {'name': 'in', 'type': 'const float*', 'ptr': True},
        {'name': 'out', 'type': 'float*', 'ptr': True},
    ]
    
    if 'strided' in op_name:
        base_args.extend([
            {'name': 'n', 'type': 'int', 'ptr': False},
            {'name': 'stride', 'type': 'int', 'ptr': False},
        ])
    elif 'segmented' in op_name:
        base_args.extend([
            {'name': 'n', 'type': 'int', 'ptr': False},
            {'name': 'segment_size', 'type': 'int', 'ptr': False},
        ])
    elif 'interleaved' in op_name:
        base_args.extend([
            {'name': 'n', 'type': 'int', 'ptr': False},
            {'name': 'interleave', 'type': 'int', 'ptr': False},
        ])
    elif 'gather' in op_name or 'scatter' in op_name:
        base_args.extend([
            {'name': 'indices', 'type': 'const int*', 'ptr': True},
            {'name': 'n', 'type': 'int', 'ptr': False},
        ])
    elif 'random' in op_name:
        base_args.extend([
            {'name': 'seeds', 'type': 'const unsigned int*', 'ptr': True},
            {'name': 'n', 'type': 'int', 'ptr': False},
        ])
    elif '2d' in op_name or 'diagonal' in op_name:
        base_args.extend([
            {'name': 'rows', 'type': 'int', 'ptr': False},
            {'name': 'cols', 'type': 'int', 'ptr': False},
        ])
    elif 'cyclic' in op_name:
        base_args.extend([
            {'name': 'n', 'type': 'int', 'ptr': False},
            {'name': 'block_size', 'type': 'int', 'ptr': False},
        ])
    else:  # coalesced
        base_args.append({'name': 'n', 'type': 'int', 'ptr': False})
    
    return base_args


def generate_kernels(
    max_kernels: int = 500,
    ops: Optional[list[str]] = None,
) -> Iterator[KernelSpec]:
    """Generate Stage-3 memory access pattern kernels."""
    ops = ops or list(MEMORY_TEMPLATES.keys())
    count = 0
    
    for op_name in ops:
        template = MEMORY_TEMPLATES[op_name]
        args = get_args(op_name)
        
        if op_name in ('coalesced_read', 'coalesced_write'):
            for block in BLOCK_1D:
                for grid in GRID_1D:
                    for n in PROBLEM_SIZES_1D:
                        total_threads = grid * block
                        if total_threads < n // 4:
                            continue
                        
                        yield KernelSpec(
                            kernel_name=_kernel_function_name(template.strip()),
                            source=template.strip(),
                            grid=[grid, 1, 1],
                            block=[block, 1, 1],
                            shared_mem_bytes=0,
                            args=args,
                            problem_size=n,
                            op_type=op_name,
                            metadata={
                                'generator': 'stage3_memory_access',
                                'access_pattern': 'coalesced',
                                'coalesced': True,
                                'stride': 1,
                            },
                        )
                        count += 1
                        if count >= max_kernels:
                            return
        
        elif 'strided' in op_name:
            for stride in STRIDES:
                for block in BLOCK_1D:
                    for grid in GRID_1D:
                        for n in PROBLEM_SIZES_1D:
                            if stride * n > 1048576:  # Limit effective size
                                continue
                            total_threads = grid * block
                            if total_threads < (n // stride) // 4:
                                continue
                            
                            yield KernelSpec(
                                kernel_name=_kernel_function_name(template.strip()),
                                source=template.strip(),
                                grid=[grid, 1, 1],
                                block=[block, 1, 1],
                                shared_mem_bytes=0,
                                args=args,
                                problem_size=n,
                                op_type=op_name,
                                metadata={
                                    'generator': 'stage3_memory_access',
                                    'access_pattern': 'strided',
                                    'coalesced': stride == 1,
                                    'stride': stride,
                                },
                            )
                            count += 1
                            if count >= max_kernels:
                                return
        
        elif op_name == 'segmented_read':
            for segment in SEGMENT_SIZES:
                for block in BLOCK_1D:
                    for grid in GRID_1D:
                        for n in PROBLEM_SIZES_1D:
                            if segment > n:
                                continue
                            total_threads = grid * block
                            if total_threads < n // 4:
                                continue
                            
                            yield KernelSpec(
                                kernel_name=_kernel_function_name(template.strip()),
                                source=template.strip(),
                                grid=[grid, 1, 1],
                                block=[block, 1, 1],
                                shared_mem_bytes=0,
                                args=args,
                                problem_size=n,
                                op_type=op_name,
                                metadata={
                                    'generator': 'stage3_memory_access',
                                    'access_pattern': 'segmented',
                                    'coalesced': False,
                                    'segment_size': segment,
                                },
                            )
                            count += 1
                            if count >= max_kernels:
                                return
        
        elif op_name == 'interleaved_read':
            for interleave in INTERLEAVES:
                for block in BLOCK_1D:
                    for grid in GRID_1D:
                        for n in PROBLEM_SIZES_1D:
                            if n % interleave != 0:
                                continue
                            total_threads = grid * block
                            if total_threads < n // 4:
                                continue
                            
                            yield KernelSpec(
                                kernel_name=_kernel_function_name(template.strip()),
                                source=template.strip(),
                                grid=[grid, 1, 1],
                                block=[block, 1, 1],
                                shared_mem_bytes=0,
                                args=args,
                                problem_size=n,
                                op_type=op_name,
                                metadata={
                                    'generator': 'stage3_memory_access',
                                    'access_pattern': 'interleaved',
                                    'coalesced': False,
                                    'interleave': interleave,
                                },
                            )
                            count += 1
                            if count >= max_kernels:
                                return
        
        elif op_name in ('gather_read', 'scatter_write'):
            for block in BLOCK_1D:
                for grid in GRID_1D:
                    for n in PROBLEM_SIZES_1D:
                        total_threads = grid * block
                        if total_threads < n // 4:
                            continue
                        
                        yield KernelSpec(
                            kernel_name=_kernel_function_name(template.strip()),
                            source=template.strip(),
                            grid=[grid, 1, 1],
                            block=[block, 1, 1],
                            shared_mem_bytes=0,
                            args=args,
                            problem_size=n,
                            op_type=op_name,
                            metadata={
                                'generator': 'stage3_memory_access',
                                'access_pattern': 'gather' if 'gather' in op_name else 'scatter',
                                'coalesced': False,
                                'indirect': True,
                            },
                        )
                        count += 1
                        if count >= max_kernels:
                            return
        
        elif op_name == 'random_access':
            for block in BLOCK_1D:
                for grid in GRID_1D:
                    for n in PROBLEM_SIZES_1D:
                        total_threads = grid * block
                        if total_threads < n // 4:
                            continue
                        
                        yield KernelSpec(
                            kernel_name=_kernel_function_name(template.strip()),
                            source=template.strip(),
                            grid=[grid, 1, 1],
                            block=[block, 1, 1],
                            shared_mem_bytes=0,
                            args=args,
                            problem_size=n,
                            op_type=op_name,
                            metadata={
                                'generator': 'stage3_memory_access',
                                'access_pattern': 'random',
                                'coalesced': False,
                                'pseudo_random': True,
                            },
                        )
                        count += 1
                        if count >= max_kernels:
                            return
        
        elif op_name in ('transpose_2d', 'diagonal_access'):
            for (bx, by) in BLOCK_2D:
                for (gx, gy) in GRID_2D:
                    for (rows, cols) in PROBLEM_SIZES_2D:
                        if op_name == 'transpose_2d' and (rows % by != 0 or cols % bx != 0):
                            continue
                        
                        yield KernelSpec(
                            kernel_name=_kernel_function_name(template.strip()),
                            source=template.strip(),
                            grid=[gx, gy, 1],
                            block=[bx, by, 1],
                            shared_mem_bytes=0,
                            args=args,
                            problem_size=rows * cols,
                            op_type=op_name,
                            metadata={
                                'generator': 'stage3_memory_access',
                                'access_pattern': 'transpose' if 'transpose' in op_name else 'diagonal',
                                'coalesced': 'transpose' in op_name,
                                '2d': True,
                            },
                        )
                        count += 1
                        if count >= max_kernels:
                            return
        
        elif op_name == 'block_cyclic':
            for block_size in BLOCK_SIZES_CYCLIC:
                for block in BLOCK_1D:
                    for grid in GRID_1D:
                        for n in PROBLEM_SIZES_1D:
                            if block_size > n:
                                continue
                            if n % block_size != 0:
                                continue
                            total_threads = grid * block
                            if total_threads < n // 4:
                                continue
                            
                            yield KernelSpec(
                                kernel_name=_kernel_function_name(template.strip()),
                                source=template.strip(),
                                grid=[grid, 1, 1],
                                block=[block, 1, 1],
                                shared_mem_bytes=0,
                                args=args,
                                problem_size=n,
                                op_type=op_name,
                                metadata={
                                    'generator': 'stage3_memory_access',
                                    'access_pattern': 'block_cyclic',
                                    'coalesced': False,
                                    'block_size': block_size,
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
        'templates': list(MEMORY_TEMPLATES.keys()),
        'strides': STRIDES,
        'interleaves': INTERLEAVES,
        'segment_sizes': SEGMENT_SIZES,
        'split_ratio': split_ratio,
        'generator_version': 'stage3_v1',
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
        output_dir = 'runs/stage3_dataset'
    
    if len(sys.argv) > 2:
        max_kernels = int(sys.argv[2])
    else:
        max_kernels = 500
    
    generate_dataset(output_dir, max_kernels=max_kernels)