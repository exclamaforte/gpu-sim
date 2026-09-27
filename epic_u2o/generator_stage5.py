#!/usr/bin/env python3
"""Stage-5 generator: control-flow divergence and atomics family.

Branch-divergent kernels and atomic-heavy kernels (histograms, counters),
stressing warp-scheduling behavior that neither pure compute-bound nor
pure memory-bound families exercise.
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

    def kernel_hash(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)


DIVERGENCE_TEMPLATES = {
    'divergent_branch': '''
__global__ void divergent_branch(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        if ((i & 1) == 0) out[i] = v * 2.0f + 1.0f;
        else out[i] = v * 0.5f - 1.0f;
    }
}
''',
    'divergent_switch': '''
__global__ void divergent_switch(const int* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int c = in[i] & 3;
        float v = (float)in[i];
        if (c == 0) out[i] = v + 1.0f;
        else if (c == 1) out[i] = v * 2.0f;
        else if (c == 2) out[i] = v * v;
        else out[i] = -v;
    }
}
''',
    'divergent_loop': '''
__global__ void divergent_loop(const int* __restrict__ iters, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc = 0.0f;
        int m = iters[i] & 15;
        for (int j = 0; j < m; ++j) acc += (float)j * 0.5f;
        out[i] = acc;
    }
}
''',
    'histogram_global': '''
__global__ void histogram_global(const int* __restrict__ in, int* __restrict__ hist, int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int b = in[i] % nbins;
        if (b < 0) b += nbins;
        atomicAdd(&hist[b], 1);
    }
}
''',
    'histogram_shared': '''
__global__ void histogram_shared(const int* __restrict__ in, int* __restrict__ hist, int n, int nbins) {
    extern __shared__ int s_hist[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    for (int b = tid; b < nbins; b += blockDim.x) s_hist[b] = 0;
    __syncthreads();
    if (i < n) {
        int b = in[i] % nbins;
        if (b < 0) b += nbins;
        atomicAdd(&s_hist[b], 1);
    }
    __syncthreads();
    for (int b = tid; b < nbins; b += blockDim.x) atomicAdd(&hist[b], s_hist[b]);
}
''',
    'atomic_counter': '''
__global__ void atomic_counter(const float* __restrict__ in, float* __restrict__ out, int* __restrict__ counter, int n, float thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && in[i] > thresh) {
        int idx = atomicAdd(counter, 1);
        out[idx] = in[i];
    }
}
''',
}

BLOCK_SIZES = [64, 128, 256, 512]
GRID_SIZES = [1, 2, 4, 8, 16, 32, 64]
PROBLEM_SIZES = [1024, 4096, 16384, 65536, 262144]
NBINS = [16, 64, 256]


def _args(op: str) -> list:
    if op in ('divergent_branch',):
        return [{'name': 'in', 'type': 'const float*', 'ptr': True},
                {'name': 'out', 'type': 'float*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False}]
    if op in ('divergent_switch', 'divergent_loop'):
        return [{'name': 'in', 'type': 'const int*', 'ptr': True},
                {'name': 'out', 'type': 'float*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False}]
    if op in ('histogram_global', 'histogram_shared'):
        return [{'name': 'in', 'type': 'const int*', 'ptr': True},
                {'name': 'hist', 'type': 'int*', 'ptr': True},
                {'name': 'n', 'type': 'int', 'ptr': False},
                {'name': 'nbins', 'type': 'int', 'ptr': False}]
    return [{'name': 'in', 'type': 'const float*', 'ptr': True},
            {'name': 'out', 'type': 'float*', 'ptr': True},
            {'name': 'counter', 'type': 'int*', 'ptr': True},
            {'name': 'n', 'type': 'int', 'ptr': False},
            {'name': 'thresh', 'type': 'float', 'ptr': False}]


def generate_kernels(max_kernels: int = 300, ops: Optional[list] = None) -> Iterator[KernelSpec]:
    ops = ops or list(DIVERGENCE_TEMPLATES.keys())
    count = 0
    for op in ops:
        tmpl = DIVERGENCE_TEMPLATES[op]
        bins_list = NBINS if 'histogram' in op else [0]
        for nb in bins_list:
            for block in BLOCK_SIZES:
                for grid in GRID_SIZES:
                    for n in PROBLEM_SIZES:
                        if grid * block < n // 4:
                            continue
                        smem = 0
                        if op == 'histogram_shared':
                            smem = nb * 4
                            if smem == 0 or smem > 48 * 1024:
                                continue
                        yield KernelSpec(
                            kernel_name=_kernel_function_name(tmpl.strip()),
                            source=tmpl.strip(),
                            grid=[grid, 1, 1],
                            block=[block, 1, 1],
                            shared_mem_bytes=smem,
                            args=_args(op),
                            problem_size=n,
                            op_type=op,
                            metadata={
                                'generator': 'stage5_divergence_atomics',
                                'op_type': op,
                                'divergent': 'histogram' not in op and 'counter' not in op,
                                'uses_atomics': 'histogram' in op or 'counter' in op,
                                'uses_shared_memory': op == 'histogram_shared',
                                'nbins': nb,
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
            'templates': list(DIVERGENCE_TEMPLATES.keys()),
            'generator_version': 'stage5_v1'}
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Generated {len(kernels)} kernels:")
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")
    return meta


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'runs/stage5_dataset'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    generate_dataset(out, max_kernels=n)
