#!/usr/bin/env python3
"""Kernel-driver synthesizer (gpu-sim-u2o.6.9).

Takes a stage-1 KernelSpec (bare __global__ body) and emits a full compilable
.cu program: cudaMalloc buffers sized from problem_size, seeded host data,
5 warmup launches, then 20 timed launches via cudaEventRecord /
cudaEventElapsedTime (per measurement decision u2o.3), printing each trial
as one JSON line to stdout.

Median/IQR are intentionally NOT computed here -- measurement_stats.py owns
that in Python. Supports the stage-1 elementwise family (float/int/double
pointers + float/int scalars). Other families fail loudly, not silently.
"""

import json
import re
import sys

WARMUP = 5
TRIALS = 20

# base CUDA type -> (sizeof, host fill expr of index i, kernel scalar value)
_SCALAR_DEFAULTS = {
    'float': '2.0f',
    'double': '2.0',
    'int': '1',
}

_BASE_SIZES = {'float': 4, 'double': 8, 'int': 4, 'char': 1,
               'half': 2, 'unsigned int': 4}


def _base_type(ctype: str) -> str:
    """Strip const/__restrict__/pointers -> base type name."""
    t = ctype.replace('const', '').replace('__restrict__', '').strip()
    t = t.rstrip('*').strip()
    return t


def _is_pointer(ctype: str) -> bool:
    return '*' in ctype


def synthesize(spec: dict) -> str:
    """Build a full .cu program from a KernelSpec dict.

    Required spec keys: source, kernel_name, grid, block, args,
    problem_size. Optional: shared_mem_bytes (default 0).
    """
    for key in ('source', 'kernel_name', 'grid', 'block', 'args', 'problem_size'):
        if key not in spec:
            raise ValueError(f"KernelSpec missing required key: {key}")
    src = spec['source']
    # gpu-sim-9ti: the name to launch comes from the SOURCE, not from the
    # dict's kernel_name. generator_stage2..5 declared f"{op}_kernel" while
    # their source defines {op}, so every stage2-5 driver failed to compile
    # with "identifier X is undefined". The source is the authority here; a
    # spec whose source declares no __global__ at all is a real error.
    from epic_u2o.kernel_names import resolve_kernel_name
    kname = resolve_kernel_name(spec)
    grid = spec['grid']
    block = spec['block']
    args = spec['args']
    n = int(spec['problem_size'])
    shmem = int(spec.get('shared_mem_bytes', 0))

    if '__global__' not in src:
        raise ValueError("spec['source'] contains no __global__ kernel")

    decls, mallocs, fills, call_args, frees = [], [], [], [], []
    for a in args:
        name = a['name']
        ctype = a['type']
        if _is_pointer(ctype):
            base = _base_type(ctype)
            if base not in _BASE_SIZES:
                raise ValueError(f"unsupported pointer base type: {ctype!r} "
                                 f"(stage-1 synthesizer handles {sorted(_BASE_SIZES)})")
            decls.append(f"    {base} *d_{name} = NULL;")
            decls.append(f"    {base} *h_{name} = ({base}*)malloc(N * sizeof({base}));")
            mallocs.append(f"    CHECK(cudaMalloc((void**)&d_{name}, N * sizeof({base})));")
            if name.startswith('out') or name == 'y':
                fills.append(f"    for (int i = 0; i < N; i++) h_{name}[i] = ({base})0;")
            else:
                fills.append(f"    for (int i = 0; i < N; i++) h_{name}[i] = ({base})(i * 0.5f + 1.0f);")
            mallocs.append(f"    CHECK(cudaMemcpy(d_{name}, h_{name}, N * sizeof({base}), cudaMemcpyHostToDevice));")
            call_args.append(f"d_{name}")
            frees.append(f"    cudaFree(d_{name}); free(h_{name});")
        else:
            base = _base_type(ctype)
            if name in ('n', 'N', 'size', 'len', 'count'):
                val = 'N'
            elif base in _SCALAR_DEFAULTS:
                val = _SCALAR_DEFAULTS[base]
            else:
                raise ValueError(f"unsupported scalar type: {ctype!r}")
            call_args.append(val)

    gx, gy, gz = grid
    bx, by, bz = block
    call = f"{kname}<<<grid, block, {shmem}>>>({', '.join(call_args)})"

    lines = [
        '#include <stdio.h>',
        '#include <stdlib.h>',
        '#include <cuda_runtime.h>',
        '',
        '#define CHECK(call) do { cudaError_t e = (call); \\',
        '    if (e != cudaSuccess) { \\',
        '        fprintf(stderr, "{\\"error\\": \\"%s:%d %s\\"} \\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\',
        '        return 2; } } while (0)',
        '',
        src.strip(),
        '',
        'int main(void) {',
        '    const int N = %d;' % n,
        '    dim3 grid(%d, %d, %d);' % (gx, gy, gz),
        '    dim3 block(%d, %d, %d);' % (bx, by, bz),
        '\n'.join(decls),
        '    size_t dev_free_before_b = 0, dev_total_b = 0;',
        '    CHECK(cudaMemGetInfo(&dev_free_before_b, &dev_total_b));',
        '\n'.join(mallocs),
        '\n'.join(fills),
        '    cudaEvent_t start, stop;',
        '    CHECK(cudaEventCreate(&start));',
        '    CHECK(cudaEventCreate(&stop));',
        '    size_t dev_free_after_b = 0;',
        '    CHECK(cudaMemGetInfo(&dev_free_after_b, &dev_total_b));',
        '    printf("{\\"mem\\": {\\"delta_mb\\": %.1f, \\"free_after_mb\\": %.1f, \\"total_mb\\": %.1f, \\"device_used_mb\\": %.1f}}\\n",',
        '           (dev_free_before_b - dev_free_after_b) / 1048576.0,',
        '           dev_free_after_b / 1048576.0, dev_total_b / 1048576.0,',
        '           (dev_total_b - dev_free_after_b) / 1048576.0);',
        '    /* delta_mb is this run footprint; device_used_mb is device-wide */',
        '    fflush(stdout);',
        '    for (int w = 0; w < %d; w++) { %s; }' % (WARMUP, call),
        '    CHECK(cudaDeviceSynchronize());',
        '    for (int t = 0; t < %d; t++) {' % TRIALS,
        '        float ms = 0.0f;',
        '        CHECK(cudaEventRecord(start));',
        f'        {call};',
        '        CHECK(cudaEventRecord(stop));',
        '        CHECK(cudaEventSynchronize(stop));',
        '        CHECK(cudaEventElapsedTime(&ms, start, stop));',
        '        printf("{\\"trial\\": %d, \\"time_ms\\": %.6f}\\n", t, ms);',
        '        fflush(stdout);',
        '    }',
        '    CHECK(cudaDeviceSynchronize());',
        '\n'.join(frees),
        '    return 0;',
        '}',
        '',
    ]
    return '\n'.join(lines)


def spec_from_stage1(op='scale', grid=None, block=None, n=4096):
    """Convenience: build one stage-1 KernelSpec dict without importing the
    generator module (keeps ts jobs dependency-light)."""
    # Derive the repo root from this file so the function works in a lab
    # worktree (or any checkout), not just the original clone path.
    # (gpu-sim-u2o.6.1.1; this branch predates the fix and carried a stale
    # full copy of the file, so the rebase replayed the old line.)
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from epic_u2o.generator_stage1 import generate_kernels
    for spec in generate_kernels(
            ops=[op], grid_sweeps=[grid or [4, 1, 1]],
            block_sweeps=[block or [256, 1, 1]], problem_sizes=[n],
            max_kernels=1):
        d = spec.to_dict()
        return d
    raise RuntimeError("generator produced no spec")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Synthesize a driver .cu from a stage-1 spec')
    p.add_argument('--op', default='scale')
    p.add_argument('--grid', default='4,1,1')
    p.add_argument('--block', default='256,1,1')
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('-o', '--output', required=True)
    a = p.parse_args()
    grid = [int(x) for x in a.grid.split(',')]
    block = [int(x) for x in a.block.split(',')]
    spec = spec_from_stage1(a.op, grid, block, a.n)
    if grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2] < a.n:
        raise SystemExit("grid*block covers fewer threads than n; pick a bigger config")
    with open(a.output, 'w') as f:
        f.write(synthesize(spec))
    print(json.dumps({'output': a.output, 'kernel': spec['kernel_name'],
                      'trials': TRIALS, 'warmup': WARMUP}))
