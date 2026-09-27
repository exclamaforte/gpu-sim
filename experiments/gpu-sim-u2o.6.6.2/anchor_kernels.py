#!/usr/bin/env python3
"""Hand-checked anchor kernels for gpu-sim-u2o.6.6.2.

These are NOT generator output. gpu-sim-9ti showed stage2..5 emit a
`kernel_name` that does not match the `__global__` function in `source`, and
an anchor that moves is not an anchor. Every kernel body below is written by
hand here, every spec is written by hand here, and the assertion that the
declared kernel_name really is the function in `source` is checked at import
time by anchor_kernels.self_check().

Each family is measured by its own ts job, so a failure costs one family and
not the sweep.

Families:
  launch    k_empty, k_noop_store -- pure launch / dispatch cost
  memory    k_saxpy, k_relu, k_copy -- memory-bound elementwise
  reduce    k_block_sum -- shared-memory block reduction
  gemm      k_tiled_gemm -- tiled GEMM, the compute-bound end of the ladder

Launch configs deliberately include poor ones (1 block, and max block 1024)
so the sweep has dynamic range instead of only measuring the best case.
"""

import re

# ---------------------------------------------------------------------------
# kernel bodies (hand-written)
# ---------------------------------------------------------------------------

SRC_EMPTY = """
__global__ void k_empty(void) { }
"""

SRC_NOOP_STORE = """
__global__ void k_noop_store(float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && n > 0) out[0] = out[0];   /* touches one element, no traffic */
}
"""

# GRID-STRIDE, deliberately. The first version of this file used the plain
# one-element-per-thread indexing, which meant the deliberately-poor "1 block"
# config did not compute the problem at all -- it computed 1024 elements and
# reported ~0.4 us. That is a broken measurement wearing a terrible config's
# clothes, and an anchor set containing one is worse than no anchor set. With a
# grid-stride loop every config covers all n, and the 1-block config is
# genuinely (and honestly) slow.
SRC_SAXPY = """
__global__ void k_saxpy(const float *x, float *y, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = i; j < n; j += gridDim.x * blockDim.x) y[j] = alpha * x[j] + y[j];
}
"""

SRC_RELU = """
__global__ void k_relu(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = i; j < n; j += gridDim.x * blockDim.x) y[j] = x[j] > 0.0f ? x[j] : 0.0f;
}
"""

SRC_COPY = """
__global__ void k_copy(const float *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = i; j < n; j += gridDim.x * blockDim.x) dst[j] = src[j];
}
"""

# Strided copy: same bytes moved, very different access pattern. This is the
# "different access width" variant the bead asks for.
SRC_STRIDED_COPY = """
__global__ void k_strided_copy(const float *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int j = i; j < n; j += stride) dst[j] = src[j];
}
"""

SRC_BLOCK_SUM = """
/* REDUCE_TILE must be >= the largest blockDim in the sweep (1024). The first
 * version used 256 while the sweep also ran block=1024 configs, which is an
 * out-of-bounds shared-memory write -- caught by reading the code, not by the
 * GPU. The tree loop starts at blockDim/2, so a smaller blockDim is fine with
 * a 1024-entry tile. */
#define REDUCE_TILE 1024
__global__ void k_block_sum(const float *in, float *out, int n) {
    __shared__ float tile[REDUCE_TILE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) tile[tid] += tile[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = tile[0];
}
"""

# Tiled GEMM. M comes from gridDim.y * blockDim.y, N and K from `n`, so the
# host driver only has to supply the single `n` scalar the synthesiser knows
# how to pass (it maps the name 'n'/'N' to problem_size).
SRC_TILED_GEMM = """
#define GTILE 32
__global__ void k_tiled_gemm(const float *A, const float *B, float *C, int n) {
    __shared__ float As[GTILE][GTILE];
    __shared__ float Bs[GTILE][GTILE];
    int D = gridDim.y * GTILE;          /* matrix dim, carried by the launch */
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * GTILE + ty;      /* M index */
    int col = blockIdx.x * GTILE + tx;      /* N index */
    float acc = 0.0f;
    for (int t = 0; t < (D + GTILE - 1) / GTILE; t++) {
        int a_col = t * GTILE + tx;
        int b_row = t * GTILE + ty;
        As[ty][tx] = (row < D && a_col < D) ? A[row * D + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < D && col < D) ? B[b_row * D + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < GTILE; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < D && col < D) C[row * D + col] = acc;
}
"""

# ---------------------------------------------------------------------------
# hand-written specs
# ---------------------------------------------------------------------------

def _spec(name, source, grid, block, args, n, shmem=0, op_type="anchor"):
    return {
        "source": source.strip() + "\n",
        "kernel_name": name,
        "grid": list(grid),
        "block": list(block),
        "shared_mem_bytes": shmem,
        "args": args,
        "problem_size": int(n),
        "op_type": op_type,
        "metadata": {"generator": "anchor_kernels (hand-written, not generated)"},
    }


F32_IN = {"name": "x", "type": "const float*", "ptr": True}
F32_OUT = {"name": "y", "type": "float*", "ptr": True}
INT_N = {"name": "n", "type": "int", "ptr": False}
# Distinct names: the synthesiser derives host/device variable names from the
# argument name, so two arguments called 'y' would collide in main().
GEMM_A = {"name": "A", "type": "const float*", "ptr": True}
GEMM_B = {"name": "B", "type": "const float*", "ptr": True}
GEMM_C = {"name": "C", "type": "float*", "ptr": True}
F_ALPHA = {"name": "alpha", "type": "float", "ptr": False}


def configs(family: str) -> list[dict]:
    """(spec, config-label) pairs for one family.

    Sizes span >=2 decades; the launch configs deliberately include a poor one
    (single block) and the maximum block size, so the anchor set has dynamic
    range rather than only best-case points.
    """
    out = []
    if family == "launch":
        for n in (1 << 10, 1 << 16, 1 << 22):
            for grid, block, tag in (("auto", 256, "auto/256"),
                                     ("auto", 1024, "auto/1024"),
                                     (1, 1024, "1block/1024")):
                out.append((_spec("k_empty", SRC_EMPTY, _grid(grid, n, block),
                                  (block, 1, 1), [], n, op_type="launch"),
                            f"empty n={n} {tag}"))
                out.append((_spec("k_noop_store", SRC_NOOP_STORE,
                                  _grid(grid, n, block), (block, 1, 1),
                                  [F32_OUT, INT_N], n, op_type="launch"),
                            f"noop_store n={n} {tag}"))
        return out

    if family == "memory":
        for n in (1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26):
            for grid, block, tag in (("auto", 256, "auto/256"),
                                     ("auto", 1024, "auto/1024"),
                                     (1, 1024, "1block/1024")):
                g = _grid(grid, n, block)
                out.append((_spec("k_saxpy", SRC_SAXPY, g, (block, 1, 1),
                                  [F32_IN, F32_OUT, F_ALPHA, INT_N], n,
                                  op_type="memory"), f"saxpy n={n} {tag}"))
                out.append((_spec("k_relu", SRC_RELU, g, (block, 1, 1),
                                  [F32_IN, F32_OUT, INT_N], n,
                                  op_type="memory"), f"relu n={n} {tag}"))
                out.append((_spec("k_copy", SRC_COPY, g, (block, 1, 1),
                                  [F32_IN, F32_OUT, INT_N], n,
                                  op_type="memory"), f"copy n={n} {tag}"))
                out.append((_spec("k_strided_copy", SRC_STRIDED_COPY, g,
                                  (block, 1, 1), [F32_IN, F32_OUT, INT_N], n,
                                  op_type="memory"),
                            f"strided_copy n={n} {tag}"))
        return out

    if family == "reduce":
        for n in (1 << 12, 1 << 16, 1 << 20, 1 << 24):
            for block, tag in ((256, "auto/256"), (1024, "auto/1024")):
                blocks = (n + block - 1) // block
                out.append((_spec("k_block_sum", SRC_BLOCK_SUM, (blocks, 1, 1),
                                  (block, 1, 1), [F32_IN, F32_OUT, INT_N], n,
                                  shmem=1024 * 4, op_type="reduce"),
                            f"block_sum n={n} {tag}"))
        return out

    if family == "gemm":
        for n in (512, 1024, 2048, 4096):
            tiles = (n + 31) // 32
            # The synthesiser's driver allocates problem_size elements per
            # pointer argument, so for a D x D matrix the buffer size is D*D and
            # the kernel derives D from the launch geometry (gridDim.y * GTILE).
            for tag, blk in (("32x32", (32, 32, 1)), ("16x16", (16, 16, 1))):
                out.append((_spec("k_tiled_gemm", SRC_TILED_GEMM,
                                  (tiles, tiles, 1), blk,
                                  [GEMM_A, GEMM_B, GEMM_C, INT_N], n * n,
                                  shmem=2 * 32 * 32 * 4, op_type="gemm"),
                            f"tiled_gemm {n}^3 {tag}"))
        return out

    raise ValueError(f"unknown family {family!r}")


def _grid(grid, n, block):
    if grid != "auto":
        return (grid, 1, 1)
    b = block if isinstance(block, int) else block[2]
    return (max(1, (n + b - 1) // b), 1, 1)


FAMILIES = ("launch", "memory", "reduce", "gemm")


def self_check() -> None:
    """Fail loudly at import if a declared kernel_name is not in its source.

    This is exactly the class of defect gpu-sim-9ti is about, so the anchor
    set asserts its own absence of it rather than trusting review.
    """
    for family in FAMILIES:
        for spec, label in configs(family):
            names = re.findall(r"__global__\s+void\s+(\w+)\s*\(", spec["source"])
            if spec["kernel_name"] not in names:
                raise AssertionError(
                    f"anchor_kernels: {label} declares kernel_name="
                    f"{spec['kernel_name']!r} but source defines {names}")


if __name__ == "__main__":
    self_check()
    for fam in FAMILIES:
        cfgs = configs(fam)
        print(f"{fam:8s} {len(cfgs):3d} configs  e.g. {cfgs[0][1]}")
    print("self_check OK")
