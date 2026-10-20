# gpu-sim-m3d.5 diagnosis (iteration 2 of 2, timebox hit)

## What works
`libfakecuda.so` builds and exports the driver API surface. Driver version 13040,
device count N, per-device attributes, contexts, memory, streams, events, modules,
`cuGetProcAddress`, `cuGetExportTable`, `cudaGetDriverEntryPoint`.

## Exact blocker
Real `libnccl.so.2` bundles **static cudart**. Its `cudawrap.cc initOnceFunc`
calls `cuGetExportTable` with two fixed UUIDs **before** any device call:

    cuGetExportTable uuid=f8cff951-2146-8b4e-b9e2-fb469e7c0dd9
    cuGetExportTable uuid=6bd5fb6c-5bf4-e74a-8987-d93912fd9df9

Return codes observed:
- `CUDA_ERROR_NOT_SUPPORTED` (801) -> NCCL fatal "API call is not supported in the installed CUDA driver"
- `CUDA_ERROR_NOT_FOUND`    (500) -> NCCL fatal "named symbol not found"
- `CUDA_SUCCESS` + zero table -> SIGSEGV at `0x0` called from
  `libcudart_static_18a98e3e05de5d265c769d6da428637b3b078702` (verified with gdb)
- `CUDA_SUCCESS` + table of stubs returning SUCCESS (`GPUSIM_EXPORT_TABLE=1`)
  -> passes that call, then `cuDeviceGetCount`/`cuDeviceGet` succeed, but cudart
  reads a driver version out of the table, gets the stub's 0, and fails with
  "CUDA driver version is insufficient for CUDA runtime version".

The tables are **opaque structs**, not flat function-pointer arrays: dumping the
real driver's tables (`/tmp/dump_table.c`) shows entries at offsets 0x10, 0x18,
0x28, 0x38 followed by pointers into libcuda — i.e. a versioned internal layout.
The static cudart resolves its driver entry points through this table, not through
dlsym, so a fake driver cannot satisfy it without reproducing that layout.

## Diagnostics added (committed)
- every `dlsym(fakeHandle)` name logged, NULL included
- `cuGetExportTable` logs the requested UUID and its return code
- `cuDeviceGetAttribute` logs every unknown enum and returns the attribute default
  (never an error), per PI instruction
- `cudaGetDriverEntryPoint` / `...ByVersion` implemented
- `-finstrument-functions` diag build (`/tmp/libfakecuda_diag.so`) gives an
  ENTER/EXIT trace of every driver call; last call before failure is always
  `cuGetExportTable`

## Requested ruling
Options: (a) relax acceptance #3 (real driver not loaded) so the shim may lazily
`dlopen("libcuda.so.1")` for `cuGetExportTable` only, keeping everything else fake;
(b) drop the static-cudart NCCL wheel and link a NCCL built against dynamic cudart
so `cuGetProcAddress`/dlsym is the entry path; (c) declare this seam NO-GO and
implement N heterogeneous devices via the m3d.7 model route + m3d.9 fault route only.

## Iteration 3-4 (PI's 2-iteration plan) -- result: NEGATIVE

Step 1 (populate version fields): implemented a synthetic table whose every slot is
a runtime-generated trampoline (`mov $slot,%r10d; jmp fc_tbl_dispatch`) so the
opaque table becomes observable (`GPUSIM_EXPORT_TABLE=1`). Running with it:

    fakecuda: table slot 2 args=(<ptr>,0,18,0,0)
    fakecuda: table slot 2 args=(<ptr>,<ptr>,16,79,64)

i.e. cudart calls slot 2 (offset +0x10) as a function. But offset +0x10 in the REAL
table (dumped offline with /tmp/dump_bytes.c) is a plain libcuda code pointer, while
offset +0x00 holds the value 0x18 (24) -- a count/size, NOT a function pointer.
Our all-pointers table therefore has the wrong shape: cudart is not indexing a flat
function array; the real table interleaves metadata (counts at +0x00, +0x20, +0x30,
+0x60, +0x90, +0xD0, +0x170 for uuid[0]) with pointer blocks.

Writing 13040 into the slot-2 out-pointer did not satisfy the version check either
(still "CUDA driver version is insufficient") and corrupted a caller struct, so the
process then crashed in `__cxa_finalize`.

Conclusion: satisfying static cudart requires reproducing the real table's
metadata+pointer layout, which is not observable through the shim and is not the
narrow shim this bead describes. Per the PI's pre-registered fallback, the
static-cudart seam is a documented NEGATIVE RESULT (kept=false). Acceptance #3 (no
real driver loaded) was never relaxed and no partial PASS is claimed.
