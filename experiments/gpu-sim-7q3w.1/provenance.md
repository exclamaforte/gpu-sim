# gpu-sim-7q3w.1 provenance

## Wheels (project-local cache: experiments/gpu-sim-7q3w.1/wheels/)
- cu12: nvidia-nccl-cu12 2.32.6 (nvidia_nccl_cu12-2.32.6-py3-none-manylinux_2_27_x86_64.whl)
  - wheel sha256: 2a6e6d5ddfc08970eca872be3dfa2650b24fa10a2043974465a9cae9c64ed6d8
- cu13: nvidia-nccl-cu13 2.32.6 (nvidia_nccl_cu13-2.32.6-py3-none-manylinux_2_27_x86_64.whl)
  - wheel sha256: 76e7aad00482092baf01e13e66f441d54c96b5cf1e3f143dee27db7f731d41ba

## ncclGetVersion
- cu12: 0x5aa6 (2.32.6)
- cu13: 0x5aa6 (2.32.6)

## Plugin
- experiments/gpu-sim-7q3w.1/libncclNetPlugin.so built from experiments/gpu-sim-7q3w.1/plugin_v11.c (copied from 465.1, then added constructor debug line and getProperties_fail returning non-success). Full 176-byte ncclNet_v11_t layout, 22 members, all non-NULL, _Static_assert == 176.

## ABI acceptance
- Top ABI on 2.32.x is v12; v11 is our fallback (plugin exports v6..v11, max v11). NCCL accepts v11 on both wheels ("Loaded net plugin u2o-minimal-net (v11)").

## Route (b)
The vtable is a stub by design and NCCL refuses the plugin deterministically (getProperties returns non-success); no SIGSEGV. NCCL falls back to Socket, so the probe completes via Socket; allreduce is not claimed as plugin evidence. Refusal message shape is nondeterministic: cu13 8/10 "Initialized NET plugin u2o-minimal-net" + "(ncclNet_getProperties) -> 1", 2/10 "Failed to initialize NET plugin u2o-minimal-net"; cu12 4/10 vs 6/10. Both shapes mean refused; both exit 0.

## Measured runs
- cu13 (output.log): NCCL 2.32.6+cuda13.5, "Loaded net plugin u2o-minimal-net (v11)", exit 0 (Socket fallback).
- cu12 (output_cu12.log): NCCL 2.32.6+cuda12.9, "Loaded net plugin u2o-minimal-net (v11)", exit 0 (Socket fallback).
