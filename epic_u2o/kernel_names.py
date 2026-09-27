#!/usr/bin/env python3
"""Kernel-name resolution for KernelSpec dicts (gpu-sim-9ti).

gpu-sim-9ti: generator_stage2..5 emit `kernel_name=f"{op}_kernel"` while the
`source` they carry declares `__global__ void {op}(...)`. Nothing noticed,
because nothing ever compiled those drivers -- until the -cudart=none
requirement enumeration (gpu-sim-u2o.6.1.3) did, and every stage2-5 driver
failed with "identifier X is undefined".

Two layers, because one is not enough:

* `declared_kernel_names(source)` is the authority. `kernel_driver.synthesize`
  launches the name the SOURCE declares, not the name a dict claims, so a
  mismatched spec still produces a driver that compiles. It raises only when
  the source declares no `__global__` at all, which is a genuinely broken
  spec rather than a naming disagreement.
* `check_kernel_name(spec)` returns None when the dict agrees with its source
  and a human-readable complaint when it does not. The generators are fixed to
  agree, and experiments/gpu-sim-9ti/test_kernel_names.py asserts that for
  stages 1..5, so future drift fails on CPU instead of costing a GPU job.
"""

import re

_GLOBAL_RE = re.compile(
    r"__global__\s+"                       # the qualifier
    r"(?:__device__\s+|__host__\s+)*"      # optional execution-space markers
    r"(?:extern\s+\"C\"\s+)?"              # optional C linkage
    r"(?:inline\s+|static\s+)*"            # optional modifiers
    r"(?:void|int|float|double|unsigned\s+\w+)\s+"   # return type
    r"(\w+)\s*\(",                         # the function name
    re.MULTILINE)


def declared_kernel_names(source: str) -> list[str]:
    """Every __global__ function name declared in `source`, in order."""
    if not source:
        return []
    return [m.group(1) for m in _GLOBAL_RE.finditer(source)]


def resolve_kernel_name(spec: dict) -> str:
    """The name to launch for this spec, taken from its source.

    Raises ValueError when the source declares no __global__ at all. A
    disagreement with spec['kernel_name'] is NOT an error here -- the source
    wins, and the disagreement is what check_kernel_name() reports.
    """
    names = declared_kernel_names(spec.get("source", ""))
    if not names:
        raise ValueError(
            "kernel spec declares no __global__ function in its source; "
            f"spec['kernel_name']={spec.get('kernel_name')!r} cannot be "
            "launched because there is nothing to launch")
    wanted = spec.get("kernel_name")
    if wanted in names:
        return wanted
    if len(names) == 1:
        return names[0]
    # Several kernels in one source: only an exact match is safe to guess.
    raise ValueError(
        f"spec['kernel_name']={wanted!r} is not among the __global__ "
        f"functions declared in its source {names}; refusing to guess which "
        "one to launch")


def check_kernel_name(spec: dict) -> str | None:
    """None if the spec agrees with its source, else a complaint string."""
    names = declared_kernel_names(spec.get("source", ""))
    wanted = spec.get("kernel_name")
    if not names:
        return (f"kernel_name={wanted!r} but the source declares no "
                "__global__ function")
    if wanted in names:
        return None
    return (f"kernel_name={wanted!r} does not match the __global__ function "
            f"in its source {names}; launching {names[0]!r} instead "
            "(gpu-sim-9ti)")
