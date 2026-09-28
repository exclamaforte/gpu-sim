#!/usr/bin/env python3
"""Pre-flight structural validator using gpu-sim fakegpu (LD_PRELOAD).

Runs every newly-generated program once under libfakegpu.so as a fast,
GPU-free smoke test to catch malformed CUDA API usage, bad launch configs,
and host-side crashes before they reach the expensive real-GPU queue.
"""

import subprocess
import tempfile
import os
import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class PreflightResult:
    """Result of pre-flight validation."""
    success: bool
    compile_ok: bool
    run_ok: bool
    stdout: str
    stderr: str
    wall_time_sec: float
    errors: list[str]
    warnings: list[str]


class PreflightValidator:
    """Structural validator using fakegpu LD_PRELOAD."""
    
    def __init__(
        self,
        fakegpu_lib: str = "libfakegpu.so",
        fakegpu_mem_mb: int = 24576,
        timeout_sec: int = 10,
        arch: str = 'sm_90',
    ):
        self.fakegpu_lib = fakegpu_lib
        self.fakegpu_mem_mb = fakegpu_mem_mb
        self.timeout_sec = timeout_sec
        self.arch = arch
        
        # Check if fakegpu library exists
        self.fakegpu_available = self._check_fakegpu()
    
    def _check_fakegpu(self) -> bool:
        """Check if libfakegpu.so is available."""
        # Check common locations
        paths = [
            self.fakegpu_lib,
            f"/usr/local/lib/{self.fakegpu_lib}",
            f"/usr/lib/{self.fakegpu_lib}",
            f"/home/gabe/Documents/gpu-sim/build/{self.fakegpu_lib}",
        ]
        
        for path in paths:
            if os.path.exists(path):
                return True
        
        # Also check LD_LIBRARY_PATH
        for path in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
            full = os.path.join(path, self.fakegpu_lib)
            if os.path.exists(full):
                return True
        
        return False
    
    def validate_kernel(
        self,
        kernel_source: str,
        launch_config: dict,
        kernel_name: str = "kernel",
    ) -> PreflightResult:
        """Validate a kernel via fakegpu structural check."""
        import time
        
        start_time = time.time()
        errors = []
        warnings = []
        
        # Step 1: Compile kernel
        compile_ok, compile_stdout, compile_stderr = self._compile_kernel(kernel_source)
        if not compile_ok:
            return PreflightResult(
                success=False,
                compile_ok=False,
                run_ok=False,
                stdout=compile_stdout,
                stderr=compile_stderr,
                wall_time_sec=time.time() - start_time,
                errors=[f"Compilation failed: {compile_stderr}"],
                warnings=[],
            )
        
        # Step 2: Run under fakegpu
        # gpu-sim-8dw: the source and size are passed through, because the
        # smoke test has to LAUNCH the kernel under test, not a stand-in.
        run_ok, run_stdout, run_stderr = self._run_under_fakegpu(
            launch_config, kernel_name,
            kernel_source=kernel_source,
            problem_size=int((launch_config or {}).get("problem_size", 4096)),
        )
        
        # Analyze results
        if not run_ok:
            errors.append(f"Fakegpu run failed: {run_stderr}")
        
        # Check for common issues
        if "out of memory" in run_stderr.lower():
            errors.append("VRAM allocation exceeded fake device limit")
        
        if "invalid configuration" in run_stderr.lower():
            errors.append("Invalid launch configuration")
        
        if "misaligned address" in run_stderr.lower():
            errors.append("Misaligned memory access")
        
        if "illegal instruction" in run_stderr.lower():
            errors.append("Illegal instruction encountered")
        
        if "timeout" in run_stderr.lower():
            errors.append("Execution timeout (infinite loop?)")
        
        return PreflightResult(
            success=compile_ok and run_ok,
            compile_ok=compile_ok,
            run_ok=run_ok,
            stdout=run_stdout,
            stderr=run_stderr,
            wall_time_sec=time.time() - start_time,
            errors=errors,
            warnings=warnings,
        )
    
    def _compile_kernel(self, kernel_source: str) -> tuple[bool, str, str]:
        """Compile kernel to executable."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_source)
            cu_file = f.name
        
        # -c: compile to an OBJECT, not an executable. A .cu that contains
        # only __global__ functions has no main(), so linking it always failed
        # with "undefined reference to `main'" -- i.e. step 1 of
        # validate_kernel could never succeed, and every caller saw a
        # compile failure rather than the smoke-test result it asked for.
        # (Found while implementing 8dw: the stub was hiding a second
        # always-wrong step behind it.)
        obj_file = cu_file.replace('.cu', '.o')
        
        try:
            result = subprocess.run(
                ['nvcc', '-c', f'-arch={self.arch}', cu_file, '-o', obj_file],
                capture_output=True, text=True, timeout=60
            )
            
            return (
                result.returncode == 0,
                result.stdout,
                result.stderr,
            )
        except subprocess.TimeoutExpired:
            return False, "", "Compilation timeout"
        except Exception as e:
            return False, "", f"Compilation error: {e}"
        finally:
            if os.path.exists(cu_file):
                os.unlink(cu_file)
            if os.path.exists(obj_file):
                os.unlink(obj_file)
    
    # The harness that actually launches the kernel under test. gpu-sim-8dw:
    # the previous body of _run_under_fakegpu returned (True, "Preflight passed
    # (placeholder)", "") unconditionally -- a validation that could not fail.
    # This is the real thing: the kernel source is pasted in, a main() launches
    # it with the CONFIGURED grid/block, and every CUDA return value is checked
    # rather than ignored.
    _HARNESS = """
#include <cstdio>
#include <cuda_runtime.h>

%(source)s

static int failures = 0;
#define CHECK_API(call)                                                  \
    do {                                                                 \
        cudaError_t e_ = (call);                                          \
        if (e_ != cudaSuccess) {                                          \
            printf("CUDA_ERROR %%s: %%s\\n", #call,                     \
                   cudaGetErrorString(e_));                               \
            failures++;                                                   \
        }                                                                 \
    } while (0)

int main() {
    CHECK_API(cudaFree(0));                       /* context creation */
    // No cudaMemGetInfo probe here: libfakegpu does not export it (see 8dw),
    // and a diagnostic print is not worth failing the link over. The device
    // totals come from the fake's own cudaMemGetInfo once that is exported.

    const int n = %(n)d;
    float *d_a = NULL, *d_b = NULL;
    CHECK_API(cudaMalloc((void**)&d_a, (size_t)n * sizeof(float)));
    CHECK_API(cudaMalloc((void**)&d_b, (size_t)n * sizeof(float)));
    CHECK_API(cudaMemset(d_a, 0, (size_t)n * sizeof(float)));
    CHECK_API(cudaMemset(d_b, 0, (size_t)n * sizeof(float)));

    dim3 grid(%(gx)d, %(gy)d, %(gz)d);
    dim3 block(%(bx)d, %(by)d, %(bz)d);
    %(launch)s;
    CHECK_API(cudaGetLastError());                 /* launch-time errors */
    CHECK_API(cudaDeviceSynchronize());            /* execution errors */
    printf("launch_done grid=(%%d,%%d,%%d) block=(%%d,%%d,%%d)\\n",
           %(gx)d, %(gy)d, %(gz)d, %(bx)d, %(by)d, %(bz)d);
    CHECK_API(cudaFree(d_a));
    CHECK_API(cudaFree(d_b));
    printf("failures=%%d\\n", failures);
    return failures == 0 ? 0 : 1;
}
"""

    def _resolve_fakegpu(self) -> str | None:
        """Absolute path of libfakegpu.so, or None. Needed for -rpath/-L."""
        for path in (self.fakegpu_lib,
                     f"/usr/local/lib/{self.fakegpu_lib}",
                     f"/usr/lib/{self.fakegpu_lib}",
                     f"/home/gabe/Documents/gpu-sim/build/{self.fakegpu_lib}",
                     f"/home/gabe/Documents/gpu-sim/{self.fakegpu_lib}",
                     f"{Path(__file__).resolve().parent.parent}/"
                     f"{self.fakegpu_lib}"):
            if os.path.exists(path):
                return os.path.abspath(path)
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
            if path and os.path.exists(os.path.join(path, self.fakegpu_lib)):
                return os.path.abspath(os.path.join(path, self.fakegpu_lib))
        return None

    def _run_under_fakegpu(self, launch_config: dict, kernel_name: str,
                           kernel_source: str = "",
                           problem_size: int = 4096,
                           ) -> tuple[bool, str, str]:
        """Compile and run a harness that LAUNCHES the kernel under libfakegpu.

        gpu-sim-8dw. This used to `return True, "Preflight passed
        (placeholder)", ""` unconditionally -- a validation that could not fail,
        reachable from safe_harness.preflight_check. It is now the real thing,
        and the important part is the LAST branch: if the smoke test cannot be
        performed, it FAILS LOUDLY rather than reporting a pass. "Not
        implemented" and "passed" must never be the same answer.

        The route is `-cudart=none` + a direct link to libfakegpu (the pattern
        gpu-sim-u2o.6.1.3 established and made mechanical), not LD_PRELOAD:
        nvcc defaults to -cudart=static, so a preloaded symbol would be
        shadowed by the one inside the executable and the interposition would
        be a no-op -- measured, not assumed.
        """
        import time

        lib = self._resolve_fakegpu()
        if lib is None:
            return False, "", (f"preflight cannot run: {self.fakegpu_lib} not "
                               f"found, and 'cannot check' is NOT a pass")

        def _dim(key, default=1):
            v = (launch_config or {}).get(key, default)
            if isinstance(v, (list, tuple)):
                v = v[0] if v else default
            try:
                return max(1, int(v))
            except (TypeError, ValueError):
                return default

        gx, gy, gz = _dim("grid_x"), _dim("grid_y"), _dim("grid_z")
        bx, by, bz = _dim("block_x"), _dim("block_y"), _dim("block_z")
        threads = bx * by * bz
        if threads > 1024:
            return False, "", (f"invalid launch config: block "
                               f"{bx}x{by}x{bz} = {threads} threads "
                               f"(max 1024)")
        if bz > 64:
            return False, "", f"invalid launch config: block z={bz} > 64"

        src = self._HARNESS % {
            "source": kernel_source,
            "n": int(problem_size),
            "gx": gx, "gy": gy, "gz": gz,
            "bx": bx, "by": by, "bz": bz,
            "launch": f"{kernel_name}<<<grid, block>>>(d_a, d_b, n);",
        }
        libdir = os.path.dirname(lib)
        with tempfile.TemporaryDirectory() as td:
            cu = os.path.join(td, "preflight_harness.cu")
            exe = os.path.join(td, "preflight_harness.out")
            with open(cu, "w") as f:
                f.write(src)
            try:
                c = subprocess.run(
                    ["nvcc", "-cudart=none", f"-arch={self.arch}", "-O2",
                     cu, "-o", exe, f"-L{libdir}", "-lfakegpu",
                     "-Xlinker", "-rpath", "-Xlinker", libdir],
                    capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                return False, "", "preflight compile timeout"
            if c.returncode != 0:
                return False, c.stdout, f"preflight harness did not compile: {c.stderr[:600]}"
            env = os.environ.copy()
            # No LD_PRELOAD: the link is the interposition. CUDA_VISIBLE_DEVICES
            # is deliberately NOT blanked -- run_kernel overwrites it, and a
            # device-free claim must not be manufactured by hiding the GPU.
            env.pop("LD_PRELOAD", None)
            env["FAKEGPU_DEVICE_MEM_MB"] = str(self.fakegpu_mem_mb)
            try:
                r = subprocess.run([exe], capture_output=True, text=True,
                                   timeout=self.timeout_sec, env=env, cwd=td)
            except subprocess.TimeoutExpired:
                return False, "", ("preflight run timed out -- possible "
                                   "infinite loop in the kernel under test")
            except Exception as e:
                return False, "", f"preflight run error: {e}"
            combined = f"{r.stdout}\n{r.stderr}"
            if "CUDA_ERROR" in combined:
                return False, r.stdout, combined
            if r.returncode != 0:
                return False, r.stdout, combined or "preflight harness failed"
            return True, r.stdout, combined


class StructuralValidator:
    """Higher-level structural validator integrated with the pipeline."""
    
    def __init__(
        self,
        fakegpu_lib: str = "libfakegpu.so",
        fakegpu_mem_mb: int = 24576,
        timeout_sec: int = 10,
    ):
        self.validator = PreflightValidator(
            fakegpu_lib=fakegpu_lib,
            fakegpu_mem_mb=fakegpu_mem_mb,
            timeout_sec=timeout_sec,
        )
    
    def validate(
        self,
        kernel_source: str,
        grid: list[int],
        block: list[int],
        shared_mem: int = 0,
    ) -> PreflightResult:
        """Validate kernel structure before real GPU execution."""
        launch_config = {
            'grid': grid,
            'block': block,
            'shared_mem': shared_mem,
        }
        
        return self.validator.validate_kernel(kernel_source, launch_config)
    
    def validate_batch(
        self,
        kernels: list[dict],
    ) -> list[PreflightResult]:
        """Validate a batch of kernels."""
        results = []
        for kernel in kernels:
            result = self.validate(
                kernel['source'],
                kernel['grid'],
                kernel['block'],
                kernel.get('shared_mem', 0),
            )
            results.append(result)
        return results


def validate_kernel_source(kernel_source: str) -> tuple[bool, list[str]]:
    """Static analysis of kernel source without compilation."""
    errors = []
    
    # Check for disallowed constructs (from v1_scope.md)
    disallowed = {
        'cudaLaunchCooperativeKernel': 'Cooperative kernels not allowed in v1',
        'cudaLaunchCooperativeKernelMultiDevice': 'Multi-device cooperative kernels not allowed',
        'cudaStreamAddCallback': 'Host callbacks not allowed',
        'cudaHostFn_t': 'Host callbacks not allowed',
        '__global__': None,  # Count them
        '<<<': None,  # Count launch sites
    }
    
    global_count = kernel_source.count('__global__')
    launch_count = kernel_source.count('<<<')
    
    if global_count == 0:
        errors.append("No __global__ kernel found")
    elif global_count > 1:
        errors.append(f"Multiple __global__ kernels ({global_count}) - only one allowed in v1")
    
    warnings: list[str] = []
    if launch_count == 0:
        # Bare kernel bodies (driver adds the launch site) legitimately
        # have no <<< >>>; only a warning, never a failure (u2o.6.1.1).
        warnings.append("No kernel launch site (<<< >>>) in this source; "
                        "expected for bare kernel bodies (driver adds it)")
    elif launch_count > 1:
        warnings = [f"Multiple kernel launch sites ({launch_count}) - only first will be measured"]
    
    # Check for dynamic parallelism (kernel launching kernel from device code)
    # Look for <<< inside a __global__ function body
    lines = kernel_source.split('\n')
    in_global = False
    brace_depth = 0
    for line in lines:
        stripped = line.strip()
        if '__global__' in stripped:
            in_global = True
        if in_global:
            if '{' in stripped:
                brace_depth += stripped.count('{')
            if '}' in stripped:
                brace_depth -= stripped.count('}')
            if brace_depth > 0 and '<<<' in stripped and '>>>' in stripped:
                # Check it's not the launch from host code
                if not stripped.startswith('//'):
                    errors.append("Dynamic parallelism detected (kernel launching kernel from device code)")
                break
            if brace_depth == 0 and in_global:
                in_global = False
    
    # Check for cooperative groups
    if 'cooperative_groups' in kernel_source:
        errors.append("Cooperative groups not allowed in v1")
    
    # Check for persistent threads
    if 'while(1)' in kernel_source or 'while (1)' in kernel_source or 'for(;;)' in kernel_source:
        if '__syncthreads()' in kernel_source:
            warnings.append("Possible persistent thread block pattern detected")
    
    return len(errors) == 0, errors + warnings


if __name__ == '__main__':
    # Demo
    test_kernel = '''
__global__ void test_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f;
}

int main() {
    float* d_out;
    cudaMalloc(&d_out, 1024 * sizeof(float));
    test_kernel<<<4, 256>>>(d_out, 1024);
    cudaFree(d_out);
    return 0;
}
'''
    
    ok, issues = validate_kernel_source(test_kernel)
    print(f"Static validation: {'PASS' if ok else 'FAIL'}")
    for issue in issues:
        print(f"  {issue}")
    
    validator = StructuralValidator()
    result = validator.validate(test_kernel, [4,1,1], [256,1,1])
    print(f"\nPreflight: {'PASS' if result.success else 'FAIL'}")
    print(f"  Compile: {result.compile_ok}")
    print(f"  Run: {result.run_ok}")
    print(f"  Errors: {result.errors}")
    print(f"  Time: {result.wall_time_sec:.3f}s")