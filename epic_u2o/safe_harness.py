#!/usr/bin/env python3
"""Safe execution harness: process isolation + timeout for untrusted CUDA kernels.

Features:
- Process isolation with new session/process group
- Hard wall-clock timeout with SIGKILL on expiry (the real safety
  mechanism; there is intentionally NO background watchdog thread --
  see gpu-sim-u2o.6.1.1 for why the old GPUWatchdog always-tripped)
- Resource limits (VRAM, CPU, memory, disk)
- Single post-timeout GPU health check (SOFT_RESET only, never reboot)
- Structured logging for experiment tracking
- Real pre-flight validation: PTX syntax check + launch-config bounds +
  disallowed-construct scan + LD_PRELOAD smoke run of the synthesized
  driver (GPU-free)
"""

import subprocess
import sys
import time
import signal
import os
import json
import resource
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from epic_u2o.task_limits import (  # noqa: E402
    DEFAULT_TASKS_MAX, cgroup_preexec, choose_mechanism_preserving_kill,
    cleanup_cgroup, make_run_cgroup, rlimit_nproc_preexec, wrap_command,
)


def _chain_preexec(fns):
    """Apply several preexec hooks in order, as one callable."""
    fns = [f for f in fns if f]
    if len(fns) == 1:
        return fns[0]

    def _apply():
        for f in fns:
            f()
    return _apply


# Default configuration
DEFAULT_TIMEOUT_SEC = int(os.environ.get("CUDA_RUN_TIMEOUT", "30"))
DEFAULT_VRAM_LIMIT_MB = int(os.environ.get("CUDA_VRAM_LIMIT_MB", "24576"))  # 24 GB
DEFAULT_PREFLIGHT_TIMEOUT_SEC = int(os.environ.get("CUDA_PREFLIGHT_TIMEOUT", "60"))

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBFAKEGPU = REPO_ROOT / "libfakegpu.so"


@dataclass
class RunResult:
    """Result of an isolated kernel run."""
    success: bool
    returncode: int
    stdout: str
    stderr: str
    timeout: bool
    oom_killed: bool
    gpu_reset: bool
    wall_time_sec: float
    peak_rss_mb: float
    metadata: dict
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResourceLimits:
    """Resource limits for isolated execution."""
    vram_limit_mb: int = DEFAULT_VRAM_LIMIT_MB
    cpu_time_limit_sec: int = 60
    memory_limit_mb: int = 8192  # 8 GB system RAM
    disk_quota_mb: int = 1024  # 1 GB disk
    open_files_limit: int = 256
    # RLIMIT_NPROC. DO NOT lower this. The kernel checks RLIMIT_NPROC against
    # the real UID's TOTAL process+thread count (not just our subtree) on every
    # clone, so on a busy box (this one runs cclm/chessvar/conway jobs) even a
    # "modest" 32 makes the CUDA driver fail its first thread creation with
    # cudaErrorOsOperationNotSupported ("OS call failed or operation not
    # supported on this OS") at the first cudaMalloc. Measured on the 5090:
    # NOFILE=256 fine, NPROC=32 fatal, 512 still fatal (user already above it).
    # Runaway processes are contained by the process-group SIGKILL in
    # run_kernel()'s timeout path and, since gpu-sim-u2o.6.1.2, by a per-run
    # cgroup v2 pids.max cap (tasks_max below). RLIMIT_NPROC stays unset.
    processes_limit: int = 0  # 0 = leave RLIMIT_NPROC untouched

    # Per-run TASK cap via cgroup v2 pids.max (gpu-sim-u2o.6.1.2). This is the
    # containment that disabling RLIMIT_NPROC cost us. 0 disables it. The
    # default is a cap no kernel driver would notice and a fork bomb cannot
    # exceed. Enforcement is measured, not assumed -- see
    # tests/test_task_limits.py, where a child is observed failing to fork
    # with EAGAIN at the cap and succeeding without it.
    tasks_max: int = 256

    # Address-space ceiling. MUST exceed the GPU BAR1 mapping or CUDA
    # context creation fails instantly (u2o.6.1.1 post-mortem: RTX 5090
    # BAR1 is 32 GiB; a 24 GiB cap killed every real-GPU run in ~2ms).
    # Honest limitation: on ReBAR GPUs, RLIMIT_AS cannot serve as a VRAM
    # cap (the BAR mapping alone dwarfs it), so this is a leak backstop
    # only, not a VRAM accounting mechanism.
    address_space_limit_mb: int = 262144  # 256 GiB
    
    def apply(self):
        """Apply resource limits to current process."""
        # Virtual memory limit (address space) - leak backstop, NOT a VRAM
        # cap (see comment above re: 32 GiB BAR1 on the 5090).
        as_bytes = self.address_space_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        
        # CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_time_limit_sec, self.cpu_time_limit_sec))
        
        # Data segment limit (heap)
        mem_bytes = self.memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
        
        # Open files
        resource.setrlimit(resource.RLIMIT_NOFILE, (self.open_files_limit, self.open_files_limit))
        
        # Processes (skipped when processes_limit == 0; see the class comment --
        # any low RLIMIT_NPROC breaks the CUDA driver on a busy multi-tenant box)
        if self.processes_limit:
            resource.setrlimit(resource.RLIMIT_NPROC, (self.processes_limit, self.processes_limit))
        
        # Core dump size (disable)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _check_gpu_health_once(gpu_id: int = 0) -> tuple[str, bool]:
    """Single post-timeout GPU health check (no background thread).

    Returns (health_state, reset_attempted). Only a SOFT_RESET
    (nvidia-smi -r) is ever attempted automatically; anything worse is
    logged for the owner -- never driver reloads or reboots (no sudo
    automation on this box, per the execution plan).
    """
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from epic_u2o.gpu_recovery import GPURecovery
        rec = GPURecovery(gpu_id=gpu_id)
        health = rec.check_gpu_health()
        state = health.value if hasattr(health, 'value') else str(health)
    except Exception as e:
        return f"health-check-error: {e}", False

    if state in ('hung', 'xid_error', 'off_bus'):
        try:
            r = subprocess.run(
                ['nvidia-smi', '-i', str(gpu_id), '-r'],
                capture_output=True, timeout=30)
            ok = (r.returncode == 0)
            time.sleep(2)
            return f"{state} -> soft-reset {'ok' if ok else 'failed'}", ok
        except Exception as e:
            return f"{state} -> soft-reset error: {e}", False
    return state, False


class SafeHarness:
    """Main harness for safe kernel execution."""
    
    def __init__(
        self,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        vram_limit_mb: int = DEFAULT_VRAM_LIMIT_MB,
        gpu_id: int = 0,
        log_dir: Optional[str] = None,
        use_fakegpu_preflight: bool = True,
        tasks_max: int = DEFAULT_TASKS_MAX,
    ):
        self.timeout_sec = timeout_sec
        self.vram_limit_mb = vram_limit_mb
        self.gpu_id = gpu_id
        self.log_dir = Path(log_dir) if log_dir else None
        self.use_fakegpu_preflight = use_fakegpu_preflight
        # Per-run task containment (gpu-sim-u2o.6.1.2). 0 disables it, which
        # would leave only the process-group SIGKILL, and is NOT the default.
        self.tasks_max = tasks_max
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _log_run(self, result: RunResult, kernel_hash: str):
        """Log run result to file."""
        if not self.log_dir:
            return
        
        log_file = self.log_dir / f"run_{kernel_hash[:8]}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                'timestamp': time.time(),
                'kernel_hash': kernel_hash,
                **result.to_dict()
            }) + '\n')
    
    def preflight_check(self, kernel_source: str, grid: list[int], block: list[int],
                       shared_mem: int = 0, spec: Optional[dict] = None) -> tuple[bool, str]:
        """Real structural validation, GPU-free (gpu-sim-u2o.6.1.1).

        Always: launch-config bounds + disallowed-construct scan + nvcc -ptx
        syntax check. When a full KernelSpec is supplied, additionally:
        synthesize the driver, compile against real headers linked with
        libfakegpu, and run it under LD_PRELOAD (fake device, GPU-free),
        requiring exit 0. Reported timings from that smoke run are FAKE
        (configured latency) and never used as measurements.
        """
        if not self.use_fakegpu_preflight:
            return True, "Pre-flight skipped (disabled)"

        errors: list[str] = []

        # 1. Launch-config bounds (RTX 5090 hardware limits).
        threads = block[0] * block[1] * block[2]
        if threads < 1 or threads > 1024:
            errors.append(f"threads/block={threads} outside [1, 1024]")
        if not (1 <= block[0] <= 1024 and 1 <= block[1] <= 1024 and 1 <= block[2] <= 64):
            errors.append(f"block dims {block} exceed (1024, 1024, 64)")
        if grid[0] < 1 or grid[1] < 1 or grid[2] < 1:
            errors.append(f"grid dims {grid} must all be >= 1")
        if shared_mem < 0 or shared_mem > 228 * 1024:
            errors.append(f"shared_mem={shared_mem} exceeds 228KB/SM")

        # 2. Disallowed-construct scan.
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from epic_u2o.preflight_validator import validate_kernel_source
            ok, issues = validate_kernel_source(kernel_source)
            if not ok:
                errors.extend(issues)
        except Exception as e:
            errors.append(f"static scan error: {e}")

        # 3. nvcc -ptx syntax check (no link, no GPU).
        if 'int main' not in kernel_source:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(kernel_source)
                cu_file = f.name
            try:
                r = subprocess.run(
                    ['nvcc', '-ptx', '-arch=sm_120', cu_file, '-o', '/dev/null'],
                    capture_output=True, text=True, timeout=120)
                if r.returncode != 0:
                    errors.append(f"nvcc -ptx failed: {r.stderr.strip()[:500]}")
            except subprocess.TimeoutExpired:
                errors.append("nvcc -ptx timed out")
            except FileNotFoundError:
                errors.append("nvcc not found")
            finally:
                if os.path.exists(cu_file):
                    os.unlink(cu_file)
        # (sources with their own main() skip -ptx: real compile is the gate)

        if errors:
            return False, "Pre-flight FAILED: " + "; ".join(errors)

        # 4. LD_PRELOAD smoke run (requires full spec for driver synthesis).
        if spec is None:
            return True, ("Pre-flight passed (bounds+scan+ptx; LD_PRELOAD smoke "
                          "skipped: no arg spec)")
        return self._preflight_smoke_run(spec)

    def _preflight_smoke_run(self, spec: dict) -> tuple[bool, str]:
        """Compile synthesized driver (real headers) + run under LD_PRELOAD."""
        import tempfile
        sys.path.insert(0, str(REPO_ROOT))
        from epic_u2o.kernel_driver import synthesize

        try:
            program = synthesize(spec)
        except Exception as e:
            return False, f"Pre-flight FAILED: driver synthesis: {e}"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(program)
            cu_file = f.name
        exe_file = cu_file + '.smoke.out'
        try:
            r = subprocess.run(
                ['nvcc', '-O2', cu_file, '-o', exe_file,
                 f'-L{REPO_ROOT}', '-lfakegpu',
                 '-Xlinker', '-rpath', '-Xlinker', str(REPO_ROOT)],
                capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                return False, f"Pre-flight FAILED: smoke compile: {r.stderr.strip()[:500]}"
            env = os.environ.copy()
            env['LD_PRELOAD'] = str(LIBFAKEGPU) + (
                f":{env['LD_PRELOAD']}" if env.get('LD_PRELOAD') else '')
            env['LD_LIBRARY_PATH'] = str(REPO_ROOT) + (
                f":{env['LD_LIBRARY_PATH']}" if env.get('LD_LIBRARY_PATH') else '')
            env['FAKEGPU_KERNEL_LATENCY_MS'] = '0.5'
            r = subprocess.run(
                [exe_file], capture_output=True, text=True, timeout=60, env=env,
                start_new_session=True)
            if r.returncode != 0:
                return False, (f"Pre-flight FAILED: smoke run rc={r.returncode}: "
                               f"{r.stderr.strip()[:500]}")
            lines = [ln for ln in r.stdout.splitlines() if ln.strip().startswith('{')]
            if not lines:
                return False, "Pre-flight FAILED: smoke run printed no trial JSON"
            return True, (f"Pre-flight passed (bounds+scan+ptx+smoke: "
                          f"{len(lines)} trial lines, exit 0)")
        except subprocess.TimeoutExpired:
            return False, "Pre-flight FAILED: smoke run timed out"
        except FileNotFoundError:
            return False, "Pre-flight FAILED: nvcc not found"
        finally:
            for p in (cu_file, exe_file):
                if os.path.exists(p):
                    os.unlink(p)
    
    def compile_kernel(self, kernel_source: str, output_path: str, arch: str = 'sm_90') -> tuple[bool, str]:
        """Compile CUDA kernel to executable."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_source)
            cu_file = f.name
        
        try:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            
            result = subprocess.run(
                ['nvcc', f'-arch={arch}', cu_file, '-o', output_path],
                capture_output=True, text=True, timeout=60, env=env
            )
            
            if result.returncode != 0:
                return False, result.stderr
            
            return True, "Compilation successful"
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except Exception as e:
            return False, f"Compilation error: {e}"
        finally:
            if os.path.exists(cu_file):
                os.unlink(cu_file)
    
    def run_kernel(
        self,
        executable_path: str,
        args: list[str] = None,
        env: dict = None,
    ) -> RunResult:
        """Run compiled kernel with isolation and timeout."""
        args = args or []
        env = env or {}
        
        # Build environment
        run_env = os.environ.copy()
        run_env.update(env)
        run_env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        run_env['FAKEGPU_DEVICE_MEM_MB'] = str(self.vram_limit_mb)
        
        cmd = [executable_path] + args

        # Per-run task containment (gpu-sim-u2o.6.1.2). The command is wrapped
        # here rather than inside the child because systemd-run must be the
        # process we supervise: --wait keeps our communicate(timeout=)
        # authoritative and --pipe keeps stdout/stderr on our pipes.
        # Per-run task containment. The MECHANISM is chosen under a hard
        # constraint discovered by review: the timeout backstop must keep
        # working. `systemd-run --user --wait` starts a transient SERVICE, so
        # the wrapped binary's parent is systemd and it is NOT in the child's
        # process group -- proc.kill() + killpg() then reach only systemd-run
        # and NOT the runaway. Measured on this host: run_kernel reported
        # timeout=True, reported the health check, reported a containment, and
        # the child kept running (heartbeat still growing a second after the
        # kill). --collect does not help; it collects after the unit FINISHES.
        #
        # So we do not wrap the command at all. The child stays a direct child
        # -- which is what keeps the existing process-group SIGKILL correct --
        # and is moved into a per-run cgroup by a preexec hook, which runs
        # after fork and before exec. The cap is therefore in force before the
        # child executes a single instruction, and there is no window in which
        # it could fork first.
        task_info = {"tasks_max": self.tasks_max, "mechanism": "none"}
        cgroup_dir = None
        preexec_parts = []
        if self.tasks_max and self.tasks_max > 0:
            mech = choose_mechanism_preserving_kill()
            task_info["mechanism"] = mech.name
            task_info["mechanism_detail"] = mech.detail
            task_info["systemd_run_not_used_because"] = (
                "systemd-run --wait detaches the child from our process group, "
                "which silently disables the timeout kill; see task_limits.py")
            if mech.name == "cgroup":
                try:
                    cgroup_dir = make_run_cgroup(tasks_max=self.tasks_max)
                    preexec_parts.append(cgroup_preexec(cgroup_dir))
                    task_info["cgroup"] = str(cgroup_dir)
                except Exception as e:
                    # Fall back rather than lose the cap silently.
                    task_info["cgroup_error"] = str(e)
                    mech = choose_mechanism_preserving_kill("rlimit_nproc")
                    task_info["mechanism"] = mech.name
            if mech.name == "rlimit_nproc":
                fn = rlimit_nproc_preexec()
                preexec_parts.append(fn)
                task_info["nproc_relative"] = fn.target_nproc
        
        start_time = time.time()
        peak_rss = 0
        
        # Start process. INSIDE the try, deliberately: Popen is where a bad path
        # or an exec error raises, and with it outside, the finally that removes
        # the per-run cgroup never runs. That is precisely how the cgroup leak
        # survived: cleanup fired on the success and timeout paths and not on
        # the launch-failure path, which for a compile-then-run harness is
        # routine.
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Isolate in new session
                env=run_env,
                preexec_fn=_chain_preexec([
                    ResourceLimits(vram_limit_mb=self.vram_limit_mb).apply]
                    + preexec_parts),
                text=True,  # Get strings instead of bytes
            )
            # Wait with timeout
            stdout, stderr = proc.communicate(timeout=self.timeout_sec)
            wall_time = time.time() - start_time
            
            # Get peak memory usage
            try:
                peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024  # MB on Linux
            except Exception:
                peak_rss = 0
            
            # Check for OOM kill
            oom_killed = proc.returncode == -9 or (stderr and 'out of memory' in stderr.lower())
            
            result = RunResult(
                success=proc.returncode == 0,
                returncode=proc.returncode,
                stdout=stdout if stdout else '',
                stderr=stderr if stderr else '',
                timeout=False,
                oom_killed=oom_killed,
                gpu_reset=False,
                wall_time_sec=wall_time,
                peak_rss_mb=peak_rss,
                metadata={'cmd': ' '.join(cmd), 'task_limits': task_info},
            )
            
        except subprocess.TimeoutExpired:
            # Timeout - kill process group
            proc.kill()
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            
            wall_time = time.time() - start_time
            
            # Single post-timeout health check (no background thread).
            health, gpu_reset = _check_gpu_health_once(self.gpu_id)
            
            result = RunResult(
                success=False,
                returncode=-1,
                stdout='',
                stderr=f'TIMEOUT/HANG DETECTED (gpu health: {health})',
                timeout=True,
                oom_killed=False,
                gpu_reset=gpu_reset,
                wall_time_sec=wall_time,
                peak_rss_mb=peak_rss,
                metadata={'cmd': ' '.join(cmd), 'gpu_health': health,
                           'task_limits': task_info},
            )
        
        except Exception as e:
            wall_time = time.time() - start_time
            result = RunResult(
                success=False,
                returncode=-1,
                stdout='',
                stderr=f'Execution error: {e}',
                timeout=False,
                oom_killed=False,
                gpu_reset=False,
                wall_time_sec=wall_time,
                peak_rss_mb=peak_rss,
                metadata={'cmd': ' '.join(cmd), 'task_limits': task_info},
            )
        finally:
            # Remove the per-run cgroup on EVERY path -- success, timeout, and
            # exception. The reviewer's measurement: 12 cgroups before, 14 after
            # three clean run_kernel calls, so cleanup fired on some paths and
            # not others. Every leaked cgroup is inert (procs empty, pids 0), so
            # nothing is running in them and no containment is affected -- but
            # they accumulate in the user's own slice for the life of the
            # session, and this harness is about to be driven in batch: the 6.6.2
            # anchor alone is 94 configs x 3 invocations = 282 run_kernel calls,
            # which is a few hundred empty directories that never go away. That
            # is exactly the "cgroup lifetime management of our own" the design
            # flagged when it ranked this route second.
            if cgroup_dir is not None:
                cleanup_cgroup(cgroup_dir)
                task_info["cgroup_cleaned"] = not cgroup_dir.exists()

        return result
    
    def execute(
        self,
        kernel_source: str,
        kernel_hash: str,
        grid: list[int],
        block: list[int],
        shared_mem: int = 0,
        arch: str = 'sm_120',
        spec: Optional[dict] = None,
    ) -> RunResult:
        """Full execution pipeline: preflight -> compile -> run.

        kernel_source must be a COMPLETE program here (with main()); use
        measure_spec() for bare KernelSpec dicts (it synthesizes the driver
        first). Default arch is sm_120 (RTX 5090).
        """
        import tempfile
        
        # Pre-flight validation
        preflight_ok, preflight_msg = self.preflight_check(
            kernel_source, grid, block, shared_mem, spec=spec)
        if not preflight_ok:
            return RunResult(
                success=False,
                returncode=-1,
                stdout='',
                stderr=f'Pre-flight failed: {preflight_msg}',
                timeout=False,
                oom_killed=False,
                gpu_reset=False,
                wall_time_sec=0,
                peak_rss_mb=0,
                metadata={'preflight_error': preflight_msg},
            )
        
        # Compile
        with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as f:
            exec_path = f.name
        
        try:
            compile_ok, compile_msg = self.compile_kernel(kernel_source, exec_path, arch)
            if not compile_ok:
                return RunResult(
                    success=False,
                    returncode=-1,
                    stdout='',
                    stderr=f'Compilation failed: {compile_msg}',
                    timeout=False,
                    oom_killed=False,
                    gpu_reset=False,
                    wall_time_sec=0,
                    peak_rss_mb=0,
                    metadata={'compile_error': compile_msg},
                )
            
            # Run
            result = self.run_kernel(exec_path)
            result.metadata['kernel_hash'] = kernel_hash
            result.metadata['grid'] = grid
            result.metadata['block'] = block
            result.metadata['shared_mem'] = shared_mem
            
            # Log
            self._log_run(result, kernel_hash)
            
            return result
        
        finally:
            if os.path.exists(exec_path):
                os.unlink(exec_path)
    
    def measure_spec(self, spec: dict, arch: str = 'sm_120') -> tuple[list[float], RunResult]:
        """generate_kernels() -> synthesizer -> compile -> run pipeline.

        Takes a KernelSpec dict, synthesizes a full driver program
        (kernel_driver.synthesize), compiles it with real nvcc, runs it,
        and parses the per-trial JSON timings from stdout. Returns
        (trials_ms, RunResult). Raises RuntimeError on preflight/compile
        failure so batch pipelines fail loudly.
        """
        import tempfile
        sys.path.insert(0, str(REPO_ROOT))
        from epic_u2o.kernel_driver import synthesize
        
        for key in ('source', 'kernel_name', 'grid', 'block', 'args', 'problem_size'):
            if key not in spec:
                raise RuntimeError(f"measure_spec: spec missing {key}")
        grid, block = spec['grid'], spec['block']
        shmem = int(spec.get('shared_mem_bytes', 0))
        khash_src = spec['source'] + json.dumps([grid, block, shmem])
        import hashlib
        khash = hashlib.sha256(khash_src.encode()).hexdigest()[:16]
        
        program = synthesize(spec)  # raises on unsupported specs
        
        with tempfile.NamedTemporaryFile(suffix='.meas.out', delete=False) as f:
            exec_path = f.name
        try:
            ok, msg = self.preflight_check(
                spec['source'], grid, block, shmem, spec=spec)
            if not ok:
                raise RuntimeError(msg)
            compile_ok, compile_msg = self.compile_kernel(program, exec_path, arch)
            if not compile_ok:
                raise RuntimeError(f"Compilation failed: {compile_msg}")
            result = self.run_kernel(exec_path)
            result.metadata.update({'kernel_hash': khash, 'grid': grid,
                                    'block': block, 'shared_mem': shmem,
                                    'op_type': spec.get('op_type', '?')})
            self._log_run(result, khash)
            if not result.success:
                return [], result
            trials = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line.startswith('{'):
                    continue
                try:
                    trials.append(float(json.loads(line)['time_ms']))
                except (ValueError, KeyError):
                    continue
            return trials, result
        finally:
            if os.path.exists(exec_path):
                os.unlink(exec_path)


def run_isolated(cmd: str, timeout: int = DEFAULT_TIMEOUT_SEC) -> tuple[bool, str, str]:
    """Legacy simple interface for backward compatibility."""
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode == 0, out.decode(), err.decode()
    except subprocess.TimeoutExpired:
        proc.kill()
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return False, "", "TIMEOUT/HANG DETECTED"


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Safe CUDA kernel execution harness')
    parser.add_argument('--kernel-file', type=str, help='Path to kernel .cu file')
    parser.add_argument('--spec-json', type=str,
                        help='Path to KernelSpec JSON file (full pipeline: synthesize->compile->run->trials)')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT_SEC, help='Timeout in seconds')
    parser.add_argument('--vram-mb', type=int, default=DEFAULT_VRAM_LIMIT_MB, help='VRAM limit in MB')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--log-dir', type=str, help='Log directory')
    parser.add_argument('--test-hang', action='store_true', help='Test with hanging kernel')
    args = parser.parse_args()
    
    if args.spec_json:
        with open(args.spec_json) as f:
            spec = json.load(f)
        harness = SafeHarness(timeout_sec=args.timeout, vram_limit_mb=args.vram_mb,
                              gpu_id=args.gpu, log_dir=args.log_dir)
        try:
            trials, result = harness.measure_spec(spec)
        except RuntimeError as e:
            print(json.dumps({'success': False, 'error': str(e)}))
            raise SystemExit(1)
        print(json.dumps({'success': result.success, 'trials_ms': trials,
                          'n_trials': len(trials),
                          'wall_time_sec': result.wall_time_sec,
                          'timeout': result.timeout,
                          'metadata': result.metadata}, indent=2))
        raise SystemExit(0 if result.success and trials else 1)
    
    if args.test_hang:
        # Test with a hanging kernel. NOTE: a bare `while (1) {}` does NOT
        # hang -- nvcc deletes it under forward-progress rules (observed:
        # exits 0 in ~2ms). The volatile spin below is unoptimizable and
        # genuinely wedges until killed. Supervised use only: this spins
        # on the display GPU until the harness SIGKILLs it.
        test_kernel = '''
__global__ void hang_kernel(volatile int *flag) {
    while (*flag == 0) { /* spin forever; host never sets the flag */ }
}
int main() {
    int *d_flag = NULL;
    cudaMalloc((void**)&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));
    hang_kernel<<<1,1>>>((volatile int*)d_flag);
    cudaDeviceSynchronize();
    return 0;
}
'''
        harness = SafeHarness(timeout_sec=5, vram_limit_mb=args.vram_mb, gpu_id=args.gpu, log_dir=args.log_dir)
        result = harness.execute(test_kernel, 'test_hang', [1,1,1], [1,1,1])
        print(json.dumps(result.to_dict(), indent=2))
    elif args.kernel_file:
        with open(args.kernel_file) as f:
            kernel = f.read()
        harness = SafeHarness(timeout_sec=args.timeout, vram_limit_mb=args.vram_mb, gpu_id=args.gpu, log_dir=args.log_dir)
        import hashlib
        khash = hashlib.sha256(kernel.encode()).hexdigest()[:16]
        result = harness.execute(kernel, khash, [1,1,1], [256,1,1])
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("Usage: python safe_harness.py --kernel-file <file.cu> [--timeout 30] [--vram-mb 24576] [--gpu 0] [--log-dir runs/logs]")
        print("       python safe_harness.py --spec-json <spec.json> [--timeout 30] [--gpu 0] [--log-dir runs/logs]")
        print("       python safe_harness.py --test-hang [--timeout 5] [--vram-mb 24576] [--gpu 0] [--log-dir runs/logs]")