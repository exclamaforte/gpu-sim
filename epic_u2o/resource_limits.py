#!/usr/bin/env python3
"""Per-example resource budgets: VRAM, disk, and runaway-kill rules
(gpu-sim-u2o.6.3).

Everything in here is deliberately *host-side*. The first real end-to-end run
of the harness (gpu-sim-u2o.6.1.1, ts job 20260927t145454-9c2420b3) plus its
bisect (ts job 20260927t123630-05fd9bd0) produced three measured facts that
this module is built around:

1. RLIMIT_AS CANNOT be a VRAM cap on this box. The RTX 5090 is a ReBAR part
   whose 32 GiB BAR1 mapping dwarfs any address-space cap you would like to
   set, so RLIMIT_AS is only a host-address-space leak backstop. Measured:
   AS=256 GiB works, 24 GiB killed every run at context creation.
2. RLIMIT_NPROC MUST NOT be lowered. The kernel checks it against the real
   UID's TOTAL process+thread count, not the child's subtree, so on this
   shared machine NPROC=32 (and 512) made the CUDA driver fail its first
   thread creation -- surfacing as "OS call failed or operation not
   supported on this OS" at the first cudaMalloc. A too-tight limit is not
   a safety feature, it is an outage.
3. Therefore a VRAM cap has to be enforced by (a) a static footprint check on
   the spec before the kernel is ever launched, and (b) a device-side
   assertion the kernel driver itself performs (cudaMemGetInfo) so an
   in-kernel allocation cannot slip past the static estimate.

Disk is bounded by a per-run scratch directory with an accounted quota plus
an RLIMIT_FSIZE backstop (per-file, not per-directory -- stated plainly
because pretending otherwise is how quota bugs get shipped).

Runaway processes are killed by the process group on timeout, followed by a
single GPU health check. No driver reload, no reboot, ever automatically.
"""

import os
import resource
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

MB = 1024 * 1024

# sizeof() by CUDA base type; mirrors kernel_driver._BASE_SIZES.
_TYPE_SIZES = {
    'char': 1, 'short': 2, 'int': 4, 'unsigned int': 4, 'long': 8,
    'float': 4, 'double': 8, 'half': 2,
}

# Head-room reserved for the CUDA context itself (measured ~300-500 MB on
# the 5090 driver; 1 GiB is deliberately conservative for a 32 GiB card).
CONTEXT_RESERVE_MB = 1024


@dataclass
class ResourcePolicy:
    """Per-example budget. Defaults are the values safe_harness uses."""
    vram_limit_mb: int = 24576
    disk_quota_mb: int = 1024
    file_size_limit_mb: int = 256      # RLIMIT_FSIZE backstop, per file
    timeout_sec: int = 30
    cpu_time_limit_sec: int = 60
    memory_limit_mb: int = 8192        # RLIMIT_DATA (host heap)
    address_space_limit_mb: int = 262144
    open_files_limit: int = 256
    # 0 == never touch RLIMIT_NPROC. See module docstring, fact 2.
    processes_limit: int = 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _base_type(ctype: str) -> str:
    return ctype.replace('const', '').replace('__restrict__', '').strip().rstrip('*').strip()


def estimate_device_bytes(spec: dict) -> dict:
    """Static device-memory footprint of a KernelSpec.

    Returns {'per_arg': {name: bytes}, 'total_bytes': int, 'unknown': [names]}.
    Unknown base types are reported rather than guessed -- a silent
    under-estimate is exactly how a VRAM cap gets bypassed.
    """
    n = int(spec['problem_size'])
    per_arg, unknown = {}, []
    total = 0
    for a in spec.get('args', []):
        if '*' not in a['type']:
            continue                      # scalars live in constant/param space
        base = _base_type(a['type'])
        size = _TYPE_SIZES.get(base)
        if size is None:
            unknown.append(f"{a['name']}:{a['type']}")
            continue
        b = n * size
        per_arg[a['name']] = b
        total += b
    return {'per_arg': per_arg, 'total_bytes': total, 'unknown': unknown}


def check_vram_budget(spec: dict, policy: ResourcePolicy) -> tuple[bool, str]:
    """Static VRAM budget check. Raises nothing; returns (ok, message)."""
    est = estimate_device_bytes(spec)
    if est['unknown']:
        return False, (f"VRAM budget: cannot size argument types "
                       f"{est['unknown']}; refusing to guess")
    need_mb = (est['total_bytes'] / MB) + CONTEXT_RESERVE_MB
    limit_mb = policy.vram_limit_mb
    if limit_mb <= 0:
        return True, "VRAM budget disabled (limit <= 0)"
    if need_mb > limit_mb:
        return False, (f"VRAM budget exceeded: need {need_mb:.1f} MB "
                       f"(allocations {est['total_bytes'] / MB:.1f} MB + "
                       f"{CONTEXT_RESERVE_MB} MB context reserve), "
                       f"limit {limit_mb} MB")
    return True, (f"VRAM budget ok: need {need_mb:.1f} MB of {limit_mb} MB "
                  f"({est['total_bytes'] / MB:.1f} MB across "
                  f"{len(est['per_arg'])} buffers)")


class ScratchDir:
    """Per-run scratch directory with an accounted disk quota.

    * refuses to hand out a directory when the base tree is already over
      quota (so one runaway example cannot starve the next ones),
    * accounts total bytes written under the run directory,
    * exposes `preexec()` which adds an RLIMIT_FSIZE backstop for the child,
    * removes everything on exit (pass keep=True to preserve for forensics).
    """

    def __init__(self, base: str | Path, quota_mb: int = 1024,
                 name: str = 'run', keep: bool = False):
        self.base = Path(base)
        self.quota_mb = quota_mb
        self.keep = keep
        self.path = self.base / f"{name}-{int(time.time())}-{os.getpid()}"
        self.violations: list[str] = []

    # -- accounting -----------------------------------------------------
    @staticmethod
    def dir_size_bytes(path: Path) -> int:
        total = 0
        for p in Path(path).rglob('*'):
            if p.is_file() and not p.is_symlink():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def size_mb(self) -> float:
        return self.dir_size_bytes(self.path) / MB if self.path.exists() else 0.0

    def remaining_mb(self) -> float:
        return self.quota_mb - self.size_mb()

    def preflight(self) -> tuple[bool, str]:
        used = self.dir_size_bytes(self.base) / MB if self.base.exists() else 0.0
        if used > self.quota_mb:
            return False, (f"disk quota already exceeded in {self.base}: "
                           f"{used:.1f} MB > {self.quota_mb} MB")
        return True, f"disk budget ok: {used:.1f} MB used of {self.quota_mb} MB in {self.base}"

    def check_quota(self) -> tuple[bool, str]:
        used = self.size_mb()
        if used > self.quota_mb:
            msg = f"disk quota exceeded: {used:.1f} MB > {self.quota_mb} MB in {self.path}"
            self.violations.append(msg)
            return False, msg
        return True, f"disk quota ok: {used:.1f} MB of {self.quota_mb} MB"

    # -- child limits ---------------------------------------------------
    def preexec(self, extra_cpu_sec: int | None = None,
                file_size_limit_mb: int = 256):
        """preexec_fn for Popen: RLIMIT_FSIZE backstop (no NPROC, no AS)."""
        fs = max(1, min(file_size_limit_mb, self.quota_mb)) * MB

        def _apply():
            resource.setrlimit(resource.RLIMIT_FSIZE, (fs, fs))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if extra_cpu_sec:
                resource.setrlimit(resource.RLIMIT_CPU, (extra_cpu_sec, extra_cpu_sec))
        return _apply

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> 'ScratchDir':
        ok, msg = self.preflight()
        if not ok:
            raise RuntimeError(msg)
        self.path.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.keep:
            shutil.rmtree(self.path, ignore_errors=True)
        return False


@dataclass(frozen=True)
class KillRule:
    tier: int
    trigger: str
    action: str
    rationale: str


# The runaway ladder. Ordered; the harness only ever reaches tier 0-1
# automatically. Everything at tier >= 2 needs a human.
KILL_RULES: tuple[KillRule, ...] = (
    KillRule(0, "wall-clock timeout (communicate(timeout=...))",
             "SIGKILL the whole process group (start_new_session=True)",
             "covers infinite loops, device-side hangs, deadlocks; the child "
             "cannot catch or ignore SIGKILL"),
    KillRule(1, "after any timeout",
             "one GPU health check; SOFT_RESET (nvidia-smi -r) only if hung/xid/off-bus",
             "a killed kernel can leave the context wedged; no passwordless "
             "sudo on this box, so DRIVER_RELOAD/SYSTEM_REBOOT are never "
             "automatic (see the execution plan, u2o.18)"),
    KillRule(2, "host memory / disk exhaustion",
             "RLIMIT_DATA (host heap) + per-run disk quota + RLIMIT_FSIZE",
             "bounds blast radius without touching limits the CUDA driver "
             "cannot tolerate (see facts 1-2 in the module docstring)"),
    KillRule(3, "GPU unusable after SOFT_RESET",
             "STOP. Report to the owner. Never auto-recover further.",
             "the 5090 also drives the display; an unattended reboot/reload "
             "is worse than a stopped batch"),
)


def kill_plan_md() -> str:
    """Human-readable kill ladder (docs and run logs)."""
    lines = ["| tier | trigger | action | why |", "|---|---|---|---|"]
    lines += [f"| {r.tier} | {r.trigger} | {r.action} | {r.rationale} |"
              for r in KILL_RULES]
    return "\n".join(lines)
