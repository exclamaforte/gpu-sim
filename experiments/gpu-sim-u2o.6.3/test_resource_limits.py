#!/usr/bin/env python3
"""CPU-only tests for gpu-sim-u2o.6.3 resource budgets.

No GPU, no nvcc, no torch. Run: python3 experiments/gpu-sim-u2o.6.3/test_resource_limits.py
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from epic_u2o.resource_limits import (  # noqa: E402
    CONTEXT_RESERVE_MB, KILL_RULES, ResourcePolicy, ScratchDir,
    check_vram_budget, estimate_device_bytes, kill_plan_md,
)

FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def spec(n, args=None, kernel_name="k"):
    return {
        'source': f'__global__ void {kernel_name}(const float* a, float* b, int n) {{}}',
        'kernel_name': kernel_name,
        'grid': [1, 1, 1], 'block': [32, 1, 1],
        'args': args if args is not None else [
            {'name': 'a', 'type': 'const float*', 'ptr': True},
            {'name': 'b', 'type': 'float*', 'ptr': True},
            {'name': 'n', 'type': 'int', 'ptr': False},
        ],
        'problem_size': n,
    }


# --- 1. static footprint estimation -------------------------------------
est = estimate_device_bytes(spec(1 << 20))
check("estimate: 1Mi floats x2 = 8 MiB",
      est['total_bytes'] == (1 << 20) * 4 * 2, str(est['total_bytes']))
check("estimate: scalars excluded", 'n' not in est['per_arg'])
check("estimate: no unknown types", est['unknown'] == [])

est_d = estimate_device_bytes(spec(1 << 20, args=[
    {'name': 'a', 'type': 'const double*', 'ptr': True},
    {'name': 'n', 'type': 'int', 'ptr': False}]))
check("estimate: double is 8 bytes", est_d['total_bytes'] == (1 << 20) * 8)

est_u = estimate_device_bytes(spec(16, args=[
    {'name': 'x', 'type': 'complex_t*', 'ptr': True}]))
check("estimate: unknown type is reported, not guessed",
      est_u['unknown'] == ['x:complex_t*'] and est_u['total_bytes'] == 0)

# --- 2. VRAM budget ------------------------------------------------------
pol = ResourcePolicy(vram_limit_mb=2048)
ok, msg = check_vram_budget(spec(1 << 18), pol)     # 2 MiB of buffers
check("budget: small spec passes", ok, msg)
need = (1 << 18) * 8 / 1048576 + CONTEXT_RESERVE_MB
check("budget: message quotes the need", f"{need:.1f} MB" in msg, msg)

huge = spec(1 << 30)                                # 1 Gi floats x2 = 8 GiB
ok2, msg2 = check_vram_budget(huge, pol)
check("budget: oversized spec refused", not ok2 and "exceeded" in msg2, msg2)

ok3, msg3 = check_vram_budget(spec(1 << 20), ResourcePolicy(vram_limit_mb=0))
check("budget: limit<=0 disables the cap", ok3 and "disabled" in msg3)

ok4, msg4 = check_vram_budget(est_u and spec(16, args=[
    {'name': 'x', 'type': 'mystery_t*', 'ptr': True}]), pol)
check("budget: unsizable args refused (no silent under-count)",
      not ok4 and "cannot size" in msg4, msg4)

# --- 3. disk quota accounting ------------------------------------------
with tempfile.TemporaryDirectory() as td:
    base = Path(td) / "scratch"
    base.mkdir()
    s = ScratchDir(base, quota_mb=1, name='t')
    with s:
        (s.path / "a.bin").write_bytes(b"x" * 512 * 1024)   # 0.5 MB
        ok, m = s.check_quota()
        check("disk: under quota ok", ok, m)
        (s.path / "b.bin").write_bytes(b"x" * 1024 * 1024)   # now > 1 MB
        ok, m = s.check_quota()
        check("disk: over quota detected", (not ok) and len(s.violations) == 1, m)
        check("disk: size accounted", abs(s.size_mb() - 1.5) < 0.01, f"{s.size_mb():.3f} MB")
        check("disk: remaining accounting", abs(s.remaining_mb() + 0.5) < 0.01)
    check("disk: scratch removed on exit", not s.path.exists())

    # base tree already over quota -> refuse to start a new run
    s2 = ScratchDir(base, quota_mb=0, name='t2')
    (base / "leftover.bin").write_bytes(b"y" * 4096)
    try:
        s2.__enter__()
        check("disk: refuses when base over quota", False, "no RuntimeError")
    except RuntimeError as e:
        check("disk: refuses when base over quota", "quota" in str(e), str(e))

# --- 4. RLIMIT_FSIZE backstop (real subprocess, no GPU) -----------------
with tempfile.TemporaryDirectory() as td:
    s = ScratchDir(td, quota_mb=1, name='fs')
    with s:
        r = subprocess.run(
            [sys.executable, "-c",
             "open('big.bin','wb').write(b'x'*(8*1024*1024))"],
            cwd=str(s.path), capture_output=True, text=True,
            preexec_fn=s.preexec(file_size_limit_mb=1), timeout=60)
        # killed by SIGXFSZ (-25) or EFBIG (rc 1 with OSError)
        grew_over = (s.path / "big.bin").stat().st_size if (s.path / "big.bin").exists() else 0
        check("disk: RLIMIT_FSIZE stops an 8 MB write at 1 MB",
              r.returncode != 0 and grew_over <= 1 * 1024 * 1024,
              f"rc={r.returncode} size={grew_over}")

# --- 5. runaway kill rule table -----------------------------------------
check("kill: tier 0 is process-group SIGKILL on timeout",
      "SIGKILL" in KILL_RULES[0].action and "process group" in KILL_RULES[0].action)
check("kill: only SOFT_RESET is automatic",
      "SOFT_RESET" in KILL_RULES[1].action and "nvidia-smi -r" in KILL_RULES[1].action)
check("kill: no automatic driver reload / reboot",
      not any("DRIVER_RELOAD" in r.action or "reboot" in r.action.lower()
              for r in KILL_RULES))
check("kill: final tier stops and escalates to a human",
      "STOP" in KILL_RULES[-1].action)
check("kill: policy never lowers RLIMIT_NPROC",
      ResourcePolicy().processes_limit == 0)
check("kill: markdown renderer works",
      kill_plan_md().count("\n") == len(KILL_RULES) + 1 and "| tier |" in kill_plan_md())

# --- 6. harness integration (no GPU: nothing is launched) ---------------
from epic_u2o.safe_harness import SafeHarness, ResourceLimits  # noqa: E402
h = SafeHarness(timeout_sec=5, vram_limit_mb=2048)
check("harness: NPROC limit is 0 (not 32)", ResourceLimits().processes_limit == 0)
check("harness: NPROC rlimit left untouched by default",
      ResourceLimits().processes_limit == 0)
# preflight_check returns (ok, msg) rather than raising -- assert the refusal
ok, msg = h.preflight_check("__global__ void k(){}", [1, 1, 1], [32, 1, 1],
                            0, spec=spec(1 << 30))
check("harness: preflight refuses oversized spec (returns FAILED)",
      (not ok) and "VRAM budget exceeded" in msg, msg[:160])

print()
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
