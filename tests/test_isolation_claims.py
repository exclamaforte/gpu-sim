#!/usr/bin/env python3
"""Executable assertions for every GPU-free / CPU-only / no-GPU isolation
claim in epic_u2o/*.py + shim/*.h (gpu-sim-u2o.6.5.1). CPU-only.

Each test names the inventory row it backs (see
experiments/gpu-sim-u2o.6.5.1/claims_inventory.md). A device-free claim must
not be manufactured by hiding the GPU: where the mechanism allows, the device
is made HOSTILE (CUDA_VISIBLE_DEVICES=nonexistent-device) rather than hidden,
and a PATH-shadowed nvidia-smi tripwire fails the test if anything shells out
to it.

Usage:
  CUDA_VISIBLE_DEVICES= python3 tests/test_isolation_claims.py
Exits nonzero on any failure.
"""
import ast
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO))

FAILS = []
HOSTILE_ENV = dict(os.environ)
HOSTILE_ENV["CUDA_VISIBLE_DEVICES"] = "nonexistent-device-651"


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail)[:200] if detail else ''}",
          flush=True)
    if not ok:
        FAILS.append(name)


def tripwire_path(tmp):
    """A fake nvidia-smi that records invocations; prepend its dir to PATH."""
    d = tmp / "shadowbin"
    d.mkdir()
    log = tmp / "smi_calls.log"
    (d / "nvidia-smi").write_text(
        f"#!/usr/bin/env bash\necho \"$@\" >> {log}\nexit 3\n")
    (d / "nvidia-smi").chmod(0o755)
    env = dict(HOSTILE_ENV)
    env["PATH"] = f"{d}{os.pathsep}{env.get('PATH', '')}"
    return env, log


def one_spec_dict():
    from epic_u2o.generator_stage1 import generate_kernels
    return list(generate_kernels(ops=["saxpy"], max_kernels=1))[0].to_dict()


# Row 1/2: BUILD phase performs no GPU contact (compile only, never executes).
def test_build_needs_no_device_and_shells_nothing():
    from epic_u2o.kernel_driver import synthesize
    with tempfile.TemporaryDirectory(prefix="u651-build-") as td:
        tmp = Path(td)
        env, smi_log = tripwire_path(tmp)
        d = one_spec_dict()
        cu = tmp / "k.cu"
        exe = tmp / "k.out"
        cu.write_text(synthesize(d))
        # The exact BUILD mechanism: one nvcc invocation, file I/O, no exec.
        r = subprocess.run(["nvcc", "-arch=sm_120", "-O2", str(cu), "-o", str(exe)],
                           capture_output=True, text=True, timeout=300, env=env)
        check("build.compile_succeeds_hostile_env", r.returncode == 0,
              r.stderr[:200])
        check("build.binary_produced", exe.exists())
        check("build.no_nvidia_smi_contact", not smi_log.exists(),
              "build shelled out to nvidia-smi" if smi_log.exists() else "")
        # Structural pin (house precedent: test_vram_hunk_present.py): the
        # BUILD function may only ever spawn nvcc -- produced exes are written,
        # never executed, in this phase.
        src = (REPO / "epic_u2o" / "batch_measure.py").read_text()
        tree = ast.parse(src)
        build_fn = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == "build_batch")
        spawns = []
        for node in ast.walk(build_fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("run", "Popen", "call", "check_call")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"):
                argv = node.args[0]
                if isinstance(argv, ast.List) and argv.elts:
                    first = argv.elts[0]
                    spawns.append(first.value if isinstance(first, ast.Constant) else "?")
                else:
                    spawns.append("?")
        check("build.only_spawns_nvcc",
              bool(spawns) and all(s == "nvcc" for s in spawns), str(spawns))


# Row 3: device_snapshot is a read-only query (touches nvidia-smi by design,
# never a reset/workload). Proved by argv capture, not by prose.
def test_device_snapshot_is_read_only():
    from epic_u2o.batch_measure import device_snapshot
    with tempfile.TemporaryDirectory(prefix="u651-snap-") as td:
        tmp = Path(td)
        env, smi_log = tripwire_path(tmp)
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = env["PATH"]
        try:
            snap = device_snapshot()
        finally:
            os.environ["PATH"] = old_path
        check("snapshot.calls_nvidia_smi", smi_log.exists())
        argv = smi_log.read_text().strip() if smi_log.exists() else ""
        check("snapshot.query_only", "--query-gpu" in argv, argv[:200])
        check("snapshot.no_reset_flag",
              ("-r" not in argv.split()) and ("--gpu-reset" not in argv), argv[:200])
        # Tripwire exits 3 with empty stdout: snapshot must return a record
        # (possibly empty output), never raise (a raise would crash main).
        check("snapshot.returns_record", isinstance(snap, dict))


# Row 7/11: the mandated smoke route links no cudart/libcuda and runs with a
# hostile device env. Locks the 6.5.1 safe_harness fix at mechanism level.
def test_smoke_route_purity():
    from epic_u2o.kernel_driver import synthesize
    with tempfile.TemporaryDirectory(prefix="u651-smoke-") as td:
        tmp = Path(td)
        d = one_spec_dict()
        cu = tmp / "drv.cu"
        exe = tmp / "drv.out"
        cu.write_text(synthesize(d))
        r = subprocess.run(
            ["nvcc", "-cudart=none", "-O2", str(cu), "-o", str(exe),
             f"-L{REPO}", "-lfakegpu",
             "-Xlinker", "-rpath", "-Xlinker", str(REPO)],
            capture_output=True, text=True, timeout=180, env=HOSTILE_ENV)
        check("smoke.cudart_none_links", r.returncode == 0, r.stderr[:200])
        if r.returncode != 0:
            return
        ldd = subprocess.run(["ldd", str(exe)], capture_output=True, text=True,
                             timeout=30, env=HOSTILE_ENV).stdout
        check("smoke.ldd_no_cudart", "libcudart" not in ldd)
        check("smoke.ldd_no_libcuda", "libcuda.so" not in ldd, ldd[:200])
        run = subprocess.run([str(exe)], capture_output=True, text=True,
                             timeout=60, env=HOSTILE_ENV)
        check("smoke.runs_hostile_env", run.returncode == 0,
              run.stderr[:200])
        trials = [ln for ln in run.stdout.splitlines() if ln.strip().startswith("{")]
        check("smoke.prints_trial_json", len(trials) > 0, f"{len(trials)} lines")


# Row 8: smoke timings never become measurements -- the fixed
# SafeHarness._preflight_smoke_run returns (bool, str) with no ms payload,
# exercised end to end on the real code path (not a replica).
def test_preflight_smoke_returns_no_timings():
    from epic_u2o.safe_harness import SafeHarness
    h = SafeHarness(timeout_sec=30)
    d = one_spec_dict()
    spec = {"source": d["source"], "kernel_name": d["kernel_name"],
            "grid": d["grid"], "block": d["block"], "args": d["args"],
            "problem_size": d["problem_size"]}
    ok, msg = h._preflight_smoke_run(spec)
    check("preflight.smoke_passes_real_path", ok, msg[:200])
    check("preflight.message_carries_no_ms_timing",
          re.search(r"\d+\.\d+\s*ms", msg) is None, msg[:200])


# Row 9: the nvcc -ptx gate exists and fires (no link, no GPU, no exec).
def test_ptx_gate_fires():
    from epic_u2o.safe_harness import SafeHarness
    h = SafeHarness(timeout_sec=30)
    bad = "this is not valid CUDA at all {{{"
    ok, msg = h.preflight_check(bad, [1, 1, 1], [256, 1, 1])
    check("preflight.ptx_gate_rejects_broken", not ok, msg[:200])
    check("preflight.ptx_gate_names_cause", "nvcc -ptx failed" in msg, msg[:200])


def main():
    test_build_needs_no_device_and_shells_nothing()
    test_device_snapshot_is_read_only()
    test_smoke_route_purity()
    test_preflight_smoke_returns_no_timings()
    test_ptx_gate_fires()
    print(f"failures={FAILS}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
