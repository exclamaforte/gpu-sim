#!/usr/bin/env python3
"""Bisect WHY SafeHarness.run_kernel fails on real CUDA while the same binary
runs fine standalone (gpu-sim-u2o.6.1.1, jobs 20260927t052025 / t094146).

Evidence so far:
  - probe (cudaFree(0)+cudaMalloc) succeeds with AND without the harness rlimits
  - kernel_driver-synthesized scale program runs real trials standalone
  - the SAME binary through SafeHarness.run_kernel dies at the first cudaMalloc
    with "OS call failed or operation not supported on this OS"

So the culprit is in the delta between "bare" and "run_kernel": the remaining
rlimits (NOFILE / NPROC) or the injected FAKEGPU_DEVICE_MEM_MB env var.

Prints a variant x result table, then (if a fix is identified) re-runs the
full measure_spec pipeline with the fix applied in-process.
"""
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness, ResourceLimits  # noqa: E402
from epic_u2o.kernel_driver import spec_from_stage1, synthesize  # noqa: E402
from epic_u2o import measurement_stats  # noqa: E402

CASES = [
    ("scale", [4, 1, 1], [256, 1, 1], 1024),
    ("add",   [8, 1, 1], [256, 1, 1], 2048),
    ("relu",  [16, 1, 1], [256, 1, 1], 4096),
    ("saxpy", [32, 1, 1], [256, 1, 1], 8192),
]

MB = 1024 * 1024


def limits(**kw):
    """Build a preexec_fn applying a chosen subset of rlimits."""
    def apply():
        if kw.get("addr"):
            resource.setrlimit(resource.RLIMIT_AS, (kw["addr"] * MB, kw["addr"] * MB))
        if kw.get("data"):
            resource.setrlimit(resource.RLIMIT_DATA, (kw["data"] * MB, kw["data"] * MB))
        if kw.get("cpu"):
            resource.setrlimit(resource.RLIMIT_CPU, (kw["cpu"], kw["cpu"]))
        if kw.get("nofile"):
            resource.setrlimit(resource.RLIMIT_NOFILE, (kw["nofile"], kw["nofile"]))
        if kw.get("nproc"):
            resource.setrlimit(resource.RLIMIT_NPROC, (kw["nproc"], kw["nproc"]))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def run(exe, env_extra, preexec):
    e = os.environ.copy()
    e.update(env_extra)
    r = subprocess.run([exe], capture_output=True, text=True, timeout=120,
                       env=e, preexec_fn=preexec)
    first = (r.stdout.splitlines() or [""])[0]
    return {"rc": r.returncode, "stderr": r.stderr.strip()[:200], "stdout0": first[:160]}


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    spec = spec_from_stage1("scale", [4, 1, 1], [256, 1, 1], 1024)
    cu = tmp / "scale.cu"
    cu.write_text(synthesize(spec))
    exe = str(tmp / "scale")
    c = subprocess.run(["nvcc", "-O2", "-arch=sm_120", str(cu), "-o", exe],
                       capture_output=True, text=True, timeout=300)
    if c.returncode != 0:
        print("COMPILE FAIL", c.stderr[:800])
        return 9

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    print("shell NOFILE soft/hard:", resource.getrlimit(resource.RLIMIT_NOFILE))
    print("shell NPROC soft/hard:", resource.getrlimit(resource.RLIMIT_NPROC))

    variants = {
        "1_bare":                ({}, None),
        "2_fakegpu_env":         ({"FAKEGPU_DEVICE_MEM_MB": "24576"}, None),
        "3_nofile256":           ({}, limits(nofile=256)),
        "4_nproc32":             ({}, limits(nproc=32)),
        "5_full_harness_set":    ({}, limits(addr=262144, data=8192, cpu=60,
                                                nofile=256, nproc=32)),
    }
    table = {}
    for name, (envx, pre) in variants.items():
        try:
            res = run(exe, envx, pre)
        except subprocess.TimeoutExpired:
            res = {"rc": "timeout", "stderr": "", "stdout0": ""}
        table[name] = res
        print(f"{name:22s} rc={res['rc']} err={res['stderr'][:110]!r} out0={res['stdout0'][:70]!r}",
              flush=True)

    culprit = None
    if table["1_bare"]["rc"] == 0 and table["2_fakegpu_env"]["rc"] != 0:
        culprit = "FAKEGPU_DEVICE_MEM_MB env var"
    elif table["1_bare"]["rc"] == 0 and table["3_nofile256"]["rc"] != 0:
        culprit = "RLIMIT_NOFILE=256"
    elif table["1_bare"]["rc"] == 0 and table["4_nproc32"]["rc"] != 0:
        culprit = "RLIMIT_NPROC=32"
    print("CULPRIT:", culprit)

    # Re-run the full pipeline with the culprit neutralized in-process.
    fixed_limits = ResourceLimits
    if culprit == "RLIMIT_NOFILE=256":
        fixed_limits.open_files_limit = 1024
    elif culprit == "RLIMIT_NPROC=32":
        fixed_limits.processes_limit = 512
    if culprit == "FAKEGPU_DEVICE_MEM_MB env var":
        import epic_u2o.safe_harness as sh
        _orig = sh.SafeHarness.run_kernel
        def patched(self, executable_path, args=None, env=None):
            return _orig(self, executable_path, args,
                         {k: v for k, v in (env or {}).items()
                          if k != "FAKEGPU_DEVICE_MEM_MB"})
        sh.SafeHarness.run_kernel = patched

    results = []
    harness = SafeHarness(timeout_sec=60, log_dir=str(HERE / "harness_logs"))
    import epic_u2o.safe_harness as sh
    sh.ResourceLimits = fixed_limits
    t0 = time.time()
    for op, grid, block, n in CASES:
        sp = spec_from_stage1(op, grid, block, n)
        entry = {"op": op, "grid": grid, "block": block, "problem_size": n}
        try:
            trials, run_res = harness.measure_spec(sp, arch="sm_120")
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            results.append(entry)
            print(f"{op}: PIPELINE FAILURE {entry['error'][:200]}")
            continue
        entry["success"] = run_res.success
        entry["n_trials"] = len(trials)
        entry["trials_ms"] = trials
        if trials:
            st = measurement_stats.compute_trial_stats(trials)
            entry.update(median_ms=st.median_ms, mean_ms=st.mean_ms, std_ms=st.std_ms,
                         cv=st.cv, is_reliable=st.is_reliable,
                         discard_reason=st.discard_reason.value)
            print(f"{op:6s} trials={len(trials)} median={st.median_ms:.6f}ms "
                  f"mean={st.mean_ms:.6f}ms cv={st.cv:.4f} reliable={st.is_reliable}")
        else:
            entry["stderr"] = run_res.stderr[:500]
            print(f"{op:6s} NO TRIALS rc={run_res.returncode} {run_res.stderr[:160]!r}")
        results.append(entry)

    summary = {
        "bead": "gpu-sim-u2o.6.1.1",
        "variant_table": table,
        "culprit": culprit,
        "n_cases": len(CASES),
        "n_with_trials": sum(1 for r in results if r.get("n_trials")),
        "total_wall_sec": time.time() - t0,
        "results": results,
    }
    (HERE / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps({k: summary[k] for k in
                                 ("culprit", "n_cases", "n_with_trials", "total_wall_sec")}))
    return 0 if summary["n_with_trials"] == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
