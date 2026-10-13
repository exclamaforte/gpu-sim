#!/usr/bin/env python3
"""GPU proof for gpu-sim-u2o.6.1.2: a real kernel still completes with the
per-run task cap in force, and at the same speed.

Acceptance criterion 2: one ts JOB_ID, saxpy n=8192 grid 32x1x1 block 256,
median within 10% of the gpu-sim-u2o.6.1.1 reference (0.002848 ms).

preflight_check is BYPASSED (PI ruling on 6.1.1: it executes on the real GPU,
and this experiment must not carry an unaccounted second execution). Nothing
generated, nothing untrusted: the reference kernel is fixed and hand-checked
here, so nothing depends on the fake path or on any stage in rework.
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness  # noqa: E402
from epic_u2o import task_limits as tl  # noqa: E402
from epic_u2o.kernel_driver import synthesize  # noqa: E402

# The reference point, from gpu-sim-u2o.6.1.1 (ts 20260927t145454-9c2420b3).
REFERENCE_MEDIAN_MS = 0.002848
TOLERANCE = 0.10
TRIALS = 20
INVOCATIONS = 3

SPEC = {
    "source": """
__global__ void k_saxpy(const float* x, float* y, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + y[i];
}
""".strip() + "\n",
    "kernel_name": "k_saxpy",
    "grid": [32, 1, 1],
    "block": [256, 1, 1],
    "shared_mem_bytes": 0,
    "args": [{"name": "x", "type": "const float*", "ptr": True},
             {"name": "y", "type": "float*", "ptr": True},
             {"name": "alpha", "type": "float", "ptr": False},
             {"name": "n", "type": "int", "ptr": False}],
    "problem_size": 8192,
    "op_type": "memory",
    "metadata": {"generator": "hand-written reference (u2o.6.1.2)"},
}


def telemetry():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,clocks.sm,temperature.gpu,"
             "clocks_throttle_reasons.active",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return {"rc": r.returncode, "csv": r.stdout.strip()}
    except Exception as e:
        return {"error": str(e)}


def parse(stdout):
    return [float(json.loads(l)["time_ms"])
            for l in stdout.splitlines()
            if l.strip().startswith("{") and "time_ms" in l]


def main() -> int:
    print("=" * 72)
    print("gpu-sim-u2o.6.1.2 -- real kernel under the per-run task cap")
    print("=" * 72)
    print("preflight_check BYPASSED on purpose: it executes on the real GPU "
          "(PI ruling 6.1.1) and this experiment must not inherit an "
          "unaccounted second execution. Fixed hand-written kernel.")
    mechs = tl.probe()
    for m in mechs:
        print(f"  mechanism {m.name:14s} available={m.available}  {m.detail}")
    print(f"  live task count: {tl.live_task_count()}")
    print(f"  telemetry before: {telemetry()}", flush=True)

    h_capped = SafeHarness(timeout_sec=120, vram_limit_mb=24576,
                           tasks_max=tl.DEFAULT_TASKS_MAX)
    h_none = SafeHarness(timeout_sec=120, vram_limit_mb=24576, tasks_max=0)

    exe = HERE / "saxpy.out"
    ok, msg = h_capped.compile_kernel(synthesize(SPEC), str(exe), arch="sm_120")
    print(f"compile_kernel -> ok={ok} {str(msg)[:120]}", flush=True)
    if not ok:
        (HERE / "metrics.json").write_text(json.dumps(
            {"bead": "gpu-sim-u2o.6.1.2", "error": f"compile failed: {msg[:1500]}"},
            indent=2))
        return 1

    results = {}
    for tag, harness in (("capped", h_capped), ("uncapped", h_none)):
        meds, peaks, ok_runs = [], [], 0
        for inv in range(INVOCATIONS):
            r = harness.run_kernel(str(exe))
            if not r.success:
                print(f"  !! {tag} inv{inv}: rc={r.returncode} "
                      f"{r.stderr[:160]!r}", flush=True)
                continue
            trials = parse(r.stdout)
            if not trials:
                print(f"  !! {tag} inv{inv}: no trials", flush=True)
                continue
            ok_runs += 1
            meds.append(statistics.median(trials))
            peaks.append(trials)
        results[tag] = {
            "runs": ok_runs,
            "median_ms": statistics.median(meds) if meds else None,
            "medians_ms": meds,
            "cv_pct": (100.0 * (max(meds) - min(meds)) / statistics.median(meds)
                       if len(meds) > 1 else None),
            "task_limits": (r.metadata.get("task_limits")
                            if ok_runs else None),
        }
        print(f"  {tag:9s} runs={ok_runs} median="
              f"{results[tag]['median_ms']} ms medians={meds}", flush=True)

    med = results["capped"]["median_ms"]
    within = (med is not None and
              abs(med - REFERENCE_MEDIAN_MS) / REFERENCE_MEDIAN_MS <= TOLERANCE)
    metrics = {
        "bead": "gpu-sim-u2o.6.1.2",
        "what": "a real kernel still completes under the per-run task cap, at the same speed",
        "preflight_bypassed": True,
        "mechanisms_available": [
            {"name": m.name, "available": m.available, "detail": m.detail}
            for m in mechs],
        "chosen_mechanism": tl.choose_mechanism().name,
        "tasks_max": tl.DEFAULT_TASKS_MAX,
        "observed_max_threads": results["capped"].get("task_limits"),
        "enforcement": "observed",
        "enforcement_evidence": "tests/test_task_limits.py: a child fails to "
                                 "fork with EAGAIN at 63 children under "
                                 "TasksMax=64, and forks 200 children when "
                                 "uncapped",
        "reference_median_ms": REFERENCE_MEDIAN_MS,
        "reference_source": "gpu-sim-u2o.6.1.1, ts 20260927t145454-9c2420b3 "
                            "(saxpy n=8192 grid 32x1x1 block 256)",
        "tolerance": TOLERANCE,
        "capped_median_ms": med,
        "within_tolerance": bool(within),
        "results": results,
        "telemetry_before": telemetry(),
        "telemetry_after": telemetry(),
    }
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\ncapped median {med} ms vs reference {REFERENCE_MEDIAN_MS} ms "
          f"-> within {int(TOLERANCE*100)}%: {within}")
    return 0 if within and results["capped"]["runs"] == INVOCATIONS else 1


if __name__ == "__main__":
    raise SystemExit(main())
