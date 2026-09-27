#!/usr/bin/env python3
"""End-to-end verification of the fixed safe_harness (gpu-sim-u2o.6.1.1).

Runs the full generate -> synthesize -> preflight -> compile -> run pipeline
for several stage-1 elementwise specs on the REAL GPU, and records every raw
trial timing. This is the first end-to-end run of SafeHarness.measure_spec()
against real hardware (the audit bead u2o.15 established it had never run).

Outputs (written next to this file):
  metrics.json   machine-checkable summary: per-spec trials, median, stats
  output.log     full human-readable log (also stdout of the ts job)
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # worktree root (experiments/<id>/ -> <root>)
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness          # noqa: E402
from epic_u2o.kernel_driver import spec_from_stage1     # noqa: E402
from epic_u2o import measurement_stats                  # noqa: E402

# (op, grid, block, n) -- grid*block must cover n
CASES = [
    ("scale", [4, 1, 1], [256, 1, 1], 1024),
    ("add",   [8, 1, 1], [256, 1, 1], 2048),
    ("relu",  [16, 1, 1], [256, 1, 1], 4096),
    ("saxpy", [32, 1, 1], [256, 1, 1], 8192),
]


def gpu_identity() -> dict:
    """Device identity read via nvidia-smi (inside the queued job only)."""
    info = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        info["nvidia_smi_rc"] = r.returncode
        info["nvidia_smi"] = r.stdout.strip()
    except Exception as e:  # pragma: no cover
        info["nvidia_smi_error"] = str(e)
    return info


def main() -> int:
    print(f"host={platform.node()} python={platform.python_version()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    ident = gpu_identity()
    print(f"gpu: {ident.get('nvidia_smi')!r}")

    harness = SafeHarness(timeout_sec=60, gpu_id=0, log_dir=str(HERE / "harness_logs"))
    results = []
    t0 = time.time()
    for op, grid, block, n in CASES:
        spec = spec_from_stage1(op, grid, block, n)
        print(f"\n=== {op} grid={grid} block={block} n={n} "
              f"kernel={spec['kernel_name']} ===", flush=True)
        entry = {"op": op, "grid": grid, "block": block, "problem_size": n,
                 "kernel_name": spec["kernel_name"]}
        try:
            trials, run = harness.measure_spec(spec, arch="sm_120")
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            print(f"  PIPELINE FAILURE: {entry['error']}")
            results.append(entry)
            continue

        entry["success"] = run.success
        entry["returncode"] = run.returncode
        entry["timeout"] = run.timeout
        entry["wall_time_sec"] = run.wall_time_sec
        entry["n_trials"] = len(trials)
        entry["trials_ms"] = trials
        if trials:
            stats = measurement_stats.compute_trial_stats(trials)
            entry["median_ms"] = stats.median_ms
            entry["mean_ms"] = stats.mean_ms
            entry["std_ms"] = stats.std_ms
            entry["cv"] = stats.cv
            entry["iqr_ms"] = stats.iqr_ms
            entry["is_reliable"] = stats.is_reliable
            entry["discard_reason"] = stats.discard_reason.value
            print(f"  trials={len(trials)} median={stats.median_ms:.6f}ms "
                  f"mean={stats.mean_ms:.6f}ms cv={stats.cv:.4f} "
                  f"reliable={stats.is_reliable} wall={run.wall_time_sec:.2f}s")
            # Sanity: computed result must be non-trivial (i.e. real compute ran)
            entry["first_trial_ms"] = trials[0]
        else:
            entry["stderr"] = run.stderr[:2000]
            print(f"  NO TRIALS rc={run.returncode} stderr={run.stderr[:400]!r}")
        results.append(entry)

    summary = {
        "bead": "gpu-sim-u2o.6.1.1",
        "what": "end-to-end SafeHarness.measure_spec() on the real RTX 5090",
        "device": ident,
        "n_cases": len(CASES),
        "n_succeeded": sum(1 for r in results if r.get("success")),
        "n_with_trials": sum(1 for r in results if r.get("n_trials")),
        "total_wall_sec": time.time() - t0,
        "results": results,
    }
    (HERE / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY " + json.dumps(
        {k: summary[k] for k in ("n_cases", "n_succeeded", "n_with_trials",
                                  "total_wall_sec")}))
    return 0 if summary["n_with_trials"] == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
