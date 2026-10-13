#!/usr/bin/env python3
"""gpu-sim-f6y COMPILE (CPU, unqueued): synthesize and compile every driver.

11.3's compile_all shape: nvcc runs here, the queued job only measures
(the queue killed the combined job for 0.7% GPU utilisation). Records wall
time for the report.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.kernel_driver import synthesize  # noqa: E402
from epic_u2o.safe_harness import SafeHarness  # noqa: E402


def main() -> int:
    build = HERE / "build"
    specs = json.loads((HERE / "manifest.json").read_text())["specs"]
    h = SafeHarness(timeout_sec=300)
    ok = bad = 0
    errors = []
    t0 = time.time()
    for i, d in enumerate(specs):
        spec = {k: d[k] for k in ("source", "kernel_name", "grid", "block",
                                  "args", "problem_size", "shared_mem_bytes")}
        spec["op_type"] = d.get("op_type", "anchor")
        spec["metadata"] = d.get("metadata", {})
        try:
            program = synthesize(spec)
        except Exception as e:
            bad += 1
            errors.append({"hash": d["kernel_hash"], "stage": "synthesize",
                           "error": str(e)[:200]})
            continue
        c, m = h.compile_kernel(program, str(build / f"k_{d['kernel_hash']}.out"),
                                arch="sm_120")
        if c:
            ok += 1
        else:
            bad += 1
            errors.append({"hash": d["kernel_hash"], "stage": "compile",
                           "error": str(m)[:200]})
        print(f"  compiled {i + 1}/{len(specs)} ({time.time() - t0:.0f}s)",
              flush=True)
    wall = time.time() - t0
    (build / "compile_errors.json").write_text(json.dumps(errors, indent=2))
    (build / "compile_summary.json").write_text(json.dumps(
        {"n_specs": len(specs), "n_ok": ok, "n_failed": bad,
         "wall_sec": wall}, indent=2))
    print(f"compiled {ok}/{len(specs)} in {wall:.0f}s, {bad} failed")
    return 0 if (ok == len(specs)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
