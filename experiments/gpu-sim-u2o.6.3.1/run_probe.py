#!/usr/bin/env python3
"""Run the rlimit/CUDA compatibility probe on the real 5090, under the
harness's EXACT limit set. gpu-sim-u2o.6.3.1 part (2).

This must run through `ts` (never directly), because it touches a GPU three
other projects share. It is cheap by design -- the design says <10s, because
this is meant to gate every batch job.

The limits are applied in the child's preexec_fn, the same mechanism and the
same values run_kernel uses, so a passing probe means the REAL limit set is
CUDA-safe rather than that a hand-written approximation of it is.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.rlimit_cuda_gate import (  # noqa: E402
    PROBE_SOURCE, diagnose_probe_failure, limits_harness_applies,
)
from epic_u2o.safe_harness import ResourceLimits  # noqa: E402

ARCH = "sm_120"   # RTX 5090 (Blackwell)


def main() -> int:
    limits = ResourceLimits()
    applied = limits_harness_applies(limits)
    print(f"applying limits: {applied}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}",
          flush=True)

    with tempfile.TemporaryDirectory() as td:
        cu = Path(td) / "probe.cu"
        exe = Path(td) / "probe"
        cu.write_text(PROBE_SOURCE)

        t0 = time.time()
        build = subprocess.run(
            ["nvcc", f"-arch={ARCH}", str(cu), "-o", str(exe)],
            capture_output=True, text=True)
        if build.returncode != 0:
            print("BUILD FAILED\n" + build.stderr, flush=True)
            return 10

        # preexec_fn, not shell, so the limits are inherited by the probe itself
        run = subprocess.run([str(exe)], capture_output=True, text=True,
                             timeout=60, preexec_fn=limits.apply)
        elapsed = time.time() - t0

    print(f"probe rc={run.returncode} in {elapsed:.2f}s", flush=True)
    print(f"stdout: {run.stdout.strip()}", flush=True)
    if run.stderr.strip():
        print(f"stderr: {run.stderr.strip()[:500]}", flush=True)

    out: dict = {
        "bead": "gpu-sim-u2o.6.3.1",
        "probe": "rlimit_cuda_compatibility",
        "applied_limits": applied,
        "limits_applied_in": "child preexec_fn (same mechanism as run_kernel)",
        "arch": ARCH,
        "rc": run.returncode,
        "elapsed_s": round(elapsed, 3),
        "stdout": run.stdout.strip(),
    }

    if run.returncode == 0:
        try:
            payload = json.loads(run.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            out["result"] = f"UNPARSEABLE probe output: {exc}"
            out["passed"] = False
            print(f"FAIL: {out['result']}", flush=True)
            (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
            return 11
        out.update(payload)
        # rc 0 with a wrong result would still be a silent pass, so the
        # read-back is asserted rather than recorded
        out["passed"] = bool(payload.get("result_correct"))
        if not out["passed"]:
            print("FAIL: probe launched but the result is wrong", flush=True)
            (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
            return 12
        print("PASS: cudaMalloc + launch + readback correct under the "
              "harness's exact limit set", flush=True)
    else:
        # the fail-fast path: name the limit, do not say "kernel failed"
        stage = "unknown"
        err = run.stderr.strip()
        try:
            payload = json.loads(run.stdout.strip().splitlines()[-1])
            stage = payload.get("stage", "unknown")
            err = payload.get("error", err)
        except (ValueError, IndexError):
            pass
        out["passed"] = False
        out["failed_stage"] = stage
        out["diagnosis"] = diagnose_probe_failure(stage, err or "no message",
                                                  applied)
        print("FAIL (diagnosed, not hand-waved):\n" + out["diagnosis"],
              flush=True)

    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {HERE / 'metrics.json'}", flush=True)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
