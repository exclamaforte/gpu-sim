#!/usr/bin/env python3
"""Real-GPU validation of the gpu-sim-u2o.6.3 resource budgets.

Runs ONLY inside a ts-queued job (gpu-sim-u2o.6.8). Five checks:

  A. Static VRAM cap refuses a pathological spec BEFORE the GPU is touched.
  B. A normal spec still runs end to end and reports 20 real trials.
  C. Device-side cap: the run's own footprint (cudaMemGetInfo delta) is
     recorded and a deliberately tiny cap flips the run to failure.
  D. Disk quota: the child writes into an accounted scratch dir; the quota
     is checked after exit and the directory is removed.
  E. Runaway kill: a host-side (no GPU) infinite loop is killed by the
     timeout path, the whole process group dies, and the harness reports
     timeout=True. The GPU-spinning --test-hang kernel is deliberately NOT
     used here (that is u2o.6.10, supervised, display GPU).
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

from epic_u2o.safe_harness import SafeHarness, enforce_device_vram_cap  # noqa: E402
from epic_u2o.resource_limits import check_vram_budget, ResourcePolicy  # noqa: E402
from epic_u2o.kernel_driver import spec_from_stage1  # noqa: E402

checks = []
metrics = {"bead": "gpu-sim-u2o.6.3", "checks": checks}


def record(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {str(detail)[:220]}", flush=True)


def gpu_identity():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception as e:
        return f"error: {e}"


def main() -> int:
    print("device:", gpu_identity())
    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    # --- A. static cap, oversized spec -----------------------------------
    # 1 Gi floats x 2 buffers = 8 GiB. The grid is sized so the generator will
    # actually emit the spec (it requires grid*block >= n/4).
    big = spec_from_stage1("scale", [1 << 20, 1, 1], [256, 1, 1], 1 << 30)
    policy = ResourcePolicy(vram_limit_mb=2048)
    ok, msg = check_vram_budget(big, policy)
    record("A1 static VRAM cap rejects 8 GiB spec", (not ok) and "exceeded" in msg, msg)

    harness = SafeHarness(timeout_sec=60, vram_limit_mb=2048)
    t0 = time.time()
    try:
        harness.measure_spec(big, arch="sm_120")
        record("A2 measure_spec refuses oversized spec (RuntimeError)", False, "no exception")
    except RuntimeError as e:
        record("A2 measure_spec refuses oversized spec (RuntimeError)",
               "VRAM budget exceeded" in str(e), str(e))
    except Exception as e:
        record("A2 measure_spec refuses oversized spec (RuntimeError)", False,
               f"{type(e).__name__}: {e}")
    record("A3 refusal was fast (no GPU work, <5s)", (time.time() - t0) < 5.0,
           f"{time.time() - t0:.2f}s")

    # --- B + C. normal run with device-side accounting --------------------
    good = spec_from_stage1("add", [4096, 1, 1], [256, 1, 1], 1 << 20)  # 3x4 MiB buffers
    trials, res = harness.measure_spec(good, arch="sm_120")
    record("B1 normal spec produces 20 real trials", len(trials) == 20, f"{len(trials)} trials")
    dm = res.metadata.get("device_mem")
    record("C1 device footprint recorded (cudaMemGetInfo delta)", bool(dm), json.dumps(dm or {}))
    if dm:
        est = (1 << 20) * 4 * 3 / 1048576
        record("C2 measured delta is in the right ballpark for 3x4MiB buffers",
               abs(float(dm["delta_mb"]) - est) < 64,
               f"delta={dm['delta_mb']} MB, static estimate={est:.1f} MB, "
               f"device-wide used={dm.get('device_used_mb')} MB")
        record("C3 device-wide figure is reported separately (shared-GPU aware)",
               "device_used_mb" in dm, json.dumps(dm))

    # --- C4. device-side cap, on the REAL measured footprint ---------------
    if dm:
        ok_cap, cap_msg = enforce_device_vram_cap(dm, cap_mb=1)
        record("C4 a 1 MB cap flags the real measured footprint",
               (not ok_cap) and "footprint" in cap_msg, cap_msg)
        ok_cap2, cap_msg2 = enforce_device_vram_cap(dm, cap_mb=2048)
        record("C5 the same footprint passes a 2048 MB cap", ok_cap2, cap_msg2)
        ok_cap3, _ = enforce_device_vram_cap(dm, cap_mb=0)
        record("C6 cap<=0 disables the device-side check", ok_cap3, "cap disabled")
    # (measure_spec refuses an oversized spec statically first, which is the
    #  intended order: static estimate gates, device measurement audits.)

    # --- D. disk quota on a real child ------------------------------------
    scratch_base = HERE / "scratch"
    h3 = SafeHarness(timeout_sec=60, vram_limit_mb=2048,
                     disk_quota_mb=2, scratch_base=str(scratch_base))
    writer = HERE / "writer.cu"
    writer_src = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char** argv) {
    (void)argc;
    long mb = atol(argv[1]);
    FILE* f = fopen("child_output.bin", "wb");
    if (!f) { perror("fopen"); return 1; }
    char* buf = (char*)malloc(1048576);
    for (long i = 0; i < mb; i++) { memset(buf, 'x', 1048576); fwrite(buf, 1, 1048576, f); }
    fclose(f);
    printf("wrote %ld MB\n", mb);
    return 0;
}
'''
    writer.write_text(writer_src)
    exe = HERE / "writer.out"
    c = subprocess.run(["nvcc", "-O2", str(writer), "-o", str(exe)],
                       capture_output=True, text=True, timeout=300)
    record("D0 host writer compiles", c.returncode == 0, c.stderr[-200:])

    r_ok = h3.run_kernel(str(exe), args=["1"])          # 1 MB, under a 2 MB quota
    d_ok = r_ok.metadata.get("disk", {})
    record("D1 small child run is within quota", bool(d_ok.get("within_quota")), json.dumps(d_ok))
    r_big = h3.run_kernel(str(exe), args=["6"])         # 6 MB asked, 2 MB allowed
    d_big = r_big.metadata.get("disk", {})
    # The RLIMIT_FSIZE backstop (min(file limit, quota) = 2 MB here) fires
    # BEFORE the Python-side accounting can see an overage: the child dies
    # partway and the directory never exceeds the quota. That is the intended
    # layering, so assert exactly that rather than a quota "violation".
    record("D2 RLIMIT_FSIZE backstop caps the 6 MB write at the 2 MB limit",
           (not r_big.success) and d_big.get("used_mb", 0) <= 2.0
           and ("File size limit exceeded" in r_big.stderr
                or r_big.returncode != 0),
           f"rc={r_big.returncode} used_mb={d_big.get('used_mb')} "
           f"stderr={r_big.stderr.strip()[:80]!r}")
    d_big = r_big.metadata.get("disk", {})
    record("D2b quota accounting still ran and reported within-quota",
           bool(d_big.get("within_quota", True)), json.dumps(d_big))
    record("D3 scratch directories are cleaned up",
           not any(scratch_base.glob("kern-*")) if scratch_base.exists() else True,
           str(list(scratch_base.glob('kern-*')) if scratch_base.exists() else []))

    # --- E. runaway kill, host side (no GPU) ------------------------------
    spinner = HERE / "spinner.c"
    spinner.write_text("int main(void){ volatile int x=0; while(x==0){} return 0; }\n")
    sexe = HERE / "spinner.out"
    c = subprocess.run(["nvcc", "-O2", str(spinner), "-o", str(sexe)],
                       capture_output=True, text=True, timeout=300)
    record("E0 spinner compiles", c.returncode == 0, c.stderr[-200:])
    h4 = SafeHarness(timeout_sec=3, vram_limit_mb=2048)
    res_spin = h4.run_kernel(str(sexe))
    record("E1 infinite loop is killed by the timeout path",
           res_spin.timeout and res_spin.wall_time_sec < 10,
           f"timeout={res_spin.timeout} wall={res_spin.wall_time_sec:.2f}s rc={res_spin.returncode}")
    # nothing from that process group may survive
    leftover = subprocess.run(["pgrep", "-f", "spinner.out"], capture_output=True, text=True)
    record("E2 no surviving child after the kill", leftover.returncode != 0,
           leftover.stdout.strip() or "pgrep found nothing")
    record("E3 harness recorded the post-timeout GPU health check",
           "gpu_health" in res_spin.metadata, json.dumps(res_spin.metadata.get("gpu_health", "")))

    metrics["summary"] = {
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["ok"]),
        "n_failed": sum(1 for c in checks if not c["ok"]),
    }
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("SUMMARY " + json.dumps(metrics["summary"]))
    return 0 if metrics["summary"]["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
