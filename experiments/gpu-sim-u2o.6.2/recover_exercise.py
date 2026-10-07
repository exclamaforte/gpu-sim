#!/usr/bin/env python3
"""gpu-sim-u2o.6.2 recovery exercise -- GPU DOSE driver (runs inside ONE ts job).

Exercises the RULED procedure (SOFT_RESET-only, foreign-holder gate) against
TWO induced failures, with a calibrator kernel before and after:

  PRE:   GPU state snapshot + calibrator median (the pre-failure baseline).
  ARM A: simulated hang -- an infinite-loop kernel run under SafeHarness with
         a 5s timeout. Expect: timeout kill, NO survivor, health HEALTHY,
         decision NO_RESET, calibrator re-runs with numbers.
  ARM B: survivor -- a CUDA holder that ignores SIGTERM. Watchdog SIGTERM +
         grace leaves survivor_pids=[holder]. Health check names it;
         The survivor observation itself indicates the SOFT_RESET attempt, run
         through the foreign-holder gate (the desktop is active on this same
         5090, so the gate DECLINES it); SURFACE_TO_OWNER is written, and no
         reset pulse ever fires. Own survivor is then
         SIGKILLed (cleanup, not escalation) and the calibrator re-runs.
  POST:  GPU state snapshot.

SAFETY INTERLOCK: arm B proceeds ONLY if foreign (non-job) GPU holders are
observed first. If the desktop is idle and no foreign holder exists, the dose
ABORTS (exit 2, recorded) rather than risk a live reset pulse. The abort is
the procedure protecting the box, and it is reported as such.

Outputs: <OUTDIR>/recover_steps.jsonl (one record per step, committed later).
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness  # noqa: E402
from epic_u2o.gpu_recovery import GPURecovery, RecoveryAction  # noqa: E402

OUTDIR = Path(os.environ.get("U2O_REC_OUTDIR",
                              "/home/gabe/Documents/gpu-sim-runs/u62-recover-20261007"))
ARCH = "sm_120"
SPIN_TIMEOUT = 5
HOLDER_GRACE = 6

CALIB_SRC = r"""
#include <cstdio>
#include <cuda_runtime.h>
__global__ void vadd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    const int n = 1 << 20, T = 20;
    float *a, *b, *c;
    cudaMalloc(&a, n * sizeof(float)); cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms[T];
    for (int t = 0; t < T; t++) {
        cudaEventRecord(s);
        vadd<<<(n + 255) / 256, 256>>>(a, b, c, n);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms[t], s, e);
    }
    float v[T]; for (int t = 0; t < T; t++) v[t] = ms[t];
    for (int i = 0; i < T; i++) for (int j = i + 1; j < T; j++)
        if (v[j] < v[i]) { float x = v[i]; v[i] = v[j]; v[j] = x; }
    printf("MEDIAN %.6f\n", v[T / 2]);
    return 0;
}
"""

SPIN_SRC = r"""
#include <cstdio>
#include <cuda_runtime.h>
#define CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); return 3; } } while(0)
__global__ void spin() { while (1) { } }
int main() {
    spin<<<1, 1>>>();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    fprintf(stderr,"SPIN-EXITED-UNEXPECTEDLY\n");
    return 0;
}
"""

HOLDER_SRC = r"""
#include <cstdio>
#include <csignal>
#include <unistd.h>
#include <cuda_runtime.h>
#define CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); return 3; } } while(0)
__global__ void nap() {
    long long t0 = clock64();
    while (clock64() - t0 < 180LL * 1000000000LL) { }
}
int main() {
    signal(SIGTERM, SIG_IGN);  /* the survivor: SIGTERM does not kill us */
    float *p = 0; CHECK(cudaMalloc(&p, 64 << 20));
    printf("READY %d\n", (int)getpid()); fflush(stdout);
    nap<<<1, 1>>>();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    fprintf(stderr,"HOLDER-EXITED-UNEXPECTEDLY\n");
    return 0;
}
"""

steps = []


def log(step, **kw):
    rec = {"step": step, "at": datetime.now(timezone.utc).isoformat(), **kw}
    steps.append(rec)
    print(f"[{step}] " + json.dumps(kw)[:300], flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=kw.get("timeout", 30))


def snapshot():
    q = sh(["nvidia-smi", "-i", "0", "-q"])
    util = temp = power = driver = "?"
    for line in q.stdout.splitlines():
        s = line.strip()
        if s.startswith("Driver Version"):
            driver = s.split(":")[-1].strip()
        elif s.startswith("Gpu") and "%" in s and "Util" not in s:
            pass
        elif "Utilization" in s:
            util = s
        elif "GPU Current Temp" in s:
            temp = s.split(":")[-1].strip()
        elif "Power Draw" in s:
            power = s.split(":")[-1].strip()
    apps = sh(["nvidia-smi", "-i", "0", "--query-compute-apps=pid,process_name",
               "--format=csv,noheader,nounits"])
    gfx = sh(["nvidia-smi", "-i", "0", "--query-graphics-apps=pid,process_name",
              "--format=csv,noheader,nounits"])
    dm = sh(["dmesg", "-T", "-l", "err,crit,alert,emerg"], timeout=10)
    xids = [l.strip() for l in dm.stdout.splitlines()
            if "Xid" in l and ("GPU" in l or "gpu" in l or "NVRM" in l)][-5:]
    return {"driver": driver, "util": util, "temp": temp, "power": power,
            "compute_apps": apps.stdout.strip().splitlines(),
            "graphics_apps": gfx.stdout.strip().splitlines(),
            "xid_tail": xids, "smi_rc": q.returncode}


def build(td):
    exe = {}
    for name, src in (("calib", CALIB_SRC), ("spin", SPIN_SRC), ("holder", HOLDER_SRC)):
        cu = td / f"{name}.cu"
        cu.write_text(src)
        out = td / f"u2o62_{name}"
        r = sh(["nvcc", f"-arch={ARCH}", str(cu), "-o", str(out)], timeout=180)
        if r.returncode != 0:
            print(f"BUILD FAILED {name}:\n{r.stderr}", flush=True)
            return None
        exe[name] = str(out)
    return exe


def calibrator(exe):
    r = sh([exe], timeout=120)
    for line in r.stdout.splitlines():
        if line.startswith("MEDIAN"):
            return float(line.split()[1]), r.returncode
    return None, r.returncode


def guarded_recover(rec, health, arm):
    """auto_recover() only when no reset pulse can fire (see module doc)."""
    if health == "healthy":
        return rec.auto_recover(), False
    holders = rec.compute_app_pids()
    foreign = [p for p, n in holders if p not in rec.own_pids]
    if not foreign:
        log(f"{arm}_aborted", health=health, reason="unhealthy GPU with no "
            "foreign holders: a reset pulse could fire -- dose refuses")
        return None, True
    return rec.auto_recover(), False


def survivor_indicated_attempt(rec, survivor_pids, arm):
    """Arm B decision path. A surviving GPU holder IS the hang evidence the
    nvidia-smi health check cannot see (the driver stays responsive while a
    context is stuck), so the survivor observation itself indicates the
    SOFT_RESET attempt -- run here through the same foreign-holder gate as
    the library path, never around it. The pulse can fire only with no
    foreign holders, so those must be observed first; otherwise ABORT."""
    holders = rec.compute_app_pids()
    foreign = [(p, n) for p, n in holders if p not in rec.own_pids]
    if not foreign:
        log(f"{arm}_aborted", survivor_pids=survivor_pids, reason="survivor "
            "indicates reset but no foreign holders observed: a pulse could "
            "fire -- dose refuses")
        return None, True
    out = rec.attempt_recovery(RecoveryAction.SOFT_RESET)
    if not out.success:
        # The surfacing auto_recover() would have done: attempt_recovery is a
        # single attempt and does not write flags itself.
        (rec.checkpoint_dir / "SURFACE_TO_OWNER.txt").write_text(
            f"survivor-indicated soft_reset declined/failed: {out.message}\n")
    return out, False


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    import tempfile
    td = Path(tempfile.mkdtemp(prefix="u62rec-"))
    log("start", outdir=str(OUTDIR))
    exe = build(td)
    if not exe:
        log("build_failed")
        flush()
        return 10

    state_before = snapshot()
    log("state_before", **state_before)
    med_pre, rc = calibrator(exe["calib"])
    log("calibrator_pre", median_ms=med_pre, rc=rc)
    if med_pre is None:
        log("calibrator_pre_failed"); flush(); return 11

    harness = SafeHarness(timeout_sec=SPIN_TIMEOUT, gpu_id=0,
                          use_fakegpu_preflight=False)
    spin_name = os.path.basename(exe["spin"])

    # ARM A: simulated hang, watchdog kill expected, no survivor.
    res = harness.run_kernel(exe["spin"], args=[], env={})
    time.sleep(2)
    holders_a = GPURecovery(checkpoint_dir=str(OUTDIR / "ckpt_a")).compute_app_pids()
    spin_alive = [p for p, n in holders_a if spin_name in (n or "")]
    rec_a = GPURecovery(checkpoint_dir=str(OUTDIR / "ckpt_a"),
                        own_pids={os.getpid()})
    health_a = rec_a.check_gpu_health().value
    dec_a, aborted_a = guarded_recover(rec_a, health_a, "arm_a")
    if aborted_a:
        flush()
        return 2
    med_post_a, rc_a = calibrator(exe["calib"])
    log("arm_a_hang", timeout=res.timeout, success=res.success,
        wall_s=round(res.wall_time_sec, 2),
        spin_stderr_tail=(res.stderr or "")[-500:],
        spin_still_holding_gpu=spin_alive, health=health_a,
        decision=dec_a.action.value, decision_msg=dec_a.message,
        calibrator_post_ms=med_post_a, calibrator_rc=rc_a)

    # ARM B: survivor. The pulse-safety interlock lives in guarded_recover
    # (below, after the survivor is observed): unhealthy + no foreign holders
    # aborts the dose instead of risking a live reset pulse.
    rec_b = GPURecovery(checkpoint_dir=str(OUTDIR / "ckpt_b"),
                        own_pids={os.getpid()})
    holder = subprocess.Popen([exe["holder"]], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, bufsize=1)
    ready = holder.stdout.readline().strip()
    rec_b.own_pids.add(holder.pid)
    time.sleep(2)
    # Alive-precheck: if the holder self-exited before SIGTERM, the survivor
    # was never induced -- record its stderr (loud CHECK failures land here)
    # and abort the arm honestly instead of a vacuous no-survivor.
    if holder.poll() is not None:
        _, herr = holder.communicate()
        log("arm_b_no_holder", ready=ready, rc=holder.returncode,
            holder_stderr=(herr or "")[-500:])
        flush()
        return 12
    holder.send_signal(signal.SIGTERM)
    try:
        holder.wait(timeout=HOLDER_GRACE)
        survived = False
    except subprocess.TimeoutExpired:
        survived = True
    alive = (holder.poll() is None)
    survivor_pids = [holder.pid] if (survived and alive) else []
    health_b = rec_b.check_gpu_health().value
    if survivor_pids:
        dec_b, aborted_b = survivor_indicated_attempt(rec_b, survivor_pids, "arm_b")
    else:
        dec_b, aborted_b = guarded_recover(rec_b, health_b, "arm_b")
    surface_exists = (OUTDIR / "ckpt_b" / "SURFACE_TO_OWNER.txt").exists()
    holder.kill()
    if aborted_b:
        try:
            holder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        log("arm_b_aborted_after_survivor", survivor_pids=survivor_pids,
            surface_file=surface_exists,
            holder_reaped=holder.poll() is not None)
        flush()
        return 2
    try:
        holder.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    cleaned = holder.poll() is not None
    try:
        herr = holder.stderr.read() or ""
    except Exception:
        herr = ""
    med_post_b, rc_b = calibrator(exe["calib"])
    log("arm_b_survivor", holder_ready=ready, sigterm_survived=survived,
        survivor_pids=survivor_pids, health=health_b,
        decision=dec_b.action.value, decision_msg=dec_b.message,
        surface_file=surface_exists, holder_sigkilled=cleaned,
        holder_stderr=herr[-500:],
        calibrator_post_ms=med_post_b, calibrator_rc=rc_b)

    state_after = snapshot()
    log("state_after", **state_after)
    log("done")
    flush()
    return 0


def flush():
    (OUTDIR / "recover_steps.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in steps))


if __name__ == "__main__":
    raise SystemExit(main())
