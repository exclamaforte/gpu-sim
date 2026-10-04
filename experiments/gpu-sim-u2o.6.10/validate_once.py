#!/usr/bin/env python3
"""gpu-sim-u2o.6.10 -- supervised single-dose validation of the kill path.

STAGED, per the criterion-7 re-run approval: --only step1 | step2 | step2b
runs one stage per ts job so step 1 can be judged nominal before the hang is
spent, and step 2b runs only after a verified-clean kill (plus explicit PI
approval). --only assemble merges the three stage files into report.json on
CPU (no GPU). --only all preserves the original one-shot shape.

Each stage carries code gates, not just prose: step2 refuses unless step 1's
stored record is device-side-nominal; step2b refuses unless step 2's stored
record shows a reported timeout AND zero survivors. A gate that cannot fire
is a comment; these exit 2.

REVISION COVERED: run_kernel AS RESTRUCTURED BY gpu-sim-u2o.6.3 in master @
c4072f3 -- a wrapper opening the per-run scratch directory plus
_run_kernel_inner carrying the supervise/timeout/kill path. The pre-6.3
single-function form was covered by the 20260928 dose (ts
20260928t072424-9816875f); this dose re-covers the kill path after the
restructure, per criterion 7 of that bead.

Deliberately NOT done here, per the bead and the shared-queue policy: no clock
locking, no sudo, no persistence-mode change (6.4 stays held), and no second
hang. One dose.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness   # noqa: E402
from epic_u2o.kernel_driver import spec_from_stage1, synthesize  # noqa: E402
from epic_u2o import queue_state as qs          # noqa: E402

HANG_TIMEOUT_SEC = 4          # 2-5 s, per the bead. Short on purpose.
MASTER_REV = "c4072f3"

# The bead's existing hang kernel, verbatim from safe_harness --test-hang: a
# volatile spin, because a bare `while(1){}` is deleted by nvcc under
# forward-progress rules and exits in ~2 ms.
HANG_KERNEL = r'''
__global__ void hang_kernel(volatile int *flag) {
    while (*flag == 0) { /* spin forever; host never sets the flag */ }
}
int main() {
    int *d_flag = NULL;
    cudaMalloc((void**)&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));
    hang_kernel<<<1,1>>>((volatile int*)d_flag);
    cudaDeviceSynchronize();
    return 0;
}
'''


def step_state(label: str) -> dict:
    """Queue state + telemetry for ONE step.

    The reviewer's item on 6.10: the queue record was correct but attached to
    nothing -- it was sampled once at the start of the run and then applied,
    implicitly, to all three steps. A record that is not attached to the
    measurement it governs cannot be checked against it. So every step samples
    its own, and the artifact carries one per step from the next run on.
    """
    q = qs.queue_state(own_description_substr="u2o.6.10",
                       own_command_substr="validate_once.py")
    return {
        "label": label,
        "recorded_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "own_job_id": q.get("own_job_id"),
        "own_job_in_active_set": q.get("own_job_in_active_set"),
        "own_job_entry_present": q.get("own_job_entry") is not None,
        "n_other_active": q.get("n_other_active"),
        "n_other_running_at_sample_time": q.get("n_other_running_at_sample_time"),
        "queue_was_empty": q.get("queue_was_empty"),
        "telemetry": telemetry(),
    }


def telemetry():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,clocks.sm,temperature.gpu,utilization.gpu,"
             "power.draw,memory.used,clocks_throttle_reasons.active",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return {"rc": r.returncode, "csv": r.stdout.strip()}
    except Exception as e:
        return {"error": str(e)}


def trials_from(stdout):
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "time_ms" in line:
            try:
                out.append(float(json.loads(line)["time_ms"]))
            except (ValueError, KeyError):
                pass
    return out


def fresh_report() -> dict:
    return {
        "bead": "gpu-sim-u2o.6.10",
        "master_revision_exercised": MASTER_REV,
        "coverage_limit": (
            "run_kernel as restructured by gpu-sim-u2o.6.3 in this master "
            "revision (wrapper + _run_kernel_inner). The pre-6.3 "
            "single-function form was covered by the 20260928 dose; this "
            "dose re-covers the kill path after the restructure."),
        "hang_timeout_sec": HANG_TIMEOUT_SEC,
        "doses": 1,
        "staged": True,
    }


def do_step1(h, report: dict):
    """A normal kernel: prove timings are device-side. Returns (step1, exe)."""
    print("\n--- STEP 1: normal stage-1 kernel ---", flush=True)
    spec = spec_from_stage1("saxpy", [32, 1, 1], [256, 1, 1], 8192)
    exe = HERE / "normal.out"
    ok, msg = h.compile_kernel(synthesize(spec), str(exe), arch="sm_120")
    print(f"  compile: ok={ok} {str(msg)[:100]}")
    if not ok:
        return {"error": f"compile failed: {msg[:400]}"}, None
    t0 = time.time()
    r1 = h.run_kernel(str(exe))
    wall = time.time() - t0
    t1 = trials_from(r1.stdout)
    step1 = {
        "step_state": step_state("step1_normal_kernel"),
        "kernel": spec["kernel_name"],
        "n_problem_size": spec["problem_size"],
        "grid": spec["grid"], "block": spec["block"],
        "success": r1.success,
        "n_trials": len(t1),
        "median_trial_ms": statistics.median(t1) if t1 else None,
        "min_trial_ms": min(t1) if t1 else None,
        "max_trial_ms": max(t1) if t1 else None,
        "process_wall_sec": wall,
        "queue_state": r1.metadata.get("task_limits"),
    }
    # The discriminator: the sum of device-side trial times is orders of
    # magnitude below the process wall time, because wall time is dominated by
    # CUDA context creation and process teardown. If the two agreed, the
    # "measurement" would be timing process startup.
    if t1:
        # UNITS: `wall` is SECONDS and sum(t1) is MILLISECONDS. Dividing one
        # by the other is a unit error -- it printed "3x" where the real
        # ratio is ~2700x, and it failed the device-side check on my own
        # arithmetic rather than on anything about the measurement.
        wall_ms = wall * 1000.0
        step1["sum_of_trials_ms"] = sum(t1)
        step1["process_wall_ms"] = wall_ms
        step1["wall_over_trials_ratio"] = wall_ms / max(sum(t1), 1e-9)
        step1["timing_is_device_side"] = wall_ms > 10 * sum(t1)
    report["step1"] = step1
    print(f"  success={r1.success} trials={len(t1)} "
          f"median={step1['median_trial_ms']} ms wall={wall:.2f}s "
          f"ratio={step1.get('wall_over_trials_ratio'):.0f}x "
          f"device_side={step1.get('timing_is_device_side')}")
    return step1, exe


def do_step2(h, report: dict):
    """The hang kernel, ONCE. Returns step2. Refuses unless step 1 nominal."""
    s1 = report.get("step1") or {}
    if not (s1.get("success") and s1.get("timing_is_device_side")):
        print("REFUSAL: step 1 is not nominal (need success + "
              "timing_is_device_side); the hang is NOT spent.", flush=True)
        return None
    print(f"\n--- STEP 2: --test-hang kernel, ONCE, "
          f"{HANG_TIMEOUT_SEC}s timeout, WATCHED ---", flush=True)
    hang_exe = HERE / "hang.out"
    ok2, msg2 = h.compile_kernel(HANG_KERNEL, str(hang_exe), arch="sm_120")
    print(f"  compile: ok={ok2} {str(msg2)[:100]}")
    if not ok2:
        return {"error": f"compile failed: {msg2[:400]}"}
    step2_state = step_state("step2_hang_kernel")
    hb = SafeHarness(timeout_sec=HANG_TIMEOUT_SEC)
    t0 = time.time()
    r2 = hb.run_kernel(str(hang_exe))
    hang_wall = time.time() - t0
    step2 = {
        "step_state": step2_state,
        "timeout_reported": r2.timeout,
        "returncode": r2.returncode,
        "wall_sec": hang_wall,
        "configured_timeout_sec": HANG_TIMEOUT_SEC,
        "stderr": r2.stderr[:300],
        "gpu_health": r2.metadata.get("gpu_health"),
        "killed_within_timeout": (
            r2.timeout and hang_wall < 3 * HANG_TIMEOUT_SEC),
    }
    report["step2"] = step2
    print(f"  timeout={r2.timeout} rc={r2.returncode} wall={hang_wall:.2f}s "
          f"health={r2.metadata.get('gpu_health')}")

    # Did the runaway actually die? A reported timeout is a claim; this is
    # the check. Matched on the binary path, which only the hang process has.
    survivors = subprocess.run(["pgrep", "-f", str(hang_exe)],
                               capture_output=True, text=True)
    step2["survivor_pids"] = [x for x in survivors.stdout.split() if x.strip()]
    step2["child_is_dead"] = not step2["survivor_pids"]
    print(f"  survivors: {step2['survivor_pids'] or 'none'} "
          f"-> child_dead={step2['child_is_dead']}")
    return step2


def do_step2b(h, report: dict, exe):
    """Recovery probe. Refuses unless step 2 shows a clean kill."""
    s2 = report.get("step2") or {}
    if not (s2.get("timeout_reported") and s2.get("child_is_dead")):
        print("REFUSAL: step 2 is not a verified-clean kill (need "
              "timeout_reported + child_is_dead); no recovery probe.",
              flush=True)
        return None
    s1 = report.get("step1") or {}
    print("\n--- STEP 2b: RECOVERY PROBE -- normal kernel after the kill ---",
          flush=True)
    time.sleep(2.0)
    step2b_state = step_state("step2b_recovery_probe")
    t0 = time.time()
    r3 = h.run_kernel(str(exe))
    t2 = trials_from(r3.stdout)
    step2b = {
        "step_state": step2b_state,
        "success": r3.success,
        "n_trials": len(t2),
        "median_trial_ms": statistics.median(t2) if t2 else None,
        "wall_sec": time.time() - t0,
        "recovered": bool(r3.success and t2),
        "median_vs_pre_hang_ratio": (
            (statistics.median(t2) / s1["median_trial_ms"])
            if t2 and s1.get("median_trial_ms") else None),
    }
    report["step2b_recovery"] = step2b
    print(f"  success={r3.success} trials={len(t2)} "
          f"median={step2b['median_trial_ms']} ms "
          f"vs pre-hang {s1.get('median_trial_ms')} ms "
          f"ratio={step2b['median_vs_pre_hang_ratio']}")
    return step2b


def build_verdict(report: dict) -> dict:
    step1 = report.get("step1") or {}
    step2 = report.get("step2") or {}
    step2b = report.get("step2b_recovery") or {}
    return {
        "step1_device_side_timing": step1.get("timing_is_device_side"),
        "hang_detected": step2.get("timeout_reported"),
        "hang_killed_within_timeout": step2.get("killed_within_timeout"),
        "hang_child_dead": step2.get("child_is_dead"),
        "gpu_recovered": step2b.get("recovered"),
        "overall_pass": bool(
            step1.get("success") and step1.get("timing_is_device_side")
            and step2.get("timeout_reported")
            and step2.get("child_is_dead")
            and step2b.get("recovered")),
    }


def load_stage(name: str) -> dict:
    p = HERE / f"report_{name}.json"
    if not p.exists():
        print(f"REFUSAL: {p.name} missing; stages run in order.", flush=True)
        raise SystemExit(2)
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=("all", "step1", "step2", "step2b", "assemble"),
                    help="run one stage per ts job (default: all, one-shot)")
    stage = ap.parse_args().only
    print("=" * 74)
    print(f"gpu-sim-u2o.6.10 -- supervised single-dose validation (master @{MASTER_REV})")
    print("=" * 74)

    if stage == "assemble":
        # CPU-only merge of the three stage files. No GPU here.
        report = fresh_report()
        for name, key in (("step1", "step1"), ("step2", "step2"),
                          ("step2b", "step2b_recovery")):
            part = load_stage(name)
            report[key] = part[key]
            report[f"{name}_ts_job"] = part.get("ts_job")
        report["queue_at_start"] = load_stage("step1").get("queue_at_start")
        report["telemetry_start"] = load_stage("step1").get("telemetry_start")
        report["verdict"] = build_verdict(report)
        (HERE / "report.json").write_text(json.dumps(report, indent=2))
        print("=== VERDICT ===")
        for k, v in report["verdict"].items():
            print(f"  {k:32s} {v}")
        return 0

    q = qs.queue_state(own_description_substr="u2o.6.10",
                       own_command_substr="validate_once.py")
    print("queue at start: " + json.dumps(
        {k: q.get(k) for k in ("own_job_id", "n_other_active",
                               "n_other_running_at_sample_time",
                               "own_job_in_active_set")}))
    h = SafeHarness(timeout_sec=120)

    if stage == "step1":
        report = fresh_report()
        if os.environ.get("TS_JOB_ID"):
            report["ts_job"] = os.environ["TS_JOB_ID"]
        # else ABSENT: gpqueue exports no job id into the job, and a null
        # ts_job would read as coverage. The submitter backfills the real id.
        report["queue_at_start"] = q
        report["telemetry_start"] = telemetry()
        step1, _ = do_step1(h, report)
        (HERE / "report_step1.json").write_text(json.dumps(report, indent=2))
        ok = bool(step1.get("success") and step1.get("timing_is_device_side"))
        print(f"STEP1 nominal={ok}")
        return 0 if ok else 1

    if stage == "step2":
        prev = load_stage("step1")
        report = fresh_report()
        if os.environ.get("TS_JOB_ID"):
            report["ts_job"] = os.environ["TS_JOB_ID"]
        # else ABSENT: gpqueue exports no job id into the job, and a null
        # ts_job would read as coverage. The submitter backfills the real id.
        report["step1"] = prev["step1"]
        report["queue_at_start"] = prev.get("queue_at_start")
        step2 = do_step2(h, report)
        if step2 is None:
            return 2
        (HERE / "report_step2.json").write_text(json.dumps(report, indent=2))
        ok = bool(step2.get("timeout_reported") and step2.get("child_is_dead"))
        print(f"STEP2 clean_kill={ok}")
        return 0 if ok else 1

    if stage == "step2b":
        prev1 = load_stage("step1")
        prev2 = load_stage("step2")
        report = fresh_report()
        if os.environ.get("TS_JOB_ID"):
            report["ts_job"] = os.environ["TS_JOB_ID"]
        # else ABSENT: gpqueue exports no job id into the job, and a null
        # ts_job would read as coverage. The submitter backfills the real id.
        report["step1"] = prev1["step1"]
        report["step2"] = prev2["step2"]
        exe = HERE / "normal.out"
        if not exe.exists():
            # Stage jobs share the worktree, but build products may not survive
            # across queue slots; recompile rather than assuming normal.out.
            spec = spec_from_stage1("saxpy", [32, 1, 1], [256, 1, 1], 8192)
            okc, msgc = h.compile_kernel(synthesize(spec), str(exe),
                                         arch="sm_120")
            if not okc:
                print(f"REFUSAL: recovery kernel recompile failed: {msgc[:200]}")
                return 2
        step2b = do_step2b(h, report, str(exe))
        if step2b is None:
            return 2
        (HERE / "report_step2b.json").write_text(json.dumps(report, indent=2))
        print(f"STEP2b recovered={step2b.get('recovered')} "
              f"ratio={step2b.get('median_vs_pre_hang_ratio')}")
        return 0 if step2b.get("recovered") else 1

    # stage == "all": original one-shot shape (preflight-compatible).
    report = fresh_report()
    report["queue_at_start"] = q
    report["telemetry_start"] = telemetry()
    step1, exe = do_step1(h, report)
    if step1.get("error"):
        (HERE / "report.json").write_text(json.dumps(report, indent=2))
        return 1
    step2 = do_step2(h, report)
    if step2 is None or step2.get("error"):
        (HERE / "report.json").write_text(json.dumps(report, indent=2))
        return 2 if step2 is None else 1
    step2b = do_step2b(h, report, str(exe))
    if step2b is None:
        (HERE / "report.json").write_text(json.dumps(report, indent=2))
        return 2
    report["telemetry_end"] = telemetry()
    report["verdict"] = build_verdict(report)
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print("\n=== VERDICT ===")
    for k, v in report["verdict"].items():
        print(f"  {k:32s} {v}")
    return 0 if report["verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
