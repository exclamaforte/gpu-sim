#!/usr/bin/env python3
"""gpu-sim-u2o.6.2 recovery-exercise verdict (CPU): the acceptance criteria
as machine gates over the committed dose steps.

Gates (verdict=pass iff ALL hold; criterion numbers are the bead's):
  AC1 exercised: arm_a (timeout kill) + arm_b (SIGTERM-survived holder) both
        ran; state_before/after present; ts job id in recover_job.json.
  AC2 no over-escalation: every recorded decision in {none, soft_reset}; the
        strings hard_reset/driver_reload/system_reboot appear NOWHERE in the
        steps; arm_b decision is soft_reset with DECLINED + named foreign
        holders in the message.
  AC3 subsequent measurement: calibrator_pre/post_a/post_b all numbers with
        rc 0; |post-pre|/pre <= 0.25 for both arms (pre-registered; generous
        for thermal drift -- the gate is "returns numbers that agree", not
        "identical").
  AC4 survivor: arm_b survivor_pids non-empty; arm_a spin holder list empty.
  AC5 not-covered: the artifact states the fixed NOT_COVERED list (the
        analyzer emits it; a restructure riding on this green is the failure
        mode 6.10 named).
  AC6 named check: arm records carry the health value; arm_b decline names
        the foreign holders (independent nvidia-smi observation, never a
        harness self-report alone).
An arm_b_aborted record (safety interlock fired) fails the exercise honestly:
the survivor path was not covered.

Deterministic; writes metrics JSON to --metrics-out.
Usage: python3 experiments/gpu-sim-u2o.6.2/recover_analyze.py --metrics-out <p>
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = HERE / "recover_steps_20261007.jsonl"
JOB = HERE / "recover_job.json"
ALLOWED = {"none", "soft_reset"}
BANNED = ("hard_reset", "driver_reload", "system_reboot")
CAL_TOL = 0.25
NOT_COVERED = [
    "an actual nvidia-smi -r pulse (declined by the foreign-holder gate)",
    "a true Xid/off-bus wedge (only a simulated hang + a SIGTERM-ignorer)",
    "the MANUAL-ONLY rungs (hard reset, driver reload, reboot)",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    fails = []
    steps = [json.loads(l) for l in STEPS.read_text().splitlines() if l.strip()]
    by = {s["step"]: s for s in steps}
    job = json.loads(JOB.read_text())
    jobs = job.get("gpu_jobs", [])
    if not jobs or not jobs[0].get("job_id"):
        fails.append("AC1: no ts job id recorded")
    for need in ("state_before", "calibrator_pre", "arm_a_hang",
                 "arm_b_survivor", "state_after"):
        if need not in by:
            fails.append(f"AC1: missing step {need}")
    if "arm_b_aborted" in by or "arm_b_aborted_after_survivor" in by or "arm_b_no_holder" in by:
        fails.append("AC1: safety interlock fired -- survivor path not covered")

    blob = "\n".join(json.dumps(s) for s in steps)
    for b in BANNED:
        if b in blob:
            fails.append(f"AC2: escalation rung present in record: {b}")
    a, b = by.get("arm_a_hang", {}), by.get("arm_b_survivor", {})
    if a.get("decision", "none") not in ALLOWED:
        fails.append(f"AC2: arm_a decision out of ruled ladder: {a.get('decision')}")
    if b.get("decision") != "soft_reset" or "DECLINED" not in b.get("decision_msg", ""):
        fails.append("AC2: arm_b must be soft_reset DECLINED "
                     f"(got {b.get('decision')}: {b.get('decision_msg', '')[:80]})")
    if "foreign" not in b.get("decision_msg", "").lower():
        fails.append("AC6: decline names no foreign holders")

    pre = by.get("calibrator_pre", {}).get("median_ms")
    pre_rc = by.get("calibrator_pre", {}).get("rc")
    if pre_rc != 0:
        fails.append(f"AC3: calibrator_pre rc={pre_rc}, not numbers")
    for arm, key in (("A", a), ("B", b)):
        post = key.get("calibrator_post_ms")
        rc = key.get("calibrator_rc")
        if rc != 0:
            fails.append(f"AC3: arm {arm} calibrator rc={rc}, not numbers")
        if pre is None or post is None:
            fails.append(f"AC3: arm {arm} missing a calibrator median")
            continue
        drift = abs(post - pre) / pre if pre else 999.0
        if drift > CAL_TOL:
            fails.append(f"AC3: arm {arm} post/pre drift {drift:.3f} > {CAL_TOL}")

    if not b.get("survivor_pids"):
        fails.append("AC4: arm_b recorded no survivor")
    if a.get("spin_still_holding_gpu"):
        fails.append(f"AC4: arm_a left a holder: {a.get('spin_still_holding_gpu')}")
    if not a.get("timeout"):
        fails.append("AC1: arm_a induced no timeout kill")
    if not b.get("sigterm_survived"):
        fails.append("AC1: arm_b holder did not survive SIGTERM")
    if not b.get("surface_file"):
        fails.append("AC2: arm_b wrote no SURFACE_TO_OWNER flag")
    if not b.get("holder_sigkilled"):
        fails.append("cleanup: arm_b holder not SIGKILL-reaped")

    metrics = {
        "bead": "gpu-sim-u2o.6.2",
        "packet": "recovery-exercise",
        "ts_job": jobs[0].get("job_id") if jobs else None,
        "arm_a": {"timeout_kill": bool(a.get("timeout")),
                  "survivor": a.get("spin_still_holding_gpu", "missing"),
                  "health": a.get("health"), "decision": a.get("decision"),
                  "post_vs_pre": _ratio(by, a)},
        "arm_b": {"sigterm_survived": bool(b.get("sigterm_survived")),
                  "survivor_pids": b.get("survivor_pids"),
                  "health": b.get("health"), "decision": b.get("decision"),
                  "surface_file": bool(b.get("surface_file")),
                  "post_vs_pre": _ratio(by, b)},
        "not_covered": NOT_COVERED,
        "verdict": "pass" if not fails else "fail",
        "failures": fails,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"arm_a kill={a.get('timeout')} dec={a.get('decision')} "
          f"arm_b surv={bool(b.get('survivor_pids'))} dec={b.get('decision')} "
          f"verdict={metrics['verdict']} -> {args.metrics_out}", flush=True)
    return 1 if fails else 0


def _ratio(by, arm):
    pre = by.get("calibrator_pre", {}).get("median_ms")
    post = arm.get("calibrator_post_ms")
    if pre is None or post is None or not pre:
        return None
    return round(post / pre, 4)


if __name__ == "__main__":
    raise SystemExit(main())
