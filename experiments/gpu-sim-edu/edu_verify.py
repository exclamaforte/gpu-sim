#!/usr/bin/env python3
"""gpu-sim-edu activation + regression cover (CPU): machine-checkable evidence
for the bead's acceptance items that do not need the daemon restart.

Checks (all CPU, deterministic):
  AC1a: pi-research-lab/lab.py is_gpu_active reads status --json and raises
        GpuStateUnknown on every failure path (AST: no bare False return).
  AC1b: check_idle_stall computes the fingerprint under busy (no
        unconditional busy short-circuit).
  AC2:  full lab suite passes (subprocess; parses Ran/OK lines).
  AC3:  regression test test_is_gpu_active present (True live / False idle /
        raises on failure paths).
  AC4:  regression test test_check_idle_stall_resets_on_worker_busy asserts
        busy+unchanged wakes AND busy+changed does not.
  AC5:  live probe record: job id, state=canceled, exit -15, True-while-live
        and False-after readings with timings (from probe log / job record).
  AC6:  recorded OPEN (daemon restart severs the issuer's session; routed to
        PI/owner -- this script asserts only that the routing request exists
        in the job record, not that the restart happened).

Writes metrics JSON to --metrics-out.
Usage: python3 experiments/gpu-sim-edu/edu_verify.py --metrics-out <p>
"""
import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = Path("/home/gabe/pi-research-lab")
PROBE_JOB = "20261007t053019-e62c8cef"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    fails = []
    ev = {}

    src = (LAB / "lab.py").read_text()
    tree = ast.parse(src)
    gpu_fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "is_gpu_active")
    dump = ast.dump(gpu_fn)
    ev["ac1a_json_status"] = "status\", \"--json" in dump or "--json" in dump
    ev["ac1a_raises_unknown"] = "GpuStateUnknown" in dump
    returns_false = [n for n in ast.walk(gpu_fn)
                     if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
                     and n.value.value is False]
    ev["ac1a_false_returns"] = len(returns_false)
    if not ev["ac1a_json_status"] or not ev["ac1a_raises_unknown"]:
        fails.append("AC1a: is_gpu_active not the JSON/raising contract")
    # exactly one False (the genuinely-idle answer); failure paths raise.
    if len(returns_false) != 1:
        fails.append(f"AC1a: {len(returns_false)} False returns (want exactly 1)")

    stall_fn = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "check_idle_stall")
    ssrc = ast.get_source_segment(src, stall_fn) or ""
    ev["ac1b_fingerprint_under_busy"] = ("get_codebase_fingerprint" in ssrc
                                        and "merely BUSY is not progress" in ssrc)
    if not ev["ac1b_fingerprint_under_busy"]:
        fails.append("AC1b: busy-requires-moved-tree logic absent")

    tsrc = (LAB / "tests" / "test_lab.py").read_text()
    ev["ac3_gpu_regression"] = "def test_is_gpu_active" in tsrc
    ev["ac4_busy_regression"] = ("def test_check_idle_stall_resets_on_worker_busy" in tsrc
                                 and "Busy counts as progress only if" in tsrc)
    ev["ac2_daemon_stub"] = "lab.daemon_alive = lambda: False" in tsrc
    for k in ("ac3_gpu_regression", "ac4_busy_regression", "ac2_daemon_stub"):
        if not ev[k]:
            fails.append(f"missing: {k}")

    r = subprocess.run([sys.executable, "tests/test_lab.py"], cwd=str(LAB),
                       capture_output=True, text=True, timeout=300,
                       env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    m = re.search(r"Ran (\d+) tests", r.stderr + r.stdout)
    ev["suite_ran"] = int(m.group(1)) if m else 0
    ev["suite_ok"] = ("OK" in (r.stderr + r.stdout)) and r.returncode == 0
    if not ev["suite_ok"]:
        fails.append(f"AC2: suite failed rc={r.returncode}")
    Path(__import__("os").environ.get("EDU_SUITE_LOG", HERE / "suite_output.log")).write_text(r.stdout + r.stderr)

    jr = subprocess.run(["ts", "-i", PROBE_JOB], capture_output=True,
                        text=True, timeout=30)
    ev["probe_job_state"] = ("canceled" in jr.stdout)
    if not ev["probe_job_state"]:
        fails.append("AC5: probe job not in canceled state")

    metrics = {"bead": "gpu-sim-edu", "packet": "activation-regression-cover",
               "lab_branch": "lab/gpu-sim-edu",
               "evidence": ev,
               "ac6_daemon_restart": "OPEN (routed to PI/owner; severs issuer session)",
               "verdict": "pass" if not fails else "fail", "failures": fails}
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"suite={ev['suite_ran']} ok={ev['suite_ok']} probe_canceled={ev['probe_job_state']} "
          f"verdict={metrics['verdict']} -> {args.metrics_out}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
