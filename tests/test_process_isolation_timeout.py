#!/usr/bin/env python3
"""Process isolation + timeout/watchdog for untrusted kernels (gpu-sim-u2o.6.1).

CPU-only. Exercises the REAL SafeHarness.run_kernel subprocess path
(Popen + start_new_session + communicate(timeout) + killpg SIGKILL) against
compiled binaries -- no GPU, no ts queue:

  * a normal binary must succeed with captured stdout,
  * a two-process hang binary (parent forks a spinner, both spin forever)
    must be reported timeout=True with wall time inside the configured budget,
  * BOTH hang processes must be dead afterwards (the kill is a process-group
    SIGKILL, so a forked child dies with its parent),
  * the pipeline must continue: a normal binary runs fine right after the kill.

One disclosed stub: run_kernel's timeout path ends in
_check_gpu_health_once(), which shells `nvidia-smi -q`. GPU access outside
the shared ts queue is banned, and a CPU test must never hold the queue, so
the health probe is stubbed IN THE TEST PROCESS ONLY and recorded as such in
the metrics (`gpu_health_probe: stubbed`). The kill/timeout/containment path
under test is fully real. The genuine post-hang GPU health probe is the
supervised dose in gpu-sim-u2o.6.10, not this bead.

Usage:
  CUDA_VISIBLE_DEVICES= python3 tests/test_process_isolation_timeout.py \
      --metrics-out /tmp/m.json
Exits nonzero on any failed assertion; writes metrics JSON to --metrics-out.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO))

import epic_u2o.safe_harness as sh  # noqa: E402

TIMEOUT_SEC = 3
# Upper bound: kill must be prompt, but this is a shared box (cclm/chessvar
# jobs), so allow generous scheduling slack. The assertion that matters is
# wall >= TIMEOUT_SEC (it waited the full budget) and wall << runaway.
WALL_SLACK_SEC = 15

NORMAL_C = r'''
#include <stdio.h>
int main(void) { printf("ok\n"); return 0; }
'''

# Parent forks one spinner child, records both pids, then both spin forever.
# Unoptimizable (no volatile needed: the loop has a live syscall-free body the
# compiler cannot prove terminates -- checked: binary spins until killed).
HANG_C = r'''
#include <stdio.h>
#include <unistd.h>
int main(int argc, char **argv) {
  pid_t c = fork();
  if (c == 0) { while (1) {} }
  FILE *f = fopen(argv[1], "w");
  if (!f) return 2;
  fprintf(f, "%d %d\n", (int)getpid(), (int)c);
  fclose(f);
  while (1) {}
  return 0;
}
'''

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail)[:200] if detail else ''}",
          flush=True)
    if not ok:
        FAILS.append(name)


def compile_c(src, tmp, name):
    cu = tmp / f"{name}.c"
    exe = tmp / name
    cu.write_text(src)
    r = subprocess.run(["gcc", "-O2", str(cu), "-o", str(exe)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"gcc failed: {r.stderr[:500]}"
    return str(exe)


def pid_dead(pid, settle_sec=2.0):
    """True once kill(pid,0) raises ESRCH. Reaps own zombies first: a SIGKILLed
    direct child lingers as a zombie (kill succeeds on zombies) until waited."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass  # not our child (reparented grandchild) -- nothing to reap
    except OSError:
        pass
    end = time.time() + settle_sec
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if time.time() > end:
            return False
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()

    # Disclosed stub (see module docstring): the post-timeout GPU health probe
    # would shell nvidia-smi outside the queue. Everything else stays real.
    sh._check_gpu_health_once = lambda gpu_id=0: ("skipped-cpu-test", False)

    harness = sh.SafeHarness(timeout_sec=TIMEOUT_SEC)
    metrics = {
        "bead": "gpu-sim-u2o.6.1",
        "timeout_sec": TIMEOUT_SEC,
        "gpu_health_probe": "stubbed (returns 'skipped-cpu-test'); "
                            "real post-hang probe is gpu-sim-u2o.6.10",
    }

    with tempfile.TemporaryDirectory(prefix="u2o61-") as td:
        tmp = Path(td)
        normal = compile_c(NORMAL_C, tmp, "normal")
        hang = compile_c(HANG_C, tmp, "hang")
        pidfile = str(tmp / "hang.pids")

        # 1. Normal binary succeeds, stdout captured.
        r1 = harness.run_kernel(normal)
        check("normal.succeeds", r1.success and r1.returncode == 0,
              f"rc={r1.returncode} err={r1.stderr[:200]}")
        check("normal.stdout_captured", r1.stdout.strip() == "ok",
              repr(r1.stdout[:100]))
        check("normal.no_timeout", not r1.timeout)
        metrics["normal"] = {"success": r1.success,
                             "returncode": r1.returncode,
                             "timeout": r1.timeout,
                             "wall_time_sec": round(r1.wall_time_sec, 3)}

        # 2. Hang binary is killed within budget.
        t0 = time.time()
        r2 = harness.run_kernel(hang, args=[pidfile])
        kill_wall = time.time() - t0
        check("hang.timeout_reported", r2.timeout and not r2.success,
              f"timeout={r2.timeout} rc={r2.returncode}")
        check("hang.waited_full_budget",
              r2.wall_time_sec >= TIMEOUT_SEC,
              f"wall={r2.wall_time_sec:.2f}s")
        check("hang.killed_promptly",
              r2.wall_time_sec <= TIMEOUT_SEC + WALL_SLACK_SEC,
              f"wall={r2.wall_time_sec:.2f}s")
        task_info = r2.metadata.get("task_limits", {})
        metrics["hang"] = {"timeout": r2.timeout, "success": r2.success,
                           "returncode": r2.returncode,
                           "wall_time_sec": round(r2.wall_time_sec, 3),
                           "task_limits": task_info}

        # 3. No survivors: both hang processes dead (process-group SIGKILL).
        try:
            pids = [int(x) for x in Path(pidfile).read_text().split()]
        except Exception as e:
            pids = []
            check("hang.pidfile_written", False, str(e))
        if pids:
            check("hang.pidfile_written", len(pids) == 2, str(pids))
            dead = {p: pid_dead(p) for p in pids}
            check("hang.no_survivors", all(dead.values()), str(dead))
            metrics["hang"]["pids"] = pids
            metrics["hang"]["survivors"] = [p for p, d in dead.items()
                                            if not d]
        metrics["hang"]["kill_wall_sec"] = round(kill_wall, 3)

        # 4. Pipeline continues without manual intervention.
        r3 = harness.run_kernel(normal)
        check("pipeline.continues", r3.success and not r3.timeout,
              f"rc={r3.returncode}")
        metrics["pipeline_continues"] = r3.success and not r3.timeout

    metrics["verdict"] = "pass" if not FAILS else "fail"
    metrics["failures"] = FAILS
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"verdict={metrics['verdict']} -> {args.metrics_out}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
