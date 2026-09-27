#!/usr/bin/env python3
"""Does the per-run task cap actually FIRE? (gpu-sim-u2o.6.1.2, CPU-only)

The bead's whole point is enforcement, not configuration: creating a unit with
TasksMax=64 proves nothing until something actually fails to fork under it.
This test does the boring version of that:

  * a child that forks past the cap must fail with EAGAIN/EPERM,
  * the SAME child without the cap must succeed,
  * and a realistic kernel driver (a handful of threads) must NOT be tripped
    by the default cap -- otherwise containment would break the work it is
    meant to protect.

No GPU and no ts: CUDA_VISIBLE_DEVICES= and a CPU-only child. The GPU-side
proof (a real kernel still completing under the cap) is a separate ts job,
recorded in experiments/gpu-sim-u2o.6.1.2/metrics.json.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]        # tests/ -> worktree root
sys.path.insert(0, str(REPO))

from epic_u2o import task_limits as tl  # noqa: E402

TASKS_MAX = 64           # small on purpose, so enforcement is observable
FORKS = 200

CHILD = r'''
import errno, os, sys, time
want = int(sys.argv[1])
made = 0
try:
    for _ in range(want):
        pid = os.fork()
        if pid == 0:
            time.sleep(0.05)
            os._exit(0)
        made += 1
except OSError as e:
    print(json.dumps({"blocked": True, "children": made,
                      "errno": errno.errorcode.get(e.errno, e.errno),
                      "msg": e.strerror}))
    sys.exit(3)
print(json.dumps({"blocked": False, "children": made}))
'''

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail)[:160] if detail else ''}")
    if not ok:
        FAILS.append(name)


def run(argv, env=None):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = ""      # this test is CPU-only by design
    e.update(env or {})
    r = subprocess.run(argv, capture_output=True, text=True, timeout=180, env=e)
    payload = None
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except ValueError:
                pass
    return r, payload


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    child = tmp / "forker.py"
    child.write_text("import json\n" + CHILD)

    mech = tl.choose_mechanism()
    print(f"mechanism: {mech.name}  ({mech.detail})")
    print(f"live task count for uid {os.getuid()}: {tl.live_task_count()}")
    check("mechanism is a cgroup route, not RLIMIT_NPROC",
          mech.name in ("systemd-run", "cgroup"), mech.name)

    # --- 1. no cap: the child succeeds ---------------------------------
    t0 = time.time()
    r_uncapped, p_uncapped = run([sys.executable, str(child), str(FORKS)])
    check("uncapped child forks to completion",
          p_uncapped is not None and not p_uncapped.get("blocked"),
          f"{p_uncapped} in {time.time() - t0:.1f}s")

    # --- 2. with the cap: the child is blocked -------------------------
    if mech.name == "systemd-run":
        argv = tl.wrap_command([sys.executable, str(child), str(FORKS)],
                               mech, TASKS_MAX)
    else:
        argv = [sys.executable, str(child), str(FORKS)]
    t0 = time.time()
    r_capped, p_capped = run(argv)
    check("capped child FAILS to fork past the cap",
          p_capped is not None and p_capped.get("blocked") is True,
          f"{p_capped} in {time.time() - t0:.1f}s")
    check("the failure is EAGAIN/EPERM, not something else",
          p_capped and p_capped.get("errno") in ("EAGAIN", "EPERM"),
          p_capped.get("errno") if p_capped else None)
    check("the cap fired at roughly the configured task count",
          p_capped and p_capped["children"] <= TASKS_MAX,
          f"blocked after {p_capped['children'] if p_capped else '?'} "
          f"children, TasksMax={TASKS_MAX}")

    # --- 3. a realistic driver must NOT be tripped by the default cap ----
    if mech.name == "systemd-run":
        argv = tl.wrap_command([sys.executable, str(child), "8"], mech,
                               tl.DEFAULT_TASKS_MAX)
    else:
        argv = [sys.executable, str(child), "8"]
    _, p_default = run(argv)
    check(f"a small child is unaffected by the default cap "
          f"(TasksMax={tl.DEFAULT_TASKS_MAX})",
          p_default is not None and not p_default.get("blocked"), p_default)

    # --- 4. the fallback must be relative, never a constant -------------
    pre = tl.rlimit_nproc_preexec()
    live = tl.live_task_count()
    check("RLIMIT_NPROC fallback is computed relative to the live count",
          getattr(pre, "target_nproc", 0) > live,
          f"live={live} -> nproc={getattr(pre, 'target_nproc', None)}")
    check("the fallback is mechanism 3, not the default",
          tl.probe()[0].available and tl.choose_mechanism().name != "rlimit_nproc",
          tl.choose_mechanism().name)

    print()
    print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
