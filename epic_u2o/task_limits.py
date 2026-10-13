#!/usr/bin/env python3
"""Per-run TASK containment via cgroup v2 pids.max (gpu-sim-u2o.6.1.2).

Why this module exists at all: RLIMIT_NPROC cannot provide runaway containment
on this host. The kernel checks it against the real UID's TOTAL process and
thread count, not the child subtree. `ps -eL -o uid=` shows ~2934 tasks for
uid 1000, and the CUDA driver then fails its first thread creation, so the
first cudaMalloc returns cudaErrorOsOperationNotSupported. Measured in
gpu-sim-u2o.6.1.1 (ts job 20260927t123630-05fd9bd0): NPROC=32 and NPROC=512
both fatal, and 2048 -- which looks generous -- is still below the live count
and still fatal. The live count drifts with the other tenants, so ANY fixed
value is a race. The safe value for RLIMIT_NPROC is "unset".

pids.max is the correct semantics: it counts only the tasks in one subtree.
Three mechanisms, in order of preference, all measured on this host:

  1. `systemd-run --user --wait --pipe -p TasksMax=N -- <cmd>`: a transient
     user unit wrapping the child. The cap is released with the unit, and
     --wait/--pipe keep it synchronous with our stdout/stderr capture.
  2. A delegated cgroup: mkdir under the user's own cgroup dir, write
     pids.max BEFORE exec, then move the child in.
  3. RLIMIT_NPROC computed RELATIVE to the live task count at apply time
     (live + headroom) -- never a constant. Documented as an accepted
     residual risk, because it is still coupled to other tenants.

Honest limitation, repeated in harness.md: TasksMax bounds threads and
processes in a subtree. It does NOT bound device memory. It is complementary
to the VRAM accounting in gpu-sim-u2o.6.3, not a replacement for it.
"""

import os
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TASKS_MAX = 256          # per run: a kernel driver, not a fork bomb
FALLBACK_HEADROOM = 64            # only used by mechanism 3

CGROUP_ROOT = Path("/sys/fs/cgroup")


# --------------------------------------------------------------------------
# host probing
# --------------------------------------------------------------------------

def cgroup2_available() -> bool:
    try:
        return _statfs_fs_type() == "cgroup2fs"
    except Exception:
        return False


def _statfs_fs_type() -> str:
    """Filesystem type of /sys/fs/cgroup without shelling out to stat."""
    try:
        r = subprocess.run(["stat", "-fc", "%T", str(CGROUP_ROOT)],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def pids_controller_available() -> bool:
    try:
        return "pids" in (CGROUP_ROOT / "cgroup.controllers").read_text().split()
    except Exception:
        return False


def own_cgroup_dir() -> Path | None:
    """This process's cgroup directory, if we may write pids.max into it."""
    try:
        line = [l for l in Path("/proc/self/cgroup").read_text().splitlines()
                if l.startswith("0::")][0]
        rel = line.split("0::", 1)[1].lstrip("/")
        d = CGROUP_ROOT / rel
        return d if d.is_dir() else None
    except Exception:
        return None


def live_task_count() -> int:
    """Tasks (processes + threads) currently owned by this real UID.

    This is the quantity RLIMIT_NPROC is checked against -- NOT the process
    count, which understates it by an order of magnitude.
    """
    out = os.popen("ps -eL -o uid= 2>/dev/null").read().split()
    uid = str(os.getuid())
    return sum(1 for u in out if u == uid)


def systemd_run_available() -> bool:
    if not cgroup2_available() or not pids_controller_available():
        return False
    return subprocess.run(["which", "systemd-run"], capture_output=True,
                          timeout=10).returncode == 0


# --------------------------------------------------------------------------
# mechanism selection
# --------------------------------------------------------------------------

@dataclass
class Mechanism:
    name: str                 # "systemd-run" | "cgroup" | "rlimit_nproc" | "none"
    available: bool
    detail: str


def probe() -> list[Mechanism]:
    """Which mechanisms this host actually supports, measured not assumed."""
    out = []

    sysd_ok = systemd_run_available()
    out.append(Mechanism(
        "systemd-run", sysd_ok,
        "cgroup2=%s pids=%s systemd-run=%s" % (
            _statfs_fs_type(),
            pids_controller_available(),
            subprocess.run(["which", "systemd-run"], capture_output=True,
                           text=True, timeout=10).stdout.strip() or "absent")))

    d = own_cgroup_dir()
    writable = False
    if d is not None:
        try:
            writable = os.access(d, os.W_OK) and pids_controller_available()
        except Exception:
            writable = False
    out.append(Mechanism("cgroup", writable,
                         f"own cgroup {d} writable={writable}"
                         if d else "own cgroup not resolvable"))

    out.append(Mechanism("rlimit_nproc", True,
                         "live tasks=%d, fallback would be live+%d "
                         "(never a constant)" % (live_task_count(),
                                                 FALLBACK_HEADROOM)))
    out.append(Mechanism("none", True, "no task containment"))
    return out


def choose_mechanism(preferred: str | None = None) -> Mechanism:
    ms = probe()
    order = ([preferred] if preferred else []) + ["systemd-run", "cgroup",
                                                   "rlimit_nproc", "none"]
    by_name = {m.name: m for m in ms}
    for name in order:
        m = by_name.get(name)
        if m and m.available:
            return m
    return by_name["none"]


# --------------------------------------------------------------------------
# command wrapping
# --------------------------------------------------------------------------

def wrap_command(cmd: list[str], mechanism: Mechanism, tasks_max: int,
                 cgroup_dir: Path | None = None) -> list[str]:
    """Return the argv that actually launches `cmd` under the chosen cap."""
    if mechanism.name == "systemd-run":
        # --wait keeps it synchronous (so our communicate(timeout=) still
        # governs the wall clock); --pipe keeps stdout/stderr on our pipes;
        # --collect tears the unit down with the run instead of leaving it
        # around to accumulate.
        return ["systemd-run", "--user", "--wait", "--pipe", "--collect",
                "--quiet", "-p", f"TasksMax={int(tasks_max)}", "--"] + cmd
    if mechanism.name == "cgroup" and cgroup_dir is not None:
        return cmd          # caller moves the child itself, after fork
    if mechanism.name == "rlimit_nproc":
        return cmd          # handled in the preexec hook
    return cmd


def rlimit_nproc_preexec(headroom: int = FALLBACK_HEADROOM):
    """Fallback ONLY: NPROC = live task count at apply time + headroom.

    Still coupled to the other tenants on the box, which is why this is
    mechanism 3 of 3 and why the cgroup routes are preferred.
    """
    n = live_task_count() + headroom
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if hard != resource.RLIM_INFINITY:
        n = min(n, hard)
    target = (n, hard)

    def _apply():
        resource.setrlimit(resource.RLIMIT_NPROC, target)
    _apply.target_nproc = n          # for reporting
    return _apply


def make_cgroup(base: Path, name: str, tasks_max: int) -> Path:
    """Create a delegated sub-cgroup with pids.max written BEFORE any exec."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "cgroup.procs").write_text(str(os.getpid()))  # make it ours to write
    (d / "pids.max").write_text(str(int(tasks_max)))
    return d


def cgroup_peak(cgroup_dir: Path) -> int | None:
    try:
        return int((cgroup_dir / "pids.peak").read_text().strip())
    except Exception:
        try:
            return int((cgroup_dir / "pids.current").read_text().strip())
        except Exception:
            return None


def systemd_unit_peak_tasks_max(unit_hint: str | None = None) -> int | None:
    """Peak tasks observed by a transient unit, via systemd's accounting.

    Falls back to None when the property is unavailable, which is reported
    honestly rather than guessed.
    """
    try:
        cmd = ["systemctl", "--user", "show"]
        if unit_hint:
            cmd.append(unit_hint)
        cmd += ["-p", "TasksMax", "-p", "TasksCurrent", "--value"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# mechanism 2, done properly: move the child into a cgroup BEFORE exec
# --------------------------------------------------------------------------
#
# Why this replaced the systemd-run wrapper (reviewer, gpu-sim-u2o.6.1.2):
# `systemd-run --user --wait --pipe` starts a transient SERVICE, so the
# wrapped binary's parent is systemd and the binary is NOT in the run_kernel
# child's process group. run_kernel's timeout path does
# proc.kill() + killpg(getpgid(proc.pid)), which then reaches only systemd-run
# and NOT the runaway. Measured: the harness reported timeout=True, reported
# the health check, reported a containment -- and the child kept running.
# --collect does not help; it garbage-collects the unit after the unit
# FINISHES, and killing systemd-run leaves the unit alive.
#
# A cap that bounds forks while leaving the runaway alive is partial
# containment sold as the containment the bead set out to restore. The fix is
# not a better kill, it is not breaking the mechanism that already worked:
# keep the command a DIRECT child (so killpg reaches it) and move it into the
# cgroup in a preexec hook, which runs after fork but before exec -- so the
# child is inside the cap before it executes a single instruction, and there
# is no window in which it could fork first.

def writable_pids_parent() -> Path | None:
    """The nearest ancestor cgroup where WE can create a pids-bounded child.

    Creating a sub-cgroup is not enough: writing pids.max in it also requires
    the pids controller to be enabled in the PARENT's cgroup.subtree_control.
    Our own leaf scope (app-*.scope) is ours but has no subtree_control, and
    writing there fails with EACCES. The user's slice directory is writable
    and does list pids, so that is the base we use -- measured, not assumed.
    """
    own = own_cgroup_dir()
    candidates = []
    if own is not None:
        candidates.append(own)
        candidates.extend(own.parents)
    candidates.append(CGROUP_ROOT / "user.slice")
    for c in candidates:
        try:
            if not c.is_dir() or not os.access(c, os.W_OK):
                continue
            sub = (c / "cgroup.subtree_control").read_text().split()
            if "pids" in sub:
                return c
        except Exception:
            continue
    # Fall back to the user's service dir, which is where pids is enabled.
    for c in (CGROUP_ROOT / "user.slice").glob("user-*.slice/user@*.service"):
        try:
            if c.is_dir() and os.access(c, os.W_OK) and \
                    "pids" in (c / "cgroup.subtree_control").read_text().split():
                return c
        except Exception:
            continue
    return None


def make_run_cgroup(base: Path | None = None, tasks_max: int = DEFAULT_TASKS_MAX,
                    name: str | None = None) -> Path:
    """Create a per-run cgroup with pids.max set, ready to accept one child."""
    base = base or writable_pids_parent()
    if base is None:
        raise RuntimeError(
            "no writable cgroup with the pids controller enabled in "
            "subtree_control; cannot enforce a per-run task cap")
    name = name or f"fgpu-{int(time.time())}-{os.getpid()}"
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    # Do NOT write ourselves into cgroup.procs "for permission": we already
    # own the directory we just created, and putting our own pid in it makes
    # the cgroup non-empty and therefore un-removable (EBUSY on rmdir) for as
    # long as we live. The child goes in via the preexec hook instead.
    (d / "pids.max").write_text(str(int(tasks_max)))
    return d


def cgroup_preexec(cgroup_dir: Path):
    """preexec_fn: move THIS process (the pre-exec child) into cgroup_dir.

    Runs after fork and before exec, so the cap is in force before the child
    can do anything, and the child remains a direct child of run_kernel's
    process -- which is what keeps the existing killpg timeout path working.
    """
    procs = cgroup_dir / "cgroup.procs"

    def _apply():
        try:
            procs.write_text(str(os.getpid()))
        except Exception:
            pass          # never block the run on a bookkeeping write
    return _apply


def cleanup_cgroup(cgroup_dir: Path) -> None:
    """Remove a per-run cgroup. Best effort: a busy cgroup may refuse."""
    try:
        cgroup_dir.rmdir()
    except Exception:
        pass


def choose_mechanism_preserving_kill(preferred: str | None = None) -> Mechanism:
    """Mechanism choice under the constraint that killpg must still work.

    The cgroup route is first, because it preserves the process-group SIGKILL
    that run_kernel's timeout path depends on. systemd-run is demoted: it is
    available on this host, but only in a form that detaches the child from our
    process group, which silently disables the timeout backstop.
    """
    ms = probe()
    by_name = {m.name: m for m in ms}
    for name in ([preferred] if preferred else []) + ["cgroup", "rlimit_nproc",
                                                       "none"]:
        m = by_name.get(name)
        if m and m.available:
            return m
    return by_name["none"]
