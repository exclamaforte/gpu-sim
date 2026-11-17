"""Preflight for /dev/shm utilisation (gpu-sim-e5w2).

Reclaims gpusim-* segments whose owning pid is dead (by liveness, not age),
then fails loudly if utilisation is still above threshold.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys


def _pid_from_name(name: str) -> int | None:
    # only the documented corpus form: gpusim-corpus-<pid>-<N>-<uuid8>
    m = re.match(r"gpusim-corpus-(\d+)-", name)
    if not m:
        return None
    return int(m.group(1))


def reclaim(directory: str) -> list[str]:
    removed = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return removed
    for base in entries:
        if not base.startswith("gpusim-"):
            continue
        pid = _pid_from_name(base)
        if pid is None or pid <= 0:
            continue
        path = os.path.join(directory, base)
        # owner liveness: remove only if the creating pid is dead
        try:
            os.kill(pid, 0)
            alive = True
        except (ProcessLookupError, OSError):
            alive = False
        if not alive:
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                removed.append(base)
            except OSError:
                pass
    return removed


def preflight(directory: str, threshold: int) -> int:
    removed = reclaim(directory)
    try:
        st = os.statvfs(directory)
        total_kb = st.f_blocks * st.f_frsize // 1024
        used_kb = (st.f_blocks - st.f_bavail) * st.f_frsize // 1024
    except OSError:
        total_kb = 0
        used_kb = 0
    if total_kb > 0:
        pct = used_kb * 100 // total_kb
    else:
        pct = 0
    if pct >= threshold:
        print(f"GATE PREFLIGHT FAIL: /dev/shm at {pct}% ({used_kb}K/{total_kb}K) after reclamation.",
              file=sys.stderr)
        return 1
    print(f"GATE PREFLIGHT: /dev/shm at {pct}% ({used_kb}K/{total_kb}K)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/dev/shm")
    ap.add_argument("--threshold", type=int, default=None)
    args = ap.parse_args()
    thr = args.threshold
    if thr is None:
        thr = int(os.environ.get("SHM_THRESHOLD_PCT", "80"))
    return preflight(args.dir, thr)


if __name__ == "__main__":
    sys.exit(main())
