#!/usr/bin/env python3
"""Effective CPU capacity helper (gpu-sim-ynaj).

A cgroup v2 CFS quota is a BANDWIDTH limit, not a CPU-set restriction, so
sched_getaffinity/os.cpu_count()/getconf are blind to it. nproc (coreutils >=9.5)
reads cpu.max and is the only correct view -- but it is version-dependent.
This helper walks the cgroup path and takes the most restrictive non-'max'
quota, cross-checks against nproc, and reports disagreement loudly.
"""
import os
import subprocess
import sys


def parse_cpu_max(text):
    """Parse 'cpu.max' content -> (quota_us or None, period_us or None)."""
    try:
        q, per = text.split()
    except (ValueError, AttributeError):
        return None, None
    quota = None if q == "max" else int(q)
    try:
        period = int(per)
    except ValueError:
        return None, None
    return quota, period


def walk_cpu_max(cgroup_rel, fs_root="/sys/fs/cgroup"):
    """Walk from /sys/fs/cgroup/<cgroup_rel> up to fs_root, collecting
    (quota, period) for every level that has a cpu.max file."""
    out = []
    path = os.path.join(fs_root, cgroup_rel.lstrip("/"))
    # normalize
    path = os.path.normpath(path)
    root = os.path.normpath(fs_root)
    # walk up
    cur = path
    seen = set()
    while True:
        cm = os.path.join(cur, "cpu.max")
        if os.path.isfile(cm):
            try:
                quota, period = parse_cpu_max(open(cm).read())
                out.append((quota, period, cm))
            except OSError:
                pass
        if cur == root or cur == "/" or cur in seen:
            break
        seen.add(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
        if not cur.startswith(root):
            break
    return out


def _cgroup_rel():
    try:
        lines = open("/proc/self/cgroup").read().strip().splitlines()
    except OSError:
        return ""
    for line in lines:
        parts = line.split("::", 1)
        if len(parts) == 2:
            return parts[1]
    return ""


def effective_cpus():
    """Return dict: affinity_cpus, quota_cpus (or None), effective, reason,
    cgroup_path, cpu_max_path, nproc, nproc_agrees."""
    affinity = len(os.sched_getaffinity(0))
    rel = _cgroup_rel()
    quotas = walk_cpu_max(rel)
    # most restrictive non-max quota
    quota_cpus = None
    quota_path = None
    for quota, period, path in quotas:
        if quota is not None and period:
            val = quota / period
            if quota_cpus is None or val < quota_cpus:
                quota_cpus = val
                quota_path = path
    if quota_cpus is None:
        effective = affinity
        reason = "no non-max cpu.max quota found; falling back to affinity"
    else:
        effective = min(affinity, quota_cpus)
        reason = "min(affinity, cgroup quota)"
    # nproc cross-check
    try:
        nproc = int(subprocess.check_output(["nproc"], text=True).strip())
    except Exception:
        nproc = None
    agrees = (nproc == effective) if nproc is not None else None
    if nproc is not None and not agrees:
        reason = reason + f"; WARNING: nproc={nproc} disagrees with effective={effective}"
    return {
        "affinity_cpus": affinity,
        "quota_cpus": quota_cpus,
        "effective_cpus": effective,
        "reason": reason,
        "cgroup_path": rel,
        "cpu_max_path": quota_path,
        "nproc": nproc,
        "nproc_agrees": agrees,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(effective_cpus(), indent=2))
