#!/usr/bin/env python3
"""btrfs METADATA headroom -- fail before rename() dies with a lying ENOSPC.

gpu-sim-cywp.  Observed: every atomic write (editor temp+rename, `git checkout`,
`git commit`, gate writes) failed

    ENOSPC: no space left on device, rename ...

while `df -h /home` reported 70 GiB free and `dd` into the same directory
succeeded.  /home is btrfs with metadata DUP and the device FULLY ALLOCATED:

    btrfs filesystem usage --raw /home
        Device allocated     998036733952
        Device unallocated         1048576     <- 1 MiB: no new chunk can be made
    btrfs filesystem df --raw /home
        Metadata, DUP: total=13421772800  used=10665132032

Appending to a file reuses existing extents, so it works; every path that needs a
new metadata transaction can fail once the metadata block group is full.  `df`
reports the DATA profile and cannot see this, which is why the error sends people
hunting the wrong thing.  /tmp is not an escape either: it is a quota-limited
tmpfs (gpu-sim-u6j5).

This check reads the two numbers that actually decide it, and fails loudly BEFORE
the box becomes unusable:

  metadata used/total  >= --pct (default 97%)           -> FAIL
  metadata used/total  >= --warm-pct (default 90%) AND
      device unallocated < --unalloc-gib (default 1)    -> FAIL (cannot grow it)

On a non-btrfs path (CI, another machine, a container) it prints one line and
exits 0: there is nothing to measure, and a skip is not a failure.

`--simulate` exists so the failure paths are testable without filling the box.
"""
import argparse
import os
import re
import subprocess
import sys

GiB = 1024 ** 3
MiB = 1024 ** 2


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def fstype(path):
    r = run(["stat", "-f", "-c", "%T", path])
    if r.returncode == 0:
        return (r.stdout or "").strip()
    return ""


def read_btrfs(path):
    """Return (meta_used, meta_total, unalloc) in bytes, or raise RuntimeError."""
    r = run(["btrfs", "filesystem", "df", "--raw", path])
    if r.returncode != 0:
        raise RuntimeError("btrfs filesystem df failed: " +
                           ((r.stderr or r.stdout or "").strip() or "no output"))
    used = total = None
    for line in (r.stdout or "").splitlines():
        m = re.match(r"\s*Metadata,\s*\S+:\s*total=(\d+),\s*used=(\d+)", line)
        if m:
            total, used = int(m.group(1)), int(m.group(2))
            break
    if used is None:
        raise RuntimeError("could not find a 'Metadata, <profile>: total=, used=' "
                           "line in `btrfs filesystem df --raw` output")
    u = run(["btrfs", "filesystem", "usage", "--raw", path])
    unalloc = None
    if u.returncode == 0:
        for line in (u.stdout or "").splitlines():
            m = re.match(r"\s*Device unallocated:\s*(\d+)", line)
            if m:
                unalloc = int(m.group(1))
                break
    # Non-root on an old btrfs-progs prints a warning and omits the per-device
    # block; a missing unallocated is "unknown", not zero.
    return used, total, unalloc


def human(n):
    if n is None:
        return "unknown"
    for unit, size in (("GiB", GiB), ("MiB", MiB)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n} B"


REMEDY = """\
Remedies, in the order that actually worked when this last fired (cheap first):
  1. python3 scripts/reap_appendonly_worktrees.py      # disposable test trees
  2. git worktree prune                                # stale registrations
  3. rm -rf .lab/tmp/implementer/lab-suite-*           # leaked pytest scratch
     (only dirs older than ~2 days with no live process -- u6j5 made .lab/tmp
     the sanctioned scratch dir, so this is the pile that grows)
  4. sudo btrfs balance start -musage=50 /home        # last resort, needs root
Appending to a file still works when this fires (it reuses existing extents);
rename() does not.  'df' is NOT the number to look at -- it reports the data
profile and said 70 GiB free during the outage."""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))),
        help="path on the filesystem to check (default: this repo)")
    ap.add_argument("--pct", type=float,
                    default=float(os.environ.get("FS_HEADROOM_PCT", "97")))
    ap.add_argument("--warm-pct", type=float,
                    default=float(os.environ.get("FS_HEADROOM_WARM_PCT", "90")))
    ap.add_argument("--unalloc-gib", type=float,
                    default=float(os.environ.get("FS_HEADROOM_UNALLOC_GIB", "1")))
    ap.add_argument("--simulate", default=None,
                    help="test hook: 'used,total,unalloc' in bytes ('-' for "
                         "unknown unallocated). Skips the btrfs calls.")
    args = ap.parse_args(argv)

    if args.simulate:
        parts = args.simulate.split(",")
        if len(parts) != 3:
            print("--simulate wants used,total,unalloc", file=sys.stderr)
            return 2
        used, total = int(parts[0]), int(parts[1])
        unalloc = None if parts[2] == "-" else int(parts[2])
        fs = "btrfs(simulated)"
    else:
        fs = fstype(args.path)
        if fs != "btrfs":
            print(f"not btrfs ({fs or 'unknown'} at {args.path}): metadata "
                  f"headroom is a btrfs property, nothing to check here")
            return 0
        try:
            used, total, unalloc = read_btrfs(args.path)
        except RuntimeError as e:
            # Unable to look is not the same as fine -- but it is also not proof
            # of a problem, and this stage must not fail on a missing tool.
            print(f"could not read btrfs metadata usage at {args.path}: {e}")
            return 0

    if total <= 0:
        print(f"btrfs reported total=0 for metadata at {args.path}; nothing to "
              f"measure")
        return 0
    pct = 100.0 * used / total
    short = f"metadata {human(used)}/{human(total)} = {pct:.1f}% " \
            f"(limit {args.pct:.0f}%), device unallocated {human(unalloc)}"
    floor = args.unalloc_gib * GiB
    full = pct >= args.pct
    warm = pct >= args.warm_pct and (unalloc is not None and unalloc < floor)

    if full or warm:
        why = ("The metadata block group is full" if full else
               "The metadata block group is filling and the device is fully "
               "allocated, so it cannot be grown")
        print(f"FS METADATA HEADROOM: FAIL -- {short}. {why}; the next "
              f"rename() can die with ENOSPC while df reports free space.")
        print(REMEDY)
        return 1

    print(f"fs metadata headroom OK: {short}. (df free space is the DATA "
          f"profile and is not authoritative on btrfs.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
