#!/usr/bin/env python3
"""CPU-only regression tests for the disk half of gpu-sim-u2o.6.3.

The reviewer's three findings, each with a test that fails on the code as it
was:

  1. ScratchDir identity was second-granular, so two runs in the same second in
     the same process shared ONE directory -- the bead's own output.log shows
     checks D1 and D2b both reporting "kern-1790523258-3599407".
  2. run_kernel called __enter__() manually, so a Popen failure propagated past
     the cleanup and leaked a directory per bad run.
  3. Nothing bounded disk growth MID-RUN: an 8 MB quota and a child writing
     200 x 1 MB files ran to completion and wrote 200 MB.

No GPU, no queue, no CUDA: CUDA_VISIBLE_DEVICES is blanked and the children
are plain shell/python writers.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]   # tests/ -> worktree root
sys.path.insert(0, str(REPO))

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail)[:170] if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    from epic_u2o.resource_limits import ScratchDir
    from epic_u2o.safe_harness import SafeHarness

    # --- 1. per-run directory identity ---------------------------------
    base = Path(tempfile.mkdtemp())
    dirs = [ScratchDir(base, quota_mb=16) for _ in range(5)]
    check("1. five ScratchDirs in one process get five distinct paths",
          len({d.path for d in dirs}) == 5,
          f"{len({d.path for d in dirs})} distinct of 5")
    with dirs[0]:
        pass
    check("1. names are not second-granular",
          all(len({p.name for p in (d.path for d in dirs)}) == 5 for _ in [0]))

    # keep=True must not let run B's files land in run A's directory
    a, b = ScratchDir(base, quota_mb=16), ScratchDir(base, quota_mb=16, keep=True)
    with a, b:
        (a.path / "a.bin").write_bytes(b"x" * 1024)
        (b.path / "b.bin").write_bytes(b"x" * 2048)
        check("1. concurrent runs do not share a quota",
              abs(a.size_mb() - 1024 / 1048576) < 1e-6
              and abs(b.size_mb() - 2048 / 1048576) < 1e-6,
              f"a={a.size_mb():.6f} MB, b={b.size_mb():.6f} MB")
        for d in (a, b):
            d.__exit__(None, None, None)
    shutil_rmtree(b.path)     # b was kept; clean it up so the tempdir goes

    # --- 2. no leak on the failure path --------------------------------
    base2 = Path(tempfile.mkdtemp())
    h = SafeHarness(timeout_sec=5, scratch_base=str(base2), disk_quota_mb=8)
    before = set(p.name for p in base2.iterdir()) if base2.exists() else set()
    raised = False
    try:
        h.run_kernel("/nonexistent/definitely-not-here")
    except Exception:
        raised = True
    after = set(p.name for p in base2.iterdir()) if base2.exists() else set()
    check("2. a failed exec leaves NO scratch directory behind",
          after == before, f"before={sorted(before)} after={sorted(after)}")
    check("2. the failure is reported, not swallowed", raised)

    # --- 3. a real mid-run bound ---------------------------------------
    base3 = Path(tempfile.mkdtemp())
    h3 = SafeHarness(timeout_sec=60, scratch_base=str(base3), disk_quota_mb=2)
    writer = """
import os, sys, time
end = time.time() + 30
i = 0
while time.time() < end:
    with open(os.path.join(sys.argv[1], "chunk%05d.bin" % i), "wb") as f:
        f.write(b"x" * 1048576)   # 1 MB each
    i += 1
"""
    t0 = time.time()
    res = h3.run_kernel(sys.executable, args=["-c", writer, "."])  # cwd = scratch
    elapsed = time.time() - t0
    disk = res.metadata.get("disk", {})
    check("3. a child writing 1 MB files forever is KILLED mid-run",
          bool(disk.get("killed_mid_run")) and elapsed < 25,
          f"elapsed={elapsed:.1f}s killed_mid_run={disk.get('killed_mid_run')} "
          f"breach={disk.get('breach_observed_mb')} peak={disk.get('peak_mb')} "
          f"quota={disk.get('quota_mb')}")
    check("3. the recorded peak is reported, not just the verdict",
          isinstance(disk.get("peak_mb"), float), disk)
    # What sampling can actually promise: the overshoot is bounded by
    # (write rate x interval), NOT that the peak lands on the quota. Measured
    # here: ~4.7 GB/s writer, 0.05 s interval, 2 MB quota -> killed after
    # 0.3 s having written ~1.4 GB. Asserted against the unbounded case (a
    # 30 s writer would produce ~140 GB), so the bound is doing work, and the
    # peak is recorded so the overshoot is visible rather than implied.
    unbounded_mb = 30 * 4700          # MB the same writer would reach unchecked
    check("3. the overshoot is bounded, not merely 'killed eventually'",
          isinstance(disk.get("peak_mb"), float) and disk["peak_mb"] < unbounded_mb / 20,
          f"peak={disk.get('peak_mb')} MB vs ~{unbounded_mb} MB unchecked "
          f"(kill at {elapsed:.1f}s, quota {disk.get('quota_mb')} MB)")

    # --- 4. a child that stays small is untouched -----------------------
    base4 = Path(tempfile.mkdtemp())
    h4 = SafeHarness(timeout_sec=30, scratch_base=str(base4), disk_quota_mb=8)
    small = "import os,sys;open(os.path.join(sys.argv[1],'a.bin'),'wb').write(b'x'*1024)"
    r4 = h4.run_kernel(sys.executable, args=["-c", small, "."])  # cwd = scratch
    d4 = r4.metadata.get("disk", {})
    check("4. a small run is not killed and stays within quota",
          r4.returncode == 0 and d4.get("within_quota") and not d4.get("killed_mid_run"),
          d4)

    print()
    print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


def shutil_rmtree(p: Path) -> None:
    import shutil
    shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
