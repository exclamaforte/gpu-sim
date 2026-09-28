#!/usr/bin/env python3
"""Regenerate the rlimit table in harness.md from LIMIT_TABLE.

The table is emitted by the writer rather than transcribed, because a
hand-kept copy of a safety table is the same defect as a hand-kept copy of a
limit list: it describes a different file than the one that runs.
Criterion 4 of gpu-sim-u2o.6.3.1.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from epic_u2o.rlimit_cuda_gate import LIMIT_TABLE, limits_harness_applies  # noqa: E402
from epic_u2o.safe_harness import ResourceLimits  # noqa: E402

HEADER = """# harness.md — resource limits on the shared RTX 5090

Every limit the harness applies, why it is safe for CUDA, and the ts job that
proves it. **Generated** by `experiments/gpu-sim-u2o.6.3.1/render_harness_md.py`
from `epic_u2o/rlimit_cuda_gate.LIMIT_TABLE`. Do not hand-edit the table.

## Why this file exists

Two real-GPU runs were killed by limits that had never been checked against
CUDA, and both times the symptom pointed at the wrong subsystem:

| what | what you saw | what it was |
|---|---|---|
| `RLIMIT_NPROC=32`/`512` | `cudaErrorOsOperationNotSupported` at the first `cudaMalloc` | the kernel compares NPROC against the real uid's **total** task count |
| `RLIMIT_AS=24 GiB` | every run died in ~2 ms | the 5090's BAR1 mapping is 32 GiB |

A misleading CUDA error is worse than a resource error, because it sends you
looking at the driver instead of at the limit you just set. So the limit set is
a **gate** (`epic_u2o/rlimit_cuda_gate.py`), not a comment.

## The table

"""


def main() -> int:
    applied = limits_harness_applies(ResourceLimits())
    rows = ["| limit | value | applied | why CUDA-safe | evidence (ts job) |",
            "|---|---|---|---|---|"]
    for e in LIMIT_TABLE:
        name = e["limit"]
        is_applied = e.get("applied", True)
        if is_applied:
            val = applied.get(name)
            shown = (f"{val // (1024*1024)} MiB"
                     if name in ("RLIMIT_AS", "RLIMIT_DATA") else str(val))
        else:
            shown = "—"
        rows.append(
            f"| `{name}` | {shown} | {'yes' if is_applied else '**no**'} | "
            f"{e['why_cuda_safe']} | `{e['evidence']}` |")

    tail = """

## Two invariants, executable rather than documented

1. **`RLIMIT_NPROC` is never set.** `ResourceLimits.processes_limit` defaults to
   0 and a non-zero value now **raises** instead of being applied. It used to be
   `if self.processes_limit: setrlimit(RLIMIT_NPROC, ...)`, and a guard around a
   live call is a footgun with a comment on it: the field exists, the name
   looks like the containment knob it is not, and setting it kills CUDA at the
   first `cudaMalloc`. This uid has ~2934 tasks against 256 processes, and a
   kernel-side probe bracketed the live total between 2048 (aborts) and 3072
   (works) — so **2048 looks generous and is already fatal**, and because the
   count drifts with the other tenants on this box, every fixed value is a race.
   Containment goes through cgroup v2 `pids.max` (gpu-sim-u2o.6.1.2).
2. **`RLIMIT_AS` is independent of `vram_limit_mb`.** Asserted on the numbers,
   not by a comment saying so, because a comment is exactly what was there when
   the 24 GiB failure happened. `RLIMIT_AS` must also exceed 32768 MiB (BAR1).

Honest limitation, unchanged: on ReBAR GPUs `RLIMIT_AS` **cannot** serve as a
VRAM cap — the BAR mapping dwarfs any useful value — so it is a leak backstop
only, not a VRAM accounting mechanism.

## Recorded discrepancy

`RLIMIT_FSIZE` is in the table as **not applied**. gpu-sim-u2o.6.3.1's
description lists `FSIZE(min(256, quota))` as part of the set the harness
applies today, but `apply()` never calls `setrlimit(RLIMIT_FSIZE)`. Recorded
rather than written in as though it were enforced, because a safety table that
describes a different file than the one that runs is worse than no table.
`tests/test_rlimit_cuda_gate.py` fails if `RLIMIT_FSIZE` ever appears in
`safe_harness.py`, so the entry cannot go stale.

## The probe

`epic_u2o/rlimit_cuda_gate.PROBE_SOURCE` builds a probe that runs under the
harness's **exact** limit set: context creation (the BAR1 mapping),
`cudaMalloc`, one launch, `cudaDeviceSynchronize`, and a read-back. It records
the live uid task count, because `RLIMIT_NPROC` is per-uid and that number is
what makes a future reintroduction detectable rather than mysterious. Cheap
(<10 s) so it can run on every batch job; see
`experiments/gpu-sim-u2o.6.3.1/metrics.json` for a real 5090 run.

On failure, `diagnose_probe_failure()` names the offending limit by stage. When
no limit in the set explains a failure it says exactly that — "cannot diagnose"
must never degrade into "add a limit to fix it".
"""
    out = HEADER + "\n".join(rows) + "\n" + tail
    (REPO / "harness.md").write_text(out)
    print(f"harness.md written: {len(LIMIT_TABLE)} limits, "
          f"{len(applied)} actually applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
