#!/usr/bin/env python3
"""scripts/ci_gates.py -- the gate the lab harness actually looks for.

THE MISSING PIECE. The orchestrator has printed "gate status unknown: gate script
not found" on every drain because its gate check looks for exactly this path
(`scripts/ci_gates.py`) and runs `python3 <it>` with a 300 s timeout, treating
exit 0 as pass. We had `scripts/gate.sh` (and `make gate`) but not this file, so
the harness could never see a gate at all.

WHAT IT RUNS. `scripts/gate.sh --fast` by default: the docs-vs-model check, the
close audit, guard uniqueness, libfakegpu freshness -- seconds of CPU, no GPU.
That matters because the harness runs this on EVERY wake; the full pytest suite
(~2 min, and it drives the GPU through the interposer) would contend with real
experiment runs. Set CI_GATES_FULL=1 to include the suite.

Output: one PASS/FAIL line per check (the harness scans for "fail"/"error" to
report what broke), then a summary. Exit 0 iff every hard check passed.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = ["bash", str(ROOT / "scripts" / "gate.sh")]
    if os.environ.get("CI_GATES_FULL") != "1":
        args.append("--fast")
    proc = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    # Relay verbatim: the harness scans this text for a failing line.
    sys.stdout.write(out)
    if proc.returncode != 0:
        print("GATE FAIL: one or more hard checks failed (see FAIL lines above)")
        return 1
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
