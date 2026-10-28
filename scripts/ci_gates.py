#!/usr/bin/env python3
"""scripts/ci_gates.py -- the gate the lab harness actually looks for.

Runs gate.sh (--fast by default, full if CI_GATES_FULL=1) with a 290 s internal
timeout so that a gate that exceeds the harness's 300 s cap reports an honest
'GATES NOT EVALUATED' instead of being killed silently. Streams output so the
harness can distinguish a running gate from a hung one.

Exit codes: 0 = gate passed, 1 = gate failed, 2 = NOT EVALUATED (timed out).
2 is deliberately non-zero: 'we did not check' must never be reported as
'green' by a caller that only looks at the return code.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 290


def _pump(stream):
    """Forward the child's output as it arrives."""
    for line in stream:
        sys.stdout.write(line)
        sys.stdout.flush()


def main() -> int:
    args = ["bash", str(ROOT / "scripts" / "gate.sh")]
    if os.environ.get("CI_GATES_FULL") != "1":
        args.append("--fast")
    proc = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    # The timeout must be enforced on a thread that does NOT read the child's
    # stdout. Iterating over proc.stdout blocks until the child exits, so a
    # proc.wait(timeout=...) placed after that loop can never fire -- measured:
    # with TIMEOUT_S=1 a broken variant ran 5 minutes and printed nothing.
    pump = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
    pump.start()
    try:
        proc.wait(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pump.join(timeout=5)
        print(f"\nGATES NOT EVALUATED: gate exceeded {TIMEOUT_S}s cap "
              f"(full gate ~10-13 min; use --fast or raise the cap)")
        return 2
    pump.join(timeout=5)
    if proc.returncode != 0:
        print("GATE FAIL: one or more hard checks failed (see FAIL lines above)")
        return 1
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())