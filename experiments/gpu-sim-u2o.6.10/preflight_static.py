#!/usr/bin/env python3
"""Submission gate for the 6.10 single dose. CPU-only, and deliberately so.

This job deliberately hangs a GPU on a box three other projects share, so a
failed dose is expensive in a way a failed compile is not. Everything up to the
point of GPU execution is checked here: the module body runs, the two kernels
COMPILE, and the timeout constant is the documented 2-5 s.
"""
import ast
import builtins
import runpy
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
fails = []

path = HERE / "validate_once.py"
tree = ast.parse(path.read_text())
defined = set()
for n in ast.walk(tree):
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
            defined.add((a.asname or a.name).split(".")[0])
    elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        defined.add(n.name)
    elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
        defined.add(n.id)
    elif isinstance(n, ast.arg):
        defined.add(n.arg)
    elif isinstance(n, ast.ExceptHandler) and n.name:
        defined.add(n.name)
used = {n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
missing = sorted(used - defined - set(dir(builtins)) - {"__file__", "__name__"})
if missing:
    fails.append(f"undefined name(s) {missing}")
else:
    print("OK  names resolve")

try:
    runpy.run_path(str(path), run_name="preflight")
    print("OK  module body executes")
except SystemExit:
    print("OK  module body executes (exited)")
except Exception as e:
    fails.append(f"module body raised: {type(e).__name__}: {e}")

# the two kernels must compile BEFORE the dose is spent (nvcc is a compiler,
# not a device; a failed compile here is a CPU-second, not a wasted hang)
try:
    sys.path.insert(0, str(HERE))
    import validate_once as vo
    from epic_u2o.safe_harness import SafeHarness
    from epic_u2o.kernel_driver import spec_from_stage1, synthesize
    if not (2 <= vo.HANG_TIMEOUT_SEC <= 5):
        fails.append(f"hang timeout {vo.HANG_TIMEOUT_SEC}s outside the 2-5 s band")
    else:
        print(f"OK  hang timeout {vo.HANG_TIMEOUT_SEC}s is in the 2-5 s band")
    with tempfile.TemporaryDirectory() as td:
        h = SafeHarness(timeout_sec=60)
        ok1, _ = h.compile_kernel(
            synthesize(spec_from_stage1("saxpy", [32, 1, 1], [256, 1, 1], 8192)),
            str(Path(td) / "normal.out"), arch="sm_120")
        ok2, _ = h.compile_kernel(vo.HANG_KERNEL,
                                 str(Path(td) / "hang.out"), arch="sm_120")
        if not ok1:
            fails.append("normal kernel does not compile")
        else:
            print("OK  normal kernel compiles")
        if not ok2:
            fails.append("hang kernel does not compile")
        else:
            print("OK  hang kernel compiles")
except Exception as e:
    fails.append(f"compile check raised: {type(e).__name__}: {e}")

print()
if fails:
    print("DO NOT SPEND THE DOSE:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("safe to submit the single dose")
