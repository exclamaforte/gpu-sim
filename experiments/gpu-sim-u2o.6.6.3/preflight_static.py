#!/usr/bin/env python3
"""Submission gate for gpu-sim-u2o.6.6.3. CPU-only.

Two jobs have now died at import time on this branch's predecessors. This
executes the module body (not just an AST pass over it) and checks every
imported name resolves, BEFORE a queue slot is spent.
"""
import ast
import builtins
import importlib.util
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
fails = []

for script in ("measure_contention.py",):
    path = HERE / script
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
    missing = sorted(used - defined - set(dir(builtins)) -
                     {"__file__", "__name__"})
    if missing:
        fails.append(f"{script}: undefined name(s) {missing}")
    else:
        print(f"OK  names resolve: {script}")

    # module-level execution: an import error is what killed the last job
    try:
        importlib.util.spec_from_file_location("probe_" + script[:-3], path)
        runpy.run_path(str(path), run_name="preflight")
        print(f"OK  module body executes: {script}")
    except SystemExit:
        print(f"OK  module body executes: {script} (exited)")
    except Exception as e:
        fails.append(f"module body raised in {script}: {type(e).__name__}: {e}")

# DRY COMPILE of every cell this job will measure. nvcc is a compiler, not a
# device: doing it here means a synthesis or kwarg failure costs a CPU second
# instead of a queue slot. Three jobs on this bead died on exactly those two
# classes of error, so the gate now performs the job's whole pre-GPU path.
try:
    sys.path.insert(0, str(HERE))
    import measure_contention as mc
    import tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory() as td:
        mc.HERE = _P(td)
        for name, fam, n, tag in mc.CELLS:
            try:
                exe, spec = mc.compile_cell(name, fam, n, tag)
                ok = exe.exists() and exe.stat().st_size > 0
                print(f"OK  dry compile {name:14s} {spec['kernel_name']:16s} "
                      f"n={spec['problem_size']}")
                if not ok:
                    fails.append(f"dry compile {name}: no executable produced")
            except Exception as e:
                fails.append(f"dry compile {name}: {type(e).__name__}: {e}")
except SystemExit:
    pass
except Exception as e:
    fails.append(f"dry compile stage: {type(e).__name__}: {e}")

# EXERCISE THE MEASUREMENT PATH TOO, on a binary that needs no device.
# Two jobs on this bead died in measure() on a SafeHarness kwarg while the
# gate only covered compile_cell -- so the gate now calls both.
try:
    meds, per_inv = mc.measure("/bin/true", invocations=1)
    print(f"OK  measure() runs (no trials from /bin/true, as expected: {meds})")
except TypeError as e:
    fails.append(f"measure() raised TypeError: {e}")
except Exception as e:
    print(f"note measure() raised {type(e).__name__}: {e}")

# the copied anchor kernels must still be self-consistent
try:
    import anchor_kernels
    anchor_kernels.self_check()
    print("OK  anchor_kernels.self_check()")
except Exception as e:
    fails.append(f"anchor_kernels: {type(e).__name__}: {e}")

print()
if fails:
    print("DO NOT SUBMIT:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("safe to submit")
