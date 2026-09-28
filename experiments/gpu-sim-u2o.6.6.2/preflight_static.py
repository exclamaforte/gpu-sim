#!/usr/bin/env python3
"""Static pre-flight for the experiment scripts, run BEFORE submitting to ts.

I have now lost three queued GPU jobs to defects a two-second check would have
caught (an undefined name, a wrong key, a bad f-string). Every one of them
surfaced only when the job started, and a queued job costs the whole queue a
slot. This script compiles the scripts and then resolves every module-level
name they reference, so an undefined import or typo is caught on the CPU.

Usage: python3 experiments/gpu-sim-u2o.6.6.2/preflight_static.py
Exit 0 = safe to submit; exit 1 = do not submit.
"""
import ast
import builtins
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

TARGETS = ["measure_family.py", "merge_metrics.py", "anchor_kernels.py",
           "verify_budgets.py" if (HERE / "verify_budgets.py").exists() else "measure_family.py"]

BUILTINS = set(dir(builtins))
fails = []

for name in sorted(set(TARGETS)):
    path = HERE / name
    if not path.exists():
        continue
    src = path.read_text()
    try:
        compile(src, str(path), "exec")
    except SyntaxError as e:
        fails.append(f"{name}: syntax error line {e.lineno}: {e.msg}")
        continue

    tree = ast.parse(src)
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            defined.add(node.name)
            for a in getattr(node, "args", None).args if hasattr(node, "args") else []:
                defined.add(a.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        elif isinstance(node, ast.Global):
            defined.update(node.names)

    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    # names bound in comprehensions/lambda args are covered above; anything
    # left that is neither defined nor a builtin is a NameError waiting to fire
    missing = sorted(used - defined - BUILTINS - {"__name__", "__file__",
                                                   "__doc__", "self", "cls"})
    if missing:
        fails.append(f"{name}: undefined name(s) {missing}")
    else:
        print(f"OK  {name}")

# the imported modules must also actually import
for mod in ("epic_u2o.kernel_driver", "epic_u2o.safe_harness",
            "epic_u2o.queue_state", "anchor_kernels"):
    try:
        __import__(mod)
        print(f"OK  import {mod}")
    except Exception as e:
        fails.append(f"import {mod}: {type(e).__name__}: {e}")

print()
if fails:
    print("DO NOT SUBMIT:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("safe to submit")
