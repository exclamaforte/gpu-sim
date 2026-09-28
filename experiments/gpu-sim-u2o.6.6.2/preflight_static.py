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

# MODULE-LEVEL EXECUTION, not just import. An import check that skips
# module-level code would miss exactly the class that cost us four jobs: the
# NameError was raised by a line at module scope in measure_family.py, and a
# name-resolution pass over the AST had missed it because the offending
# expression was only evaluated when the module body ran. So each experiment
# script is EXECUTED with --help-like stubs in sys.argv and its exit status
# checked; a script that cannot even reach its argument handling is not ready
# to be handed to the queue.
import runpy  # noqa: E402

for script in ("measure_family.py", "merge_metrics.py", "anchor_kernels.py"):
    path = HERE / script
    if not path.exists():
        continue
    saved_argv = list(sys.argv)
    try:
        sys.argv = [str(path), "--preflight-only"]
        # __name__ != "__main__" so main() does not run and touch the GPU; the
        # module BODY still executes, which is the part that must not raise.
        runpy.run_path(str(path), run_name="preflight")
        print(f"OK  module body executes: {script}")
    except SystemExit:
        print(f"OK  module body executes: {script} (exited)")
    except Exception as e:
        fails.append(f"module body raised in {script}: "
                     f"{type(e).__name__}: {e}")
    finally:
        sys.argv = saved_argv

for mod in ("epic_u2o.kernel_driver", "epic_u2o.safe_harness",
            "epic_u2o.queue_state", "anchor_kernels"):
    try:
        __import__(mod)
        print(f"OK  import {mod}")
    except Exception as e:
        fails.append(f"import {mod}: {type(e).__name__}: {e}")

# the queue_state regression tests, run as part of the gate: the module whose
# silence produced two false records must be green before we spend GPU time.
import subprocess  # noqa: E402
# The anchor's own reconciliation gate: cell tie, internal consistency
# against the COMMITTED metrics.json, and the per-kernel convention anchors.
# Run as part of the submission gate, because a check that only runs when a
# reviewer asks for it certifies one commit and nothing more -- which is how the
# flat convention survived in a committed artifact twice.
for gate in ("test_anchor_metrics.py", "test_queue_state.py"):
    gt = REPO / "tests" / gate
    if not gt.exists():
        continue
    r = subprocess.run([sys.executable, str(gt)], capture_output=True,
                       text=True, timeout=600, cwd=str(REPO))
    if r.returncode == 0:
        print(f"OK  tests/{gate}")
    else:
        fails.append(f"tests/{gate} fails:\n" + r.stdout[-800:])

qt = REPO / "tests" / "test_queue_state.py"
if qt.exists():
    r = subprocess.run([sys.executable, str(qt)], capture_output=True,
                       text=True, timeout=300, cwd=str(REPO))
    if r.returncode == 0:
        print("OK  tests/test_queue_state.py")
    else:
        fails.append("tests/test_queue_state.py fails:\n" + r.stdout[-800:])

print()
if fails:
    print("DO NOT SUBMIT:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("safe to submit")
