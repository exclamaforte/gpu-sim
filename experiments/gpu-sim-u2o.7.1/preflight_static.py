#!/usr/bin/env python3
"""Submission gate for the 7.1 measured batch. CPU-only.

A batch job that dies at spec 200 of 264 has burned a queue slot and
produced a partial dataset that LOOKS like data. So the gate performs the
job's whole pre-GPU path: build the batch, synthesize a sample of drivers,
compile them, and tokenize one -- all of which are CPU work.
"""
import ast
import builtins
import json
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
fails = []

for script in ("build_batch.py", "measure_batch.py"):
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
    missing = sorted(used - defined - set(dir(builtins)) - {"__file__", "__name__"})
    if missing:
        fails.append(f"{script}: undefined name(s) {missing}")
    else:
        print(f"OK  names resolve: {script}")

# build the batch for real (CPU) and check the manifest invariants
try:
    r = subprocess.run([sys.executable, str(HERE / "build_batch.py")],
                       capture_output=True, text=True, timeout=1800, cwd=str(HERE))
    print(r.stdout.strip()[-200:])
    if r.returncode != 0:
        fails.append("build_batch.py failed: " + r.stderr[-400:])
    else:
        m = json.loads((HERE / "build" / "manifest.json").read_text())
        need = {"kernel_hash", "kernel_name", "grid", "block", "problem_size",
                "covers", "regime_hint", "deliberately_poor", "source_path"}
        bad = [s["kernel_hash"] for s in m["specs"] if not need <= set(s)]
        if bad:
            fails.append(f"{len(bad)} manifest rows missing required keys")
        else:
            print(f"OK  manifest: {m['n_specs']} rows, all with required keys")
        from epic_u2o.kernel_names import declared_kernel_names
        mism = [s["kernel_hash"] for s in m["specs"]
                 if s["kernel_name"] not in declared_kernel_names(s["source"])]
        if mism:
            fails.append(f"kernel_name/source mismatch (gpu-sim-9ti) in {mism[:3]}")
        else:
            print("OK  every kernel_name appears in its own source (9ti guard)")
except Exception as e:
    fails.append(f"build stage: {type(e).__name__}: {e}")

# synthesize + compile a SAMPLE, and tokenize one: the job's pre-GPU path
try:
    from epic_u2o.kernel_driver import synthesize
    from epic_u2o.safe_harness import SafeHarness
    from epic_u2o.tokenizer import tokenize_for_model
    m = json.loads((HERE / "build" / "manifest.json").read_text())
    sample = m["specs"][:3] + m["specs"][-3:]
    h = SafeHarness(timeout_sec=120)
    with tempfile.TemporaryDirectory() as td:
        for d in sample:
            spec = {k: d[k] for k in ("source", "kernel_name", "grid", "block",
                                       "args", "problem_size", "shared_mem_bytes")}
            spec["op_type"] = d.get("op_type", "anchor")
            ok, msg = h.compile_kernel(synthesize(spec),
                                       str(Path(td) / "s.out"), arch="sm_120")
            if not ok:
                fails.append(f"sample {d['kernel_hash']} does not compile: {str(msg)[:200]}")
        if not fails:
            print(f"OK  {len(sample)} sampled drivers synthesize + compile")
        n = len(tokenize_for_model(sample[0]["source"], max_length=2048))
        print(f"OK  tokenizer produces {n} model ids for a sample source")
except Exception as e:
    fails.append(f"sample stage: {type(e).__name__}: {e}")

# phase B must not compile: the queue kills a job that leaves the GPU idle,
# and 264 nvcc runs inside the job is what caused that.
try:
    build = HERE / "build"
    m = json.loads((build / "manifest.json").read_text())
    missing = [s["kernel_hash"] for s in m["specs"]
               if not (build / f"k_{s['kernel_hash']}.out").exists()]
    if missing:
        fails.append(f"{len(missing)} drivers are not prebuilt; run compile_all.py "
                     f"first (e.g. {missing[:2]}) -- phase B must not compile")
    else:
        print(f"OK  all {len(m['specs'])} drivers prebuilt; phase B will not compile")
except Exception as e:
    fails.append(f"prebuilt check: {type(e).__name__}: {e}")

print()
if fails:
    print("DO NOT SUBMIT:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("safe to submit the batch")
