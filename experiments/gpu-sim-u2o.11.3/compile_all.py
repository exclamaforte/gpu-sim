#!/usr/bin/env python3
"""u2o.11.3 PHASE A (CPU, unqueued): synthesize and compile every driver.

e5q's compile_all, carried over unchanged in shape: the queue killed the
combined job for 0.7% GPU utilisation, so nvcc runs here and the queued job
only measures. Records wall time for acceptance item 7.

Subsets (the full 2322-spec compile is ~50 CPU-minutes; subsets avoid
rebuilding current binaries):
--only-stage STAGE: that stage only (the s4 re-run: arity-sized buffers).
--only-missing: only specs with no binary yet.
--slice LO:HI: manifest index range only (the mm-regime extension lives at
  [2211:2322]; the base binaries were unlinked after successful measurement
  and must NOT be rebuilt -- their rows are already measured).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.kernel_driver import synthesize  # noqa: E402
from epic_u2o.safe_harness import SafeHarness  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-stage", default=None)
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--slice", default=None)
    args = ap.parse_args()
    build = HERE / "build"
    specs = json.loads((build / "manifest.json").read_text())["specs"]
    if args.slice:
        lo, hi = (int(x) for x in args.slice.split(":"))
        specs = specs[lo:hi]
    if args.only_stage:
        specs = [s for s in specs if s["stage"] == args.only_stage]
    if args.only_missing:
        specs = [s for s in specs
                 if not (build / f"k_{s['kernel_hash']}.out").exists()]
    h = SafeHarness(timeout_sec=300)
    ok = bad = 0
    errors = []
    t0 = time.time()
    for i, d in enumerate(specs):
        spec = {k: d[k] for k in ("source", "kernel_name", "grid", "block",
                                  "args", "problem_size", "shared_mem_bytes")}
        spec["op_type"] = d.get("op_type", "anchor")
        spec["metadata"] = d.get("metadata", {})
        try:
            program = synthesize(spec)
        except Exception as e:
            bad += 1
            errors.append({"hash": d["kernel_hash"], "stage": "synthesize",
                           "error": str(e)[:200]})
            continue
        c, m = h.compile_kernel(program, str(build / f"k_{d['kernel_hash']}.out"),
                                arch="sm_120")
        if c:
            ok += 1
        else:
            bad += 1
            errors.append({"hash": d["kernel_hash"], "stage": "compile",
                           "error": str(m)[:200]})
        if (i + 1) % 20 == 0:
            print(f"  compiled {i + 1}/{len(specs)} ({time.time() - t0:.0f}s)",
                  flush=True)
    wall = time.time() - t0
    tag = ""
    if args.only_stage:
        tag = f"_{args.only_stage}"
    if args.only_missing:
        tag = "_missing"
    if args.slice:
        tag = f"_slice_{args.slice.replace(':', '-')}"
    (build / f"compile_errors{tag}.json").write_text(json.dumps(errors, indent=2))
    print(f"compiled {ok}/{len(specs)} in {wall:.0f}s, {bad} failed")
    prev = json.loads((HERE / "build" / "compile_summary.json").read_text())
    if not tag:
        (build / "compile_summary.json").write_text(json.dumps(
            {"n_specs": len(specs), "n_ok": ok, "n_failed": bad,
             "wall_sec": wall}, indent=2))
    else:
        prev["recompiles"] = prev.get("recompiles", []) + [
            {"mode": tag, "n_specs": len(specs), "n_ok": ok, "n_failed": bad,
             "wall_sec": wall}]
        (build / "compile_summary.json").write_text(json.dumps(prev, indent=2))
    return 0 if (ok or not specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
