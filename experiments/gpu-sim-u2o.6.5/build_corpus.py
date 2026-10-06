#!/usr/bin/env python3
"""gpu-sim-u2o.6.5 CPU half: build a real generated-kernel corpus and run every
entry through the full SafeHarness.preflight_check (bounds + scan + nvcc -ptx
+ -cudart=none smoke, i.e. the 6.5.1-fixed route). CPU-only.

Corpus: 8 good kernels from generator_stage1 (mixed ops/sizes) + 4
deliberately broken ones (bad launch, undefined symbol, syntax garbage,
oversize shared-mem). Output: preflight_results.json {passes:[...], rejects:
[{reason}]}. The GPU half (measure passes via ts) runs after the PI approves
the bead's acceptance; nothing here touches a device (hostile env +
tripwire-PATH inherited from test_isolation_claims where applicable).

Usage: python3 experiments/gpu-sim-u2o.6.5/build_corpus.py --out <path>
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.generator_stage1 import generate_kernels  # noqa: E402
from epic_u2o.safe_harness import SafeHarness  # noqa: E402


def spec_dict(s):
    d = s.to_dict()
    return {"source": d["source"], "kernel_name": d["kernel_name"],
            "grid": d["grid"], "block": d["block"], "args": d["args"],
            "problem_size": d["problem_size"],
            "shared_mem_bytes": d.get("shared_mem_bytes", 0),
            "op_type": d.get("op_type", "?"), "label": d.get("label", "?")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    h = SafeHarness(timeout_sec=30)
    goods = [spec_dict(s) for s in
             generate_kernels(ops=["saxpy", "relu", "copy", "mul"],
                              problem_sizes=[4096, 65536], max_kernels=8)]
    corpus = [dict(s, cid=f"good-{i}", expect="pass")
              for i, s in enumerate(goods)]
    base = dict(goods[0])
    broken = [
        dict(base, cid="bad-launch", expect="reject",
             block=[2048, 1, 1],
             why="block 2048 threads exceeds 1024"),
        dict(base, cid="bad-symbol", expect="reject",
             source=base["source"] + "\n__global__ void oops(float *x){ x[0] = no_such_fn(x[0]); }\n",
             why="undefined symbol must fail smoke compile"),
        dict(base, cid="bad-syntax", expect="reject",
             source="this is not CUDA {{{",
             why="syntax garbage must fail nvcc -ptx"),
        dict(base, cid="bad-shmem", expect="reject",
             shared_mem_bytes=512 * 1024,
             why="512KB exceeds 228KB/SM"),
    ]
    corpus += broken
    passes, rejects, surprises = [], [], []
    for entry in corpus:
        t0 = time.time()
        ok, msg = h.preflight_check(
            entry["source"], entry["grid"], entry["block"],
            entry.get("shared_mem_bytes", 0), spec=entry)
        dt = round(time.time() - t0, 1)
        rec = {"cid": entry["cid"], "op": entry.get("op_type"),
               "problem_size": entry["problem_size"], "ok": ok,
               "msg": msg[:300], "cpu_sec": dt}
        if ok:
            # GPU phase consumes EXACTLY these bits: full spec + grid/block.
            rec["spec"] = {k: entry[k] for k in (
                "source", "kernel_name", "grid", "block", "args",
                "problem_size", "shared_mem_bytes", "op_type")}
        (passes if ok else rejects).append(rec)
        if ok != (entry["expect"] == "pass"):
            surprises.append(entry["cid"])
        print(f"{'PASS' if ok else 'REJECT'} {entry['cid']} ({dt}s): {msg[:120]}",
              flush=True)
    out = {"bead": "gpu-sim-u2o.6.5", "phase": "cpu-preflight",
           "n_corpus": len(corpus), "n_pass": len(passes),
           "n_reject": len(rejects), "surprises": surprises,
           "passes": passes, "rejects": rejects,
           "verdict": "pass" if not surprises else "fail"}
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"pass={len(passes)} reject={len(rejects)} surprises={surprises} "
          f"verdict={out['verdict']} -> {args.out}", flush=True)
    return 1 if surprises else 0


if __name__ == "__main__":
    raise SystemExit(main())
