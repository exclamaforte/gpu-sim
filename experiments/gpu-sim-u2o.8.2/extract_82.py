#!/usr/bin/env python3
"""gpu-sim-u2o.8.2 EXTRACT (CPU, unqueued): PTX+SASS+ptxas over the corpus.

For every measured corpus row (3261): compile its source to PTX and SASS
(nvcc CPU-side, no GPU), tokenize both streams, parse ptxas -v resource
usage. Writes one record per row to ptx_corpus.jsonl (shard files merged
by merge_82.py). Full PTX/SASS texts are NOT committed (100MB+); the
script regenerates them deterministically from manifests, and the artifact
carries digests + counts + numeric features.

Shardable: --shard lo:hi over the spec list; --workers N (default 2).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.ptx_tokenizer import (compile_to_ptx, compile_to_sass,  # noqa: E402
                                    get_ptxas_info, tokenize_ptx,
                                    tokenize_sass, extract_ptx_features)

ARCH = "sm_120"
CORPUS = ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
          "experiments/gpu-sim-e5q/measured.jsonl",
          "experiments/gpu-sim-u2o.11.3/measured.jsonl",
          "experiments/gpu-sim-f6y/measured_new.jsonl",
          "experiments/gpu-sim-0vk/measured_new.jsonl")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_module(name, path):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(path)
    exec(compile(Path(path).read_text(), str(path), "exec"), mod.__dict__)
    return mod


def corpus_specs():
    """{kernel_hash: source} for all 3261 measured rows. 11.3/f6y/0vk from
    committed manifests; 7.1/e5q by deterministic regen (hash-verified)."""
    specs = {}
    for mp in ("experiments/gpu-sim-u2o.11.3/manifest.json",
               "experiments/gpu-sim-f6y/manifest.json",
               "experiments/gpu-sim-0vk/manifest.json"):
        man = json.loads((REPO / mp).read_text())
        for d in man["specs"]:
            specs[d["kernel_hash"]] = (d["source"], d["op_type"])
    tmp = tempfile.mkdtemp(prefix="82-regen-")
    for name, script in (
            ("regen71", "experiments/gpu-sim-u2o.7.1/build_batch.py"),
            ("regene5q", "experiments/gpu-sim-e5q/build_batch.py")):
        mod = load_module(name, REPO / script)
        scratch = Path(tmp) / name
        scratch.mkdir(parents=True)
        mod.HERE = scratch
        assert mod.main() == 0
        man = json.loads((scratch / "build" / "manifest.json").read_text())
        for d in man["specs"]:
            specs[d["kernel_hash"]] = (d["source"], d["op_type"])
    return specs


def measured_hashes():
    out = {}
    for f in CORPUS:
        for line in (REPO / f).read_text().splitlines():
            r = json.loads(line)
            out[r["kernel_hash"]] = r["op_type"]
    return out


def one(job):
    h, source, op = job
    t0 = time.time()
    rec = {"kernel_hash": h, "op": op, "arch": ARCH}
    try:
        ptx, _ = compile_to_ptx(source, arch=ARCH)
        rec["ptx_ok"] = bool(ptx)
        if ptx:
            toks = tokenize_ptx(ptx)
            rec["ptx_tokens"] = len(toks)
            rec["ptx_digest"] = digest(ptx)
            try:
                rec["ptx_features"] = extract_ptx_features(
                    ptx, get_ptxas_info(source, arch=ARCH))
            except Exception as e:  # noqa: BLE001 -- per-row record, not fatal
                rec["ptx_features_error"] = str(e)[:120]
    except Exception as e:  # noqa: BLE001
        rec["ptx_ok"] = False
        rec["ptx_error"] = str(e)[:200]
    try:
        sass, _ = compile_to_sass(source, arch=ARCH)
        rec["sass_ok"] = bool(sass)
        if sass:
            stoks = tokenize_sass(sass)
            rec["sass_tokens"] = len(stoks)
            rec["sass_digest"] = digest(sass)
    except Exception as e:  # noqa: BLE001
        rec["sass_ok"] = False
        rec["sass_error"] = str(e)[:200]
    try:
        info = get_ptxas_info(source, arch=ARCH)
        rec["resources"] = {k: getattr(info, k)
                            for k in ("registers_per_thread",
                                      "shared_mem_per_block",
                                      "local_mem_per_thread",
                                      "spilled_registers",
                                      "spilled_local_mem")}
    except Exception as e:  # noqa: BLE001
        rec["resources_error"] = str(e)[:120]
    rec["compile_ms"] = round((time.time() - t0) * 1000, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    os.environ["PATH"] = ("/home/gabe/.local/cuda-sysroot/opt/cuda/bin:"
                          + os.environ.get("PATH", ""))
    specs = corpus_specs()
    hashes = measured_hashes()
    missing = [h for h in hashes if h not in specs]
    print(f"specs: {len(specs)}, measured: {len(hashes)}, "
          f"unresolvable: {len(missing)}", flush=True)
    assert not missing, missing[:5]
    jobs = [(h, specs[h][0], hashes[h]) for h in sorted(hashes)]
    if args.shard:
        lo, hi = (int(x) for x in args.shard.split(":"))
        jobs = jobs[lo:hi]
    tag = f"_{args.shard.replace(':', '-')}" if args.shard else ""
    t0 = time.time()
    with Pool(args.workers) as pool:
        recs = pool.map(one, jobs, chunksize=4)
    out = HERE / f"ptx_shard{tag}.jsonl"
    with out.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    wall = time.time() - t0
    ok = sum(1 for r in recs if r.get("ptx_ok"))
    print(f"shard{tag}: {len(recs)} rows, ptx_ok={ok}, wall={wall:.0f}s",
          flush=True)
    (HERE / f"extract{tag}.json").write_text(json.dumps(
        {"n": len(recs), "ptx_ok": ok,
         "sass_ok": sum(1 for r in recs if r.get("sass_ok")),
         "wall_sec": round(wall, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
