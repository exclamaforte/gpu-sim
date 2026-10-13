#!/usr/bin/env python3
"""gpu-sim-u2o.8.4 PTX-ID CACHE (CPU, unqueued): corpus PTX -> int sequences.

Deterministic hash mapping (PTXToken.type + text -> 1..255, 0 = pad), no
vocab fitting. One JSONL: {kernel_hash, ids}. Sources resolved exactly as
extract_82 (manifests + hash-verified regen). Shardable --shard lo:hi.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.ptx_tokenizer import compile_to_ptx, tokenize_ptx  # noqa: E402

ARCH = "sm_120"
MAXLEN = 512


def ptx_id(tok):
    h = hashlib.sha256((tok.type.name + "\x00" + tok.value).encode()
                       ).hexdigest()
    return (int(h[:8], 16) % 255) + 1


def load_module(name, path):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(path)
    exec(compile(Path(path).read_text(), str(path), "exec"), mod.__dict__)
    return mod


def corpus_sources():
    specs = {}
    for mp in ("experiments/gpu-sim-u2o.11.3/manifest.json",
               "experiments/gpu-sim-f6y/manifest.json",
               "experiments/gpu-sim-0vk/manifest.json"):
        for d in json.loads((REPO / mp).read_text())["specs"]:
            specs[d["kernel_hash"]] = d["source"]
    tmp = tempfile.mkdtemp(prefix="84-regen-")
    for name, script in (
            ("regen71", "experiments/gpu-sim-u2o.7.1/build_batch.py"),
            ("regene5q", "experiments/gpu-sim-e5q/build_batch.py")):
        mod = load_module(name, REPO / script)
        scratch = Path(tmp) / name
        scratch.mkdir(parents=True)
        mod.HERE = scratch
        assert mod.main() == 0
        for d in json.loads(
                (scratch / "build" / "manifest.json").read_text())["specs"]:
            specs[d["kernel_hash"]] = d["source"]
    return specs


def one(job):
    h, source = job
    try:
        ptx, _ = compile_to_ptx(source, arch=ARCH)
        if not ptx:
            return {"kernel_hash": h, "ok": False}
        ids = [ptx_id(t) for t in tokenize_ptx(ptx)][:MAXLEN]
        return {"kernel_hash": h, "ok": True, "ids": ids}
    except Exception as e:  # noqa: BLE001 -- per-row record
        return {"kernel_hash": h, "ok": False, "error": str(e)[:120]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    specs = corpus_sources()
    jobs = sorted(specs.items())
    if args.shard:
        lo, hi = (int(x) for x in args.shard.split(":"))
        jobs = jobs[lo:hi]
    tag = f"_{args.shard.replace(':', '-')}" if args.shard else ""
    with Pool(args.workers) as pool:
        recs = pool.map(one, jobs, chunksize=8)
    out = HERE / f"ptx_ids{tag}.jsonl"
    with out.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    ok = sum(1 for r in recs if r.get("ok"))
    print(f"ptx-ids{tag}: {len(recs)} rows, ok={ok}", flush=True)
    return 0 if ok == len(recs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
