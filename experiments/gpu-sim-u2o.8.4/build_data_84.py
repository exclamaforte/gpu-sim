#!/usr/bin/env python3
"""gpu-sim-u2o.8.4 DATA (CPU): join rows + source ids into ablation_data.jsonl.

Per measured row: kernel_hash, op, log_target (log median ms), src_ids
(tokenize_for_model, cap 512), ptx_ids (from ptx_ids.jsonl, cap 512).
Sources from committed manifests + hash-verified regen (7.1/e5q).
"""
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.tokenizer import tokenize_for_model  # noqa: E402

SRC_CAP = 512
PTX_CAP = 512
CORPUS = ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
          "experiments/gpu-sim-e5q/measured.jsonl",
          "experiments/gpu-sim-u2o.11.3/measured.jsonl",
          "experiments/gpu-sim-f6y/measured_new.jsonl",
          "experiments/gpu-sim-0vk/measured_new.jsonl")


def load_module(name, path):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(path)
    exec(compile(Path(path).read_text(), str(path), "exec"), mod.__dict__)
    return mod


def main() -> int:
    sources = {}
    for mp in ("experiments/gpu-sim-u2o.11.3/manifest.json",
               "experiments/gpu-sim-f6y/manifest.json",
               "experiments/gpu-sim-0vk/manifest.json"):
        for d in json.loads((REPO / mp).read_text())["specs"]:
            sources[d["kernel_hash"]] = d["source"]
    tmp = tempfile.mkdtemp(prefix="84-src-")
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
            sources[d["kernel_hash"]] = d["source"]
    ptx = {}
    for line in (HERE / "ptx_ids.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("ok"):
            ptx[r["kernel_hash"]] = r["ids"][:PTX_CAP]
    rows, skipped = [], 0
    for f in CORPUS:
        for line in (REPO / f).read_text().splitlines():
            r = json.loads(line)
            h = r["kernel_hash"]
            if h not in sources or h not in ptx:
                skipped += 1
                continue
            rows.append({"kernel_hash": h, "op_type": r["op_type"],
                         "problem_size": r["problem_size"],
                         "block": r["launch_config"]["block"],
                         "grid": r["launch_config"]["grid"],
                         "metadata": {"family_id": r["op_type"]},
                         "log_target": math.log(r["median_runtime_ms"] + 1e-6),
                         "true_ms": r["median_runtime_ms"],
                         "src_ids": tokenize_for_model(sources[h], SRC_CAP),
                         "ptx_ids": ptx[h]})
    with (HERE / "ablation_data.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"ablation rows: {len(rows)}, skipped: {skipped}", flush=True)
    assert skipped == 0, f"{skipped} rows missing source/ptx"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
