#!/usr/bin/env python3
"""gpu-sim-u2o.9.1 DATA (CPU): join 456 stage-1 rows + in-spec guard sweep.

Per row: kernel_hash, op, triple, regime, y (log ms), sigma_log (6.6),
src_ids (8.1 tokenize_for_model, cap 512), numeric33 (8.3 vector).
Sources: 7.1/e5q build specs regenerated hash-verified (9yt method).

In-spec marker (acceptance 9): every training source run through f6y's
check_subscripts_bounded with driver capacities; any refusal ABORTS the
build (an OOB row must never silently enter training). The 4 known-OOB
frozen hashes (0vk oob_frozen.json, LOADED not recomputed) are asserted
absent from the join.
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
from epic_u2o.kernel_driver import (  # noqa: E402
    _buffer_counts, _is_pointer, check_subscripts_bounded)

SRC_CAP = 512


def load_module(name, path):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(path)
    exec(compile(Path(path).read_text(), str(path), "exec"), mod.__dict__)
    return mod


def main() -> int:
    oob = json.loads((REPO / "experiments/gpu-sim-0vk/oob_frozen.json").read_text())
    oob_hashes = {r["kernel_hash"] for r in oob["rows"]}
    assert len(oob_hashes) == 4, "OOB set changed -- investigate, do not train"
    feats = [json.loads(l) for l in
             (REPO / "experiments/gpu-sim-u2o.8.3/features.jsonl")
             .read_text().splitlines()]
    targs = {json.loads(l)["kernel_hash"]: json.loads(l) for l in
             (REPO / "experiments/gpu-sim-u2o.6.6/targets.jsonl")
             .read_text().splitlines()}
    assert len(feats) == 456 and len(targs) == 456
    tmp = tempfile.mkdtemp(prefix="91-src-")
    specs = {}
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
            specs[d["kernel_hash"]] = d
    rows, refused = [], []
    for f in feats:
        h = f["kernel_hash"]
        assert h not in oob_hashes, f"OOB row in training join: {h}"
        t = targs[h]
        d = specs[h]
        dims = dict(d.get("driver_scalars") or {})
        n = f["problem_size"]
        grid, block = d["grid"], d["block"]
        threads = (grid[0] * grid[1] * grid[2]
                   * block[0] * block[1] * block[2])
        caps = {a["name"]: _buffer_counts(
            {"op_type": d["op_type"]}, dims, n).get(a["name"], n)
            for a in d["args"] if _is_pointer(a.get("type", ""))}
        try:
            check_subscripts_bounded(
                d["source"], kernel_name=d["op_type"], args=d["args"],
                capacities=caps, scalars=dict(dims, n=n), n=n, threads=threads)
        except Exception as e:  # noqa: BLE001 -- OOB must abort, loudly
            refused.append({"kernel_hash": h, "guard": str(e)[:160]})
            continue
        rows.append({"hash": h, "op": f["op_type"],
                     "triple": t["roofline_triple"], "regime": t["measured_regime"],
                     "y": math.log(t["median_runtime_ms"] + 1e-6),
                     "ms": t["median_runtime_ms"], "sigma_log": t["sigma_log"],
                     "src_ids": tokenize_for_model(d["source"], SRC_CAP),
                     "num33": f["vector"]})
    assert not refused, f"OOB sources in training join: {refused}"
    assert len(rows) == 456, f"join shortfall: {len(rows)}/456"
    with (HERE / "train_data_91.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"91 rows: {len(rows)}/456, guard refusals: 0, OOB overlap: 0",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
