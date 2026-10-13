#!/usr/bin/env python3
"""gpu-sim-9yt REGEN PROOF (CPU): both corpora regenerate with identical hashes.

The bead's acceptance: regenerate the 7.1 (264) and e5q (192) spec sets
through the CURRENT build scripts and assert every row hash still matches
the committed measured rows. .cu output is redirected to scratch (module
HERE is monkeypatched); nothing in the tree is written except proof.json
and metrics.json here.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def load_build(name, path):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(path)
    exec(compile(Path(path).read_text(), str(path), "exec"), mod.__dict__)
    return mod


def regen(mod_name, script_rel, tmp):
    mod = load_build(mod_name, REPO / script_rel)
    scratch = Path(tmp) / mod_name
    scratch.mkdir(parents=True)
    mod.HERE = scratch  # .cu + manifest land in scratch, never the tree
    assert mod.main() == 0
    man = json.loads((scratch / "build" / "manifest.json").read_text())
    return {d["kernel_hash"] for d in man["specs"]}


def measured(path_rel):
    return {json.loads(l)["kernel_hash"]
            for l in (REPO / path_rel).read_text().splitlines()}


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="9yt-regen-")
    results = {}
    for bead, script, rows in (
            ("7.1", "experiments/gpu-sim-u2o.7.1/build_batch.py",
             "experiments/gpu-sim-u2o.7.1/measured.jsonl"),
            ("e5q", "experiments/gpu-sim-e5q/build_batch.py",
             "experiments/gpu-sim-e5q/measured.jsonl")):
        gen = regen(f"build_{bead}", script, tmp)
        meas = measured(rows)
        results[bead] = {"n_regen": len(gen), "n_measured": len(meas),
                         "match": sorted(gen) == sorted(meas),
                         "missing": sorted(meas - gen)[:5],
                         "extra": sorted(gen - meas)[:5]}
        print(f"{bead}: regen {len(gen)}, measured {len(meas)}, "
              f"match={results[bead]['match']}", flush=True)
    ok = all(r["match"] for r in results.values())
    (HERE / "proof.json").write_text(json.dumps(
        {"bead": "gpu-sim-9yt", "results": results,
         "verdict": "PASS" if ok else "FAIL"}, indent=2))
    (HERE / "metrics.json").write_text(json.dumps(
        {"bead": "gpu-sim-9yt", "stage": "regen",
         "rows_71": [results["7.1"]["n_regen"], results["7.1"]["n_measured"]],
         "rows_e5q": [results["e5q"]["n_regen"], results["e5q"]["n_measured"]],
         "verdict": "PASS" if ok else "FAIL"}, indent=2))
    print("REGEN", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
