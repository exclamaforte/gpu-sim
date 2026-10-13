#!/usr/bin/env python3
"""gpu-sim-0vk CENSUS (CPU): pre-registered pass/fail on the real corpus.

Classes are census.py's ("memory->memory" etc. -- shared classify(), no
second scheme). Three verdicts:

(a) require_census() over the new-rows artifact (acceptance item 9: a
    total alone FAILS) -- structural, binding.
(b) The bead's pre-registered bar, decided BEFORE any dose: mm_new >= 250
    AND meas_new >= 750 (bdzat progress 2026-10-11) -- binding.
(c) The shared diversity gate over the FULL corpus (frozen 2278 + 983 new):
    the extension must not unbalance the training set -- binding. The
    same gate over the new-rows-only slice is reported as INFORMATIVE
    (extensions are legitimately regime-concentrated; the gate was
    calibrated on whole corpora, cf. 6.7.1's 73% precedent).
(d) Item-9 negative control in scratch: a 90%-launch_bound mix from the
    full pool must FAIL the shared gate (proves the gate fires).
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "gpu-sim-u2o.11.3"))

from census import classify, census, require_census  # noqa: E402
from epic_u2o.features import LaunchConfig, extract_features  # noqa: E402
from epic_u2o.generator_stage1 import generate_kernels as g1  # noqa: E402
from epic_u2o.generator_stage2 import generate_kernels as g2  # noqa: E402
from epic_u2o.generator_stage3 import generate_kernels as g3  # noqa: E402
from epic_u2o.generator_stage4 import generate_kernels as g4  # noqa: E402
from epic_u2o.generator_stage5 import generate_kernels as g5  # noqa: E402
from epic_u2o import corpus_protocol as cp  # noqa: E402

SOURCES = {}
for _gen in (g1, g2, g3, g4, g5):
    for _s in _gen(max_kernels=100000):
        SOURCES.setdefault(_s.op_type, _s.to_dict()["source"])

MM = "memory->memory"
LB = "latency->memory"

FROZEN_FILES = ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
                "experiments/gpu-sim-e5q/measured.jsonl",
                "experiments/gpu-sim-u2o.11.3/measured.jsonl",
                "experiments/gpu-sim-f6y/measured_new.jsonl")


def frozen_specs():
    """The frozen 2278 rows as classify-able specs (measured rows carry
    op_type + launch_config + problem_size, which is all cls_of needs)."""
    specs = []
    for p in FROZEN_FILES:
        for line in (REPO / p).read_text().splitlines():
            r = json.loads(line)
            lc = r["launch_config"]
            specs.append({"op_type": r["op_type"],
                          "grid": list(lc["grid"]),
                          "block": list(lc["block"]),
                          "shared_mem_bytes": lc.get("shared_mem_bytes", 0),
                          "problem_size": r["problem_size"],
                          "kernel_hash": r["kernel_hash"]})
    return specs


def features_for(spec):
    grid, block = list(spec["grid"]), list(spec["block"])
    while len(grid) < 3:
        grid.append(1)
    while len(block) < 3:
        block.append(1)
    lc = LaunchConfig(grid_x=grid[0], grid_y=grid[1], grid_z=grid[2],
                      block_x=block[0], block_y=block[1],
                      block_z=block[2],
                      shared_mem_bytes=spec.get("shared_mem_bytes", 0))
    return extract_features(SOURCES[spec["op_type"]], lc,
                            spec["problem_size"])


_CLS_CACHE = {}


def cls_of(spec):
    key = spec.get("kernel_hash")
    if key is not None and key in _CLS_CACHE:
        return _CLS_CACHE[key]
    c = classify(features_for(spec))
    if key is not None:
        _CLS_CACHE[key] = c
    return c


def main() -> int:
    manifest = {d["kernel_hash"]: d for d in
                json.loads((HERE / "manifest.json").read_text())["specs"]}
    new_rows = [json.loads(l)
                for l in (HERE / "measured_new.jsonl").read_text().splitlines()]
    print(f"new rows: {len(new_rows)}", flush=True)

    new_specs = [manifest[r["kernel_hash"]] for r in new_rows]
    counts = dict(Counter(cls_of(s) for s in new_specs))
    print("new-row census:", counts, flush=True)
    agree = sum(1 for r, s in zip(new_rows, new_specs)
                if (r.get("predicted_class") == MM) == (cls_of(s) == MM))
    print(f"forecast-vs-verdict mm agreement: {agree}/{len(new_rows)}",
          flush=True)
    mm_new = counts.get(MM, 0)
    meas_new = sum(1 for r in new_rows if r["measured_regime"] == "measurable")

    artifact = {"total_rows": len(new_rows),
                "regime_census": {"per_class": counts}}
    cen = require_census(artifact)
    print("require_census: structure OK", cen["per_class"] if isinstance(
        cen, dict) else cen, flush=True)
    bar_ok = mm_new >= 250 and meas_new >= 750
    print(f"bead bar: mm_new={mm_new} (>=250) meas_new={meas_new} (>=750) "
          f"-> {'PASS' if bar_ok else 'FAIL'}", flush=True)
    new_only = census(counts)
    print("new-only diversity gate (INFORMATIVE):", new_only["verdict"],
          new_only.get("reasons", ""), flush=True)

    frozen = frozen_specs()
    assert len(frozen) == 2278, f"frozen corpus moved: {len(frozen)}"
    frozen_hashes = {s["kernel_hash"] for s in frozen}
    new_hashes = {r["kernel_hash"] for r in new_rows}
    assert not (frozen_hashes & new_hashes), \
        "new rows collide with frozen hashes: not append-only"
    full_specs = list(frozen) + new_specs
    full_counts = dict(Counter(cls_of(s) for s in full_specs))
    print("full-corpus census:", full_counts, flush=True)
    full_gate = census(full_counts)
    print("FULL diversity gate (binding):", full_gate["verdict"],
          full_gate.get("reasons", ""), flush=True)

    scratch = (HERE / "scratch_neg"); scratch.mkdir(exist_ok=True)
    lb_pool = [s for s in full_specs if cls_of(s) == LB]
    mm_pool = [s for s in full_specs if cls_of(s) == MM]
    mix = (lb_pool * ((900 // max(1, len(lb_pool))) + 1))[:900] + mm_pool[:100]
    mix_counts = dict(Counter(cls_of(s) for s in mix))
    neg = census(mix_counts)
    print("NEG-CONTROL 90%-lb mix (must FAIL):", neg["verdict"],
          neg.get("reasons", ""), flush=True)
    (scratch / "neg_control.json").write_text(json.dumps(
        {"mix": len(mix), "counts": mix_counts,
         "verdict": neg["verdict"], "reasons": neg.get("reasons", [])},
        indent=2))
    assert neg["verdict"] == "FAIL", "negative control must FAIL the gate"

    neg_ok = (neg["verdict"] == "FAIL")
    metrics = {
        "bead": "gpu-sim-0vk", "stage": "census",
        "n_new": len(new_rows), "mm_new": mm_new, "meas_new": meas_new,
        "forecast_mm_agreement": [agree, len(new_rows)],
        "require_census": cen["per_class"] if isinstance(cen, dict) else cen,
        "bead_bar": "PASS" if bar_ok else "FAIL",
        "new_only_gate_informative": new_only["verdict"],
        "full_corpus_specs": len(full_specs),
        "full_corpus_census": full_counts,
        "full_gate": full_gate["verdict"],
        "full_gate_reasons": full_gate.get("reasons", []),
        "neg_control": {"mix": len(mix), "counts": mix_counts,
                          "gate_verdict": neg["verdict"],
                          "reasons": neg.get("reasons", []),
                          "control": "PASS" if neg_ok else "FAIL"},
    }
    (HERE / "census.json").write_text(json.dumps(metrics, indent=2))
    if full_gate["verdict"] != "PASS" or not bar_ok or not neg_ok:
        print("CENSUS FAIL")
        return 1
    print("CENSUS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
