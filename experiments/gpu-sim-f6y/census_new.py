#!/usr/bin/env python3
"""gpu-sim-f6y CENSUS (CPU): regime of the 18 fixed cells + the 1822 corpus.

Item 4: the 18 rows' regime classes. Item 8: the new total (1804 + 18) with
the regime census included. Classification is the IDENTICAL code path as
11.3 (experiments/gpu-sim-u2o.11.3/census.py: classify over 8.3 features);
the 11.3 files are read, never written. measured_regime (launch_bound /
measurable, a timing observation) is NOT the census class (a roofline
bottleneck pair) -- both are reported, conflated nowhere.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.11.3"))

from epic_u2o.features import extract_features, LaunchConfig  # noqa: E402
from census import classify, census  # noqa: E402 identical path (11.3)

U113 = REPO / "experiments" / "gpu-sim-u2o.11.3"


def spec_by_hash():
    specs = {}
    for m in json.loads((U113 / "manifest.json").read_text())["specs"]:
        specs.setdefault(m["kernel_hash"], m)
    for m in json.loads((HERE / "manifest.json").read_text())["specs"]:
        specs.setdefault(m["kernel_hash"], m)
    return specs


def features_for(row, spec):
    lc = LaunchConfig(
        grid_x=row["launch_config"]["grid"][0],
        grid_y=row["launch_config"]["grid"][1],
        grid_z=row["launch_config"]["grid"][2],
        block_x=row["launch_config"]["block"][0],
        block_y=row["launch_config"]["block"][1],
        block_z=row["launch_config"]["block"][2],
        shared_mem_bytes=row["launch_config"].get("shared_mem_bytes", 0))
    return extract_features(spec["source"], lc,
                            problem_size=row["problem_size"])


def main() -> int:
    specs = spec_by_hash()
    old_rows = [json.loads(l) for l in
                (U113 / "measured.jsonl").read_text().splitlines()]
    new_rows = [json.loads(l) for l in
                (HERE / "measured_new.jsonl").read_text().splitlines()]
    assert len(old_rows) == 1804, len(old_rows)
    assert len(new_rows) == 18, len(new_rows)

    new_classes = Counter()
    for r in new_rows:
        cls = classify(features_for(r, specs[r["kernel_hash"]]))
        r["census_class"] = cls
        new_classes[cls] += 1
    print("18 fixed cells by census class:",
          json.dumps(dict(sorted(new_classes.items()))))
    print("18 fixed cells by measured_regime:",
          json.dumps(dict(Counter(r["measured_regime"] for r in new_rows))))

    total = Counter()
    for r in old_rows + new_rows:
        spec = specs.get(r["kernel_hash"])
        if spec is None:
            print(f"CENSUS FAIL: row {r['kernel_hash']} joins no spec")
            return 1
        total[classify(features_for(r, spec))] += 1
    result = census(dict(sorted(total.items())))
    result["per_class"] = dict(sorted(total.items()))
    result["total_rows"] = sum(total.values())
    print("1822 corpus census:", json.dumps(result, indent=2))
    (HERE / "census.json").write_text(json.dumps(
        {"bead": "gpu-sim-f6y",
         "new_18_classes": dict(sorted(new_classes.items())),
         "new_18_measured_regime": dict(
             Counter(r["measured_regime"] for r in new_rows)),
         "corpus_total": sum(total.values()),
         "corpus_census": dict(sorted(total.items())),
         "verdict": result["verdict"],
         "reasons": result.get("reasons", [])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
