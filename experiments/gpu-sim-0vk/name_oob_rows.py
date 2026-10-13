#!/usr/bin/env python3
"""gpu-sim-0vk NAME-OOB (CPU): name the 4 known-OOB frozen rows, by proof.

Method: load the PRE-FIX stage-3 templates from git history (parent of the
f6y OOB-fix commit), regenerate every pre-fix spec, join to the frozen 2278
rows by kernel_hash (a hash match proves the row was built from the broken
template), and run the f6y OOB guard on the matched sources. Refused yet
shipped = OOB-but-completing. Asserts exactly 4 and writes oob_frozen.json.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.generator_stage1 import kernel_hash_for  # noqa: E402
from epic_u2o.kernel_driver import (  # noqa: E402
    _buffer_counts, _is_pointer, check_subscripts_bounded)

FIX_COMMIT = "3dd024b"  # f6y: fix OOB segmented/strided templates
FROZEN_FILES = ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
                "experiments/gpu-sim-e5q/measured.jsonl",
                "experiments/gpu-sim-u2o.11.3/measured.jsonl",
                "experiments/gpu-sim-f6y/measured_new.jsonl")


def load_prefixed():
    out = subprocess.run(
        ["git", "-C", str(REPO), "show",
         f"{FIX_COMMIT}^:epic_u2o/generator_stage3.py"],
        capture_output=True, text=True, check=True).stdout
    assert "if (i < n) out[i * stride]" in out, \
        "loaded source lacks the broken strided_write idiom -- wrong commit?"
    assert "in[segment * n + offset]" in out, \
        "loaded source lacks the broken segmented idiom -- wrong commit?"
    spec = importlib.util.spec_from_loader("prefix_stage3", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = "<git-history>/generator_stage3.py"
    exec(compile(out, "<git-history>/generator_stage3.py", "exec"),
         mod.__dict__)
    return mod


def main() -> int:
    old3 = load_prefixed()
    prefixed = {}
    for s in old3.generate_kernels(max_kernels=100000):
        d = s.to_dict()
        h = kernel_hash_for(d["source"], d["grid"], d["block"],
                            d["problem_size"])
        prefixed[h] = d
    print(f"pre-fix stage3 specs: {len(prefixed)}")

    frozen = []
    for f in FROZEN_FILES:
        for line in (REPO / f).read_text().splitlines():
            frozen.append(json.loads(line))
    assert len(frozen) == 2278, len(frozen)

    matched = [(r, prefixed[r["kernel_hash"]]) for r in frozen
               if r["kernel_hash"] in prefixed]
    print(f"frozen rows from pre-fix stage3 templates: {len(matched)}")

    refused = []
    for r, d in matched:
        dims = dict(r.get("driver_scalars") or {})
        n = r["problem_size"]
        threads = (d["grid"][0] * d["grid"][1] * d["grid"][2]
                   * d["block"][0] * d["block"][1] * d["block"][2])
        caps = {a["name"]: _buffer_counts(
            {"op_type": d["op_type"]}, dims, n).get(a["name"], n)
            for a in d["args"] if _is_pointer(a.get("type", ""))}
        try:
            check_subscripts_bounded(
                d["source"], kernel_name=d["op_type"], args=d["args"],
                capacities=caps, scalars=dict(dims, n=n), n=n, threads=threads)
        except Exception as e:
            refused.append({"kernel_hash": r["kernel_hash"],
                            "op_type": d["op_type"], "n": n,
                            "grid": d["grid"], "block": d["block"],
                            "dims": dims, "guard": str(e)[:200]})
    print(f"OOB-but-completing frozen rows: {len(refused)}")
    for o in refused:
        print(f"  {o['kernel_hash']} {o['op_type']} n={o['n']} "
              f"grid={o['grid']} block={o['block']} dims={o['dims']}")
    (HERE / "oob_frozen.json").write_text(
        json.dumps({"bead": "gpu-sim-0vk", "n": len(refused),
                    "rows": refused}, indent=2))
    assert len(refused) == 4, f"expected the 4 known-OOB rows, got {len(refused)}"
    print("OOB-NAMING PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
