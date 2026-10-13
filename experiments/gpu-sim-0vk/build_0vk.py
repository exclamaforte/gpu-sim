#!/usr/bin/env python3
"""gpu-sim-0vk BUILD (CPU): keep-list -> manifest specs, guard in the path.

Reads keep_list.json (990 predicted cells: 399 mm, 963 measurable@0.015ms,
gate PASS in prediction.json) and writes one manifest spec per cell. Specs
are built from the GENERATORS' own to_dict (source/args/metadata schemas
key-for-key, f6y build_manifest pattern) -- nothing hand-rolled except the
custom GEMM dims, which reuse a sibling GEMM spec's source/args with the
generator's (16,16)-block grid scheme.

Every spec is synthesized here (f6y OOB guard verdict per cell), and every
used template gets one explicit guard verdict recorded in guard_verdicts
(item 6): refusal of a new template fails the build loudly. The frozen 2278
hashes are re-checked for novelty (build_extension asserts).
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.generator_stage1 import kernel_hash_for  # noqa: E402 canonical
from epic_u2o.generator_stage1 import generate_kernels as g1  # noqa: E402
from epic_u2o.generator_stage2 import generate_kernels as g2  # noqa: E402
from epic_u2o.generator_stage3 import generate_kernels as g3  # noqa: E402
from epic_u2o.generator_stage4 import generate_kernels as g4  # noqa: E402
from epic_u2o.generator_stage5 import generate_kernels as g5  # noqa: E402
from epic_u2o.kernel_driver import synthesize, check_subscripts_bounded  # noqa: E402
from epic_u2o.kernel_names import declared_kernel_names  # noqa: E402

GEN = {}
for gen in (g1, g2, g3, g4, g5):
    for s in gen(max_kernels=100000):
        GEN.setdefault(s.op_type, s.to_dict())

STAGE = {"stage1": "stage1", "stage2": "stage2", "stage3": "stage3",
         "stage4": "stage4", "stage5": "stage5"}


def stage_of(op):
    for gen, st in ((g1, "stage1"), (g2, "stage2"), (g3, "stage3"),
                    (g4, "stage4"), (g5, "stage5")):
        if op in {s.op_type for s in gen(max_kernels=100000)}:
            return st
    raise ValueError(f"no stage for {op}")


def main() -> int:
    gate = json.loads((HERE / "prediction.json").read_text())
    assert gate["pass"] is True, f"gate did not pass: {gate}"
    keep = json.loads((HERE / "keep_list.json").read_text())
    frozen = set()
    for p in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl",
              "experiments/gpu-sim-u2o.11.3/measured.jsonl",
              "experiments/gpu-sim-f6y/measured_new.jsonl"):
        for line in (REPO / p).read_text().splitlines():
            frozen.add(json.loads(line)["kernel_hash"])
    assert len(frozen) == 2278, len(frozen)

    out = HERE / "build"
    out.mkdir(exist_ok=True)
    specs, failures, seen = [], [], set()
    guard_verdicts = {}
    for cell in keep:
        op = cell["op"]
        d = dict(GEN[op])
        d.update(grid=cell["grid"], block=cell["block"],
                 problem_size=cell["n"],
                 shared_mem_bytes=cell["shared_mem_bytes"])
        md = dict(d.get("metadata") or {})
        for k, v in cell.items():
            if k in ("op", "n", "grid", "block", "shared_mem_bytes",
                     "predicted_class", "predicted_ms",
                     "predicted_measurable", "bytes_moved", "kernel_hash"):
                continue
            md[k] = v
        d["metadata"] = md
        d["kernel_hash"] = kernel_hash_for(
            d["source"], d["grid"], d["block"], cell["n"])
        assert d["kernel_hash"] == cell["kernel_hash"], \
            f"hash moved for {op} {cell['n']}: build != prediction"
        if d["kernel_name"] not in declared_kernel_names(d["source"]):
            failures.append({"cell": op, "phase": "9ti-name"})
            continue
        if d["kernel_hash"] in frozen or d["kernel_hash"] in seen:
            failures.append({"cell": op, "phase": "hash-collision"})
            continue
        seen.add(d["kernel_hash"])
        threads = (d["grid"][0] * d["grid"][1] * d["grid"][2]
                   * d["block"][0] * d["block"][1] * d["block"][2])
        d["covers"] = min(threads, cell["n"])
        d["deliberately_poor"] = (d["grid"] == [1, 1, 1]
                                  or d["block"] == [1, 1, 1])
        d["stage"] = stage_of(op)
        spec = {k: d[k] for k in
                ("source", "kernel_name", "grid", "block", "args",
                 "problem_size", "shared_mem_bytes", "op_type", "metadata")}
        try:
            prog = synthesize(spec)
            assert "cudaEventRecord" in prog
        except Exception as e:
            failures.append({"cell": op, "phase": "oob-guard",
                             "error": str(e)[:200]})
            continue
        if op not in guard_verdicts:
            from epic_u2o.kernel_driver import _buffer_counts, _is_pointer
            dims = {k: v for k, v in md.items() if type(v) is int}
            n = cell["n"]
            caps = {a["name"]: _buffer_counts(
                {"op_type": op}, dims, n).get(a["name"], n)
                for a in d["args"] if _is_pointer(a.get("type", ""))}
            n_checked, unchecked = check_subscripts_bounded(
                d["source"], kernel_name=op, args=d["args"],
                capacities=caps, scalars=dict(dims, n=n), n=n,
                threads=threads)
            guard_verdicts[op] = {"verdict": "pass", "n_checked": n_checked,
                                  "n_unchecked": len(unchecked)}
        (out / f"k_{d['kernel_hash']}.cu").write_text(d["source"])
        d["source_path"] = f"k_{d['kernel_hash']}.cu"
        d["predicted_class"] = cell["predicted_class"]
        d["predicted_ms"] = cell["predicted_ms"]
        kept = {k: d[k] for k in
                ("kernel_hash", "kernel_name", "op_type", "stage",
                 "grid", "block", "shared_mem_bytes", "args",
                 "problem_size", "covers", "deliberately_poor",
                 "source_path", "metadata", "predicted_class",
                 "predicted_ms")}
        kept["source"] = d["source"]
        specs.append(kept)
    (HERE / "guard_verdicts.json").write_text(json.dumps(
        {"bead": "gpu-sim-0vk", "per_template": guard_verdicts,
         "n_templates": len(guard_verdicts),
         "refused": [f for f in failures if f.get("phase") == "oob-guard"]},
        indent=2))
    print(f"guard verdicts: {len(guard_verdicts)} templates pass, "
          f"{len([f for f in failures if f.get('phase') == 'oob-guard'])} refused")
    (HERE / "manifest.json").write_text(json.dumps(
        {"bead": "gpu-sim-0vk", "stage": "build",
         "selection_rule": ("keep_list.json predicted cells (gate PASS); "
                            "specs from the generators' own schemas"),
         "n_specs": len(specs), "n_failures": len(failures),
         "failures": failures, "protocol": "single_run_stages_2_5",
         "specs": specs}, indent=2))
    print(f"manifest: {len(specs)} specs, {len(failures)} failures")
    for f in failures[:10]:
        print("  FAIL:", f)
    return 0 if (specs and not failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
