#!/usr/bin/env python3
"""gpu-sim-u2o.11.3 STAGE BUILD (CPU): size-stratified manifest over stages 2-5.

Selection is FOR REGIME COVERAGE, NOT ROW COUNT (the design decision that
matters): per (stage, op) take ALL specs at the two largest sizes, then fill
smaller sizes up to a per-op cap. Large-n memory-touching cells are where
memory-bound rows can come from; small-n cells are kept for launch-bound
coverage but capped so they cannot repeat 6.7.1's 73% launch-bound first pass.
Deterministic: enumeration order is the generator's, no sampling randomness.

Per spec the manifest carries the e5q row shape (covers, deliberately_poor,
kernel_hash via the canonical stage-1 definition, 9ti name guard) plus the
stage label. Synthesize runs over EVERY spec here as a preflight: stages 3
and 5 were never synthesize-tested (the PI's 41/41 covered stages 2 and 4
samples only), and an unsupported arg type must be a NAMED shortfall now,
not a hole in the batch later.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.generator_stage1 import kernel_hash_for  # noqa: E402 canonical (8.3/e5q)
from epic_u2o.kernel_names import declared_kernel_names  # noqa: E402
from epic_u2o.kernel_driver import synthesize  # noqa: E402

# Per-op caps AFTER taking all of the two largest sizes. Budget: s2 ~700,
# s3 ~900, s4 all (~96 after the u2o.11.3 extension), s5 all (560).
PER_OP_CAP = {"stage2": 90, "stage3": 70, "stage4": 10 ** 9, "stage5": 10 ** 9}
STAGES = ("stage2", "stage3", "stage4", "stage5")


def _spec_size_key(d):
    return d["problem_size"]


def _select(specs, cap):
    """All of the two largest sizes per op, then smaller sizes up to cap."""
    by_op = defaultdict(list)
    for s in specs:
        by_op[s["op_type"]].append(s)
    out = []
    for op, group in sorted(by_op.items()):
        sizes = sorted({_spec_size_key(s) for s in group}, reverse=True)
        big = set(sizes[:2])
        take_big = [s for s in group if _spec_size_key(s) in big]
        rest = sorted((s for s in group if _spec_size_key(s) not in big),
                      key=lambda s: (-_spec_size_key(s), s["kernel_hash"]))
        take = take_big + rest[:max(0, cap - len(take_big))]
        out.extend(take)
    return out


def main() -> int:
    import epic_u2o.generator_stage2 as s2
    import epic_u2o.generator_stage3 as s3
    import epic_u2o.generator_stage4 as s4
    import epic_u2o.generator_stage5 as s5
    mods = {"stage2": s2, "stage3": s3, "stage4": s4, "stage5": s5}
    out = HERE / "build"
    out.mkdir(exist_ok=True)

    manifest_specs, preflight_failures, per_stage = [], [], {}
    for stage in STAGES:
        specs = [s.to_dict() for s in
                 mods[stage].generate_kernels(max_kernels=100000)]
        # kernel_hash FIRST: selection ordering below must be stable and the
        # join later keys on it.
        for d in specs:
            d["kernel_hash"] = kernel_hash_for(
                d["source"], d["grid"], d["block"], d["problem_size"])
        chosen = _select(specs, PER_OP_CAP[stage])
        kept = []
        for d in chosen:
            names = declared_kernel_names(d["source"])
            if d["kernel_name"] not in names:
                preflight_failures.append(
                    {"stage": stage, "hash": d["kernel_hash"],
                     "phase": "9ti-name", "kernel_name": d["kernel_name"]})
                continue
            threads = (d["grid"][0] * d["grid"][1] * d["grid"][2]
                       * d["block"][0] * d["block"][1] * d["block"][2])
            d["covers"] = min(threads, d["problem_size"])
            d["deliberately_poor"] = (d["grid"] == [1, 1, 1]
                                      or d["block"] == [1, 1, 1])
            d["stage"] = stage
            try:
                program = synthesize(d)
            except Exception as e:
                preflight_failures.append(
                    {"stage": stage, "hash": d["kernel_hash"],
                     "phase": "synthesize", "error": str(e)[:200]})
                continue
            cu = out / f"k_{d['kernel_hash']}.cu"
            cu.write_text(d["source"])
            d["source_path"] = cu.name
            d["driver_compiles"] = None  # filled by compile_all.py
            kept.append({k: d[k] for k in
                         ("kernel_hash", "kernel_name", "op_type", "stage",
                          "grid", "block", "shared_mem_bytes", "args",
                          "problem_size", "covers", "deliberately_poor",
                          "source_path", "metadata")})
            kept[-1]["source"] = d["source"]
        per_stage[stage] = {"enumerated": len(specs), "selected": len(chosen),
                            "kept": len(kept)}
        manifest_specs.extend(kept)
        print(f"{stage}: enumerated={len(specs)} selected={len(chosen)} "
              f"kept={len(kept)}", flush=True)

    (HERE / "build" / "manifest.json").write_text(json.dumps(
        {"bead": "gpu-sim-u2o.11.3", "stage": "build",
         "selection_rule": ("per (stage, op): all of the two largest sizes, "
                            "then smaller sizes up to the per-op cap; "
                            "generator order, no sampling"),
         "per_op_cap": PER_OP_CAP, "per_stage": per_stage,
         "n_specs": len(manifest_specs),
         "preflight_failures": preflight_failures,
         "protocol": "single_run_stages_2_5",
         "specs": manifest_specs}, indent=2))
    print(f"manifest: {len(manifest_specs)} specs, "
          f"{len(preflight_failures)} preflight failures (named above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
