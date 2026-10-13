#!/usr/bin/env python3
"""gpu-sim-f6y BUILD (CPU): the 80 fixed-template occurrences.

The 11.3 shortfall ledger: 18 unique kernel hashes (segmented_read 13,
strided_write 5) with 0 successful invocations -- 55 + 25 = 80 manifest
occurrences whose binaries crashed by construction (in[segment*n+offset],
out[i*stride] with the `if (i<n)` guard). The templates are fixed in
epic_u2o/generator_stage3.py (segment*segment_size; `if (i*stride<n)`),
so the sources -- and therefore the hashes -- are new.

Selection rule (no sampling, no discretion): every 11.3 manifest occurrence
whose kernel_hash is in the 18 missing hashes, regenerated with the CURRENT
(fixed) generator source for its (op, n, grid, block, scalars). Same
enrichment as build_batch.py (covers, deliberately_poor, kernel_hash via the
canonical definition, 9ti name guard) plus the OOB-guard synthesize preflight:
a fixed cell that the guard refuses is a NAMED build failure, never queued.

Output: manifest.json (80 specs) + k_<hash>.cu kernel sources under build/.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.generator_stage1 import kernel_hash_for  # noqa: E402 canonical (8.3/e5q)
from epic_u2o.kernel_names import declared_kernel_names  # noqa: E402
from epic_u2o.kernel_driver import synthesize  # noqa: E402

U113 = REPO / "experiments" / "gpu-sim-u2o.11.3"


def main() -> int:
    import epic_u2o.generator_stage3 as s3
    old_manifest = json.loads((U113 / "manifest.json").read_text())["specs"]
    missing = set(json.loads(
        (U113 / "measure_summary.json").read_text())["missing_hashes"])
    assert len(missing) == 18, f"expected the 18 shortfall hashes, got {len(missing)}"
    targets = [m for m in old_manifest if m["kernel_hash"] in missing]
    print(f"shortfall occurrences: {len(targets)}")
    assert len(targets) == 80, f"expected 55+25=80, got {len(targets)}"

    gen = {s.op_type: s.to_dict()
           for s in s3.generate_kernels(max_kernels=100000)}
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    specs, failures = [], []
    for m in targets:
        op = m["op_type"]
        d = dict(gen[op])
        d.update(grid=m["grid"], block=m["block"],
                 problem_size=m["problem_size"])
        md = dict(d.get("metadata") or {})
        md.update(m.get("metadata") or {})
        d["metadata"] = md
        # New hash: the fixed source is part of the hash inputs.
        d["kernel_hash"] = kernel_hash_for(
            d["source"], d["grid"], d["block"], d["problem_size"])
        if d["kernel_name"] not in declared_kernel_names(d["source"]):
            failures.append({"hash": d["kernel_hash"], "phase": "9ti-name"})
            continue
        threads = (d["grid"][0] * d["grid"][1] * d["grid"][2]
                   * d["block"][0] * d["block"][1] * d["block"][2])
        d["covers"] = min(threads, d["problem_size"])
        d["deliberately_poor"] = (d["grid"] == [1, 1, 1]
                                  or d["block"] == [1, 1, 1])
        d["stage"] = "stage3"
        try:
            synthesize({k: d[k] for k in
                        ("source", "kernel_name", "grid", "block", "args",
                         "problem_size", "shared_mem_bytes", "op_type",
                         "metadata")})
        except Exception as e:
            failures.append({"hash": d["kernel_hash"], "phase": "oob-guard",
                             "error": str(e)[:200]})
            continue
        (out / f"k_{d['kernel_hash']}.cu").write_text(d["source"])
        d["source_path"] = f"k_{d['kernel_hash']}.cu"
        specs.append({k: d[k] for k in
                      ("kernel_hash", "kernel_name", "op_type", "stage",
                       "grid", "block", "shared_mem_bytes", "args",
                       "problem_size", "covers", "deliberately_poor",
                       "source_path", "metadata")})
        specs[-1]["source"] = d["source"]
        specs[-1]["old_hash"] = m["kernel_hash"]
    (HERE / "manifest.json").write_text(json.dumps(
        {"bead": "gpu-sim-f6y", "stage": "build",
         "selection_rule": ("every 11.3 manifest occurrence under the 18 "
                            "missing hashes, regenerated with the fixed "
                            "generator_stage3 templates; no sampling"),
         "n_specs": len(specs), "n_failures": len(failures),
         "failures": failures, "protocol": "single_run_stages_2_5",
         "specs": specs}, indent=2))
    print(f"manifest: {len(specs)} specs, {len(failures)} failures")
    for f in failures:
        print("  FAIL:", f)
    return 0 if (specs and not failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
