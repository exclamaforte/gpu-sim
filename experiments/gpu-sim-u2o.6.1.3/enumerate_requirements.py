#!/usr/bin/env python3
"""Enumerate the REAL cudart requirement of the generated drivers
(gpu-sim-u2o.6.1.3) — by compiling them, not by guessing.

For each stage generator (1..5) we take the first few KernelSpecs, render
them through kernel_driver.synthesize() where the driver synthesiser
supports them, and compile with:

    nvcc -cudart=none -arch=sm_120 <src> -o <exe> -L. -lfakegpu

Anything that fails to link is reported with its missing symbols, so the
shim's surface can be extended from evidence instead of recall. Runs on CPU
only; the GPU is never touched.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.kernel_driver import synthesize  # noqa: E402

STAGES = ["generator_stage1", "generator_stage2", "generator_stage3",
          "generator_stage4", "generator_stage5"]
PER_STAGE = 3


def undefined_syms(stderr: str) -> list[str]:
    syms = set()
    for line in stderr.splitlines():
        if "undefined reference to" in line:
            syms.add(line.split("`")[-1].rstrip("'"))
    return sorted(syms)


def main() -> int:
    out = {"bead": "gpu-sim-u2o.6.1.3",
           "stages": {}, "all_linked": True}
    tmp = Path(tempfile.mkdtemp())
    arch = "sm_120"

    for stage in STAGES:
        mod_name = f"epic_u2o.{stage}"
        rec = {"linked": 0, "synthesizable": 0, "unlinkable_due_to_unsupported_spec": 0,
               "failures": [], "missing_symbols": []}
        try:
            mod = __import__(mod_name, fromlist=["generate_kernels"])
        except Exception as e:
            rec["import_error"] = f"{type(e).__name__}: {e}"
            out["stages"][stage] = rec
            out["all_linked"] = False
            continue

        try:
            specs = []
            for i, spec in enumerate(mod.generate_kernels()):
                specs.append(spec)
                if len(specs) >= PER_STAGE:
                    break
        except Exception as e:
            rec["generate_error"] = f"{type(e).__name__}: {e}"
            out["stages"][stage] = rec
            continue

        for j, spec in enumerate(specs):
            d = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
            # The synthesiser only handles the stage-1 elementwise family
            # (float/int/double pointers + scalars) and fails loudly otherwise.
            try:
                src = synthesize(d)
                # generator_stage2..5 emit a `kernel_name` that does not match
                # the function name inside `source` (e.g. kernel_name
                # 'reduce_sum_block_kernel' vs a __global__ named
                # 'reduce_sum_block'), so the synthesized call site references a
                # kernel that does not exist. That is a GENERATOR bug, not a shim
                # gap -- reported separately -- but it would mask the cudart
                # surface we are here to enumerate, so align the call site with
                # the real function name before compiling.
                m = re.search(r"__global__\s+(?:void|extern\s+\"C\")\s+"
                              r"(?:\w+\s+)*(\w+)\s*\(", d["source"])
                if m and m.group(1) != d["kernel_name"]:
                    d2 = dict(d)
                    d2["kernel_name"] = m.group(1)
                    rec.setdefault("kernel_name_mismatches", []).append(
                        {"spec": j, "declared": d["kernel_name"], "actual": m.group(1)})
                    d = d2
                    src = synthesize(d)
            except Exception as e:
                rec["unlinkable_due_to_unsupported_spec"] += 1
                rec["failures"].append({"spec": j, "stage": "synthesize",
                                        "error": f"{type(e).__name__}: {e}"})
                continue
            rec["synthesizable"] += 1
            cu = tmp / f"{stage}_{j}.cu"
            cu.write_text(src)
            r = subprocess.run(
                ["nvcc", f"-cudart=none", f"-arch={arch}", "-O2", str(cu),
                 "-o", str(tmp / f"{stage}_{j}.out"), "-L", str(REPO), "-lfakegpu",
                 "-Xlinker", "-rpath", "-Xlinker", str(REPO)],
                capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                rec["linked"] += 1
            else:
                out["all_linked"] = False
                miss = undefined_syms(r.stderr)
                rec["missing_symbols"].extend(miss)
                rec["failures"].append({"spec": j, "stage": "link",
                                        "missing": miss,
                                        "stderr_head": r.stderr[:600]})
        rec["missing_symbols"] = sorted(set(rec["missing_symbols"]))
        out["stages"][stage] = rec

    # The canonical case: the stage-1 driver the harness actually synthesizes.
    from epic_u2o.kernel_driver import spec_from_stage1
    canon = tmp / "canon.cu"
    canon.write_text(synthesize(spec_from_stage1("scale", [4, 1, 1], [256, 1, 1], 1024)))
    r = subprocess.run(
        ["nvcc", "-cudart=none", f"-arch={arch}", "-O2", str(canon),
         "-o", str(tmp / "canon.out"), "-L", str(REPO), "-lfakegpu",
         "-Xlinker", "-rpath", "-Xlinker", str(REPO)],
        capture_output=True, text=True, timeout=600)
    out["canonical_stage1_driver"] = {"linked": r.returncode == 0,
                                      "missing": undefined_syms(r.stderr)}

    run = subprocess.run(["bash", str(HERE.parent.parent / "tests" / "test_cudart_none.sh")],
                         capture_output=True, text=True, timeout=900, cwd=str(REPO))
    out["test_cudart_none_sh"] = {"rc": run.returncode,
                                  "output": run.stdout[-2000:]}
    if run.returncode != 0:
        out["all_linked"] = False

    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "stages"}, indent=2)[:1500])
    for stage, rec in out["stages"].items():
        print(f"{stage}: linked={rec['linked']} synthesized={rec['synthesizable']} "
              f"unsupported={rec['unlinkable_due_to_unsupported_spec']} "
              f"missing={rec['missing_symbols']}")
    return 0 if out["all_linked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
