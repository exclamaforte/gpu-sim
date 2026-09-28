#!/usr/bin/env python3
"""Before/after demonstration for gpu-sim-8dw, self-contained.

Both sides are shown, and both modules are inside the repository:

  BEFORE  experiments/gpu-sim-8dw/pv_pre_fix.py   (the stub, restored verbatim)
  AFTER   epic_u2o/preflight_validator.py        (the repaired module)

Two probes, chosen because the stub passes both without running anything:

  1. a launch config naming a kernel that does not exist
  2. an empty launch config

and, for the repaired side, the same two probes plus one good kernel, so the
log shows a pass as well as the refusals -- a demonstration that only ever
prints failures would not distinguish "fixed" from "always fails".

No /tmp, no machine-specific path: `git clone` + `python3` is enough. The
reviewer's point on 8dw was exactly that the committed demonstration could not
be re-run by a reader, and that it only showed the 'before'.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # experiments/<id>/ -> worktree root
sys.path.insert(0, str(REPO))

GOOD = """
__global__ void add_k(const float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] + 1.0f;
}
"""
PROBES = [
    ("a kernel that does not exist", {"block_x": 999999}, "no_such_kernel"),
    ("an empty launch config", {}, "add_k"),
]


def load_pre_fix():
    spec = importlib.util.spec_from_file_location(
        "pv_pre_fix", HERE / "pv_pre_fix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    import epic_u2o.preflight_validator as fixed
    pre = load_pre_fix()

    print("=" * 72)
    print("gpu-sim-8dw -- the preflight that could not fail, before and after")
    print("=" * 72)
    print(f"pre-fix module : {(HERE / 'pv_pre_fix.py').relative_to(REPO)}")
    print(f"repaired module: epic_u2o/preflight_validator.py")

    print("\n--- BEFORE: _run_under_fakegpu as it was ---")
    pv = pre.PreflightValidator(arch="sm_120")
    for desc, cfg, kname in PROBES:
        ok, out, err = pv._run_under_fakegpu(dict(cfg), kname)
        print(f"  {desc:32s} -> passed={ok}  {out.strip()!r}")

    print("\n--- AFTER: the same two probes, plus one good kernel ---")
    fv = fixed.PreflightValidator(arch="sm_120")
    for desc, cfg, kname in PROBES:
        ok, out, err = fv._run_under_fakegpu(
            dict(cfg), kname, kernel_source=GOOD, problem_size=1024)
        print(f"  {desc:32s} -> passed={ok}  {(err or out).strip()[:90]!r}")
    res = fv.validate_kernel(GOOD, {"grid_x": 8, "block_x": 128,
                                   "problem_size": 4096}, "add_k")
    print(f"  {'a good kernel (end to end)':32s} -> passed={res.success}  "
          f"evidence: {'launch_done' in (res.stdout or '')}")

    print("\n--- AFTER: the case the stub could never satisfy ---")
    nv = fixed.PreflightValidator(arch="sm_120", fakegpu_lib="definitely-not-here.so")
    r = nv.validate_kernel(GOOD, {"grid_x": 8, "block_x": 128,
                                  "problem_size": 4096}, "add_k")
    print(f"  {'cannot check (no libfakegpu)':32s} -> passed={r.success}  "
          f"{str(r.errors)[:80]!r}")

    print("\nReading this honestly, which is the point of running it:")
    print("  * the stub answered True to EVERY probe, including one naming a")
    print("    kernel that does not exist and one with no config at all;")
    print("  * the repaired module refuses the impossible launch (999999 threads)")
    print("    and refuses outright when it cannot check;")
    print("  * but it PASSES the empty config, because an empty config defaults")
    print("    to a legal 1x1x1 launch and add_k really does run on the fake.")
    print("    That is correct behaviour, not a miss -- and an earlier draft of")
    print("    this file's closing sentence claimed the repaired module answered")
    print("    False to BOTH probes, which the output above contradicts. The")
    print("    sentence was written before the data existed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
