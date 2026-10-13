#!/usr/bin/env bash
# gpu-sim-urr -- recompute + evidence. CPU-ONLY, device-free throughout
# (libfakegpu answers registration and launch; no device touched).
# Never writes a tracked file: builds go to gitignored build/, scratch to
# $XDG_CACHE_HOME, comparisons against committed metrics.json.
#
# Evidence collected:
#  1. identity: test_launch_identity.cu trace (FAKEGPU_DEBUG_REGISTRY=1) --
#     every launch pointer must equal a registered pointer AND resolve to
#     its registered name. Verdict computed, not eyeballed.
#  2. attach gate: build + run tests/test_resolver_attach.c (also in check).
#  3. timing: bench scalar lower-bound (default path sleeps the configured
#     latency) + resolver0 replacement + per-launch lookup cost vs the
#     1400 ns launch-bound subject (6.6.6.1 rule: bar 700 ns).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=
SCRATCH="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/urr-recompute-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p build
nice -n 10 make > "$SCRATCH/make.log" 2>&1 || { echo "FAIL make"; tail -n 3 "$SCRATCH/make.log"; exit 1; }

echo "--- 1. identity ---"
nice -n 10 nvcc -cudart=none -arch=sm_120 -Xlinker -rpath -Xlinker "$PWD" tests/test_launch_identity.cu -o build/test_launch_identity -L. -lfakegpu || exit 1
FAKEGPU_DEBUG_REGISTRY=1 ./build/test_launch_identity > "$SCRATCH/identity.out" 2> "$SCRATCH/identity.err" || exit 1
cat "$SCRATCH/identity.out"

echo "--- 2. attach gate ---"
nice -n 10 gcc -O2 -g -Iinclude -Ishim -o build/test_resolver_attach tests/test_resolver_attach.c -L. -lfakegpu -Wl,-rpath,"$PWD" || exit 1
./build/test_resolver_attach > "$SCRATCH/attach.out" 2>&1 || { cat "$SCRATCH/attach.out"; exit 1; }
tail -n 1 "$SCRATCH/attach.out"

echo "--- 3. timing ---"
nice -n 10 gcc -O2 -g -Iinclude -Ishim -o build/bench_launch experiments/gpu-sim-urr/bench_launch.c -L. -lfakegpu -Wl,-rpath,"$PWD" || exit 1
FAKEGPU_KERNEL_LATENCY_MS=20 ./build/bench_launch scalar 50 | tee "$SCRATCH/bench_scalar.out"
./build/bench_launch resolver0 2000 | tee "$SCRATCH/bench_r0.out"

CUDA_VISIBLE_DEVICES= nice -n 10 python3 - "$SCRATCH" "$HERE/metrics.json" <<'EOF'
import json, re, sys
scratch, committed = sys.argv[1], sys.argv[2]
err = open(f"{scratch}/identity.err").read()
regs = re.findall(r"\[fg-register\] deviceFun=(\S+) hostFun=(\S+) name=(\S+)", err)
launches = re.findall(r"\[fg-launch-dyn\] kernel=(\S+) name=(\S+)", err)
assert len(regs) == 2, f"expected 2 registrations, trace has {len(regs)}"
assert len(launches) == 2, f"expected 2 launches, trace has {len(launches)}"
keys = {r[0]: r[2] for r in regs} | {r[1]: r[2] for r in regs}
for ptr, name in launches:
    assert ptr in keys, f"launch pointer {ptr} was never registered"
    assert keys[ptr] == name, f"{ptr} resolved {name}, registered {keys[ptr]}"
scalar = open(f"{scratch}/bench_scalar.out").read()
r0 = open(f"{scratch}/bench_r0.out").read()
total_scalar = int(re.search(r"total_ns=(\d+)", scalar).group(1))
total_r0 = int(re.search(r"total_ns=(\d+)", r0).group(1))
per_r0 = float(re.search(r"per_launch_ns=([\d.]+)", r0).group(1))
# Acceptance 3: scalar path sleeps the configured latency (lower bound only).
assert total_scalar >= 0.9 * 50 * 20e6, f"scalar path short: {total_scalar}"
# Replacement: resolver0 must take a small fraction of the scalar expectation.
assert total_r0 < 0.25 * 2000 * 20e6, f"resolver path did not replace scalar: {total_r0}"
# Acceptance 4: lookup cost vs the 1400 ns launch-bound subject (bar: half).
assert per_r0 < 700.0, f"lookup dominates its subject: {per_r0} ns/launch"
got = {
    "identity": {"n_registered": len(regs), "n_launches": len(launches),
                 "all_launches_resolve": True,
                 "deviceFun_eq_hostFun": any(r[0] == r[1] for r in regs),
                 "names": sorted({r[2] for r in regs})},
    "timing": {"scalar_50x20ms_total_ns": total_scalar,
               "resolver0_2000_total_ns": total_r0,
               "lookup_per_launch_ns": per_r0},
}
exp = json.load(open(committed))
print("identity+timing evidence recomputed")
# timing values vary run to run: gate on MARGINS, compare structure only.
assert exp["identity"]["n_registered"] == got["identity"]["n_registered"]
assert exp["identity"]["n_launches"] == got["identity"]["n_launches"]
assert exp["identity"]["all_launches_resolve"] is True
assert got["timing"]["scalar_50x20ms_total_ns"] >= 0.9 * 50 * 20e6
assert got["timing"]["lookup_per_launch_ns"] < 700.0
print(f"RECOMPUTE PASS: identity holds; lookup {per_r0:.1f} ns/launch "
      f"(subject ~1400 ns); scalar lower bound met ({total_scalar/1e6:.0f} ms)")
EOF
echo "run.sh rc=$?"
