#!/usr/bin/env bash
# gpu-sim-m64 + gpu-sim-hje, one change (same comment block).
# Recompute check. CPU-ONLY: gcc/nvcc compile + device-free runs
# (-cudart=none, libfakegpu; no device touched). Never writes a tracked file:
# everything rebuilds into gitignored paths (build/, src/*.o) and compares
# against the committed metrics.json. WARN lines are evidence in output.log,
# NOT gates (their count is flaky across runs -- that flakiness is itself
# u2o.19's undiagnosed gap, which this bead must not adjudicate).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
SCRATCH="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/m64-recompute-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
export CUDA_VISIBLE_DEVICES=
nice -n 10 make > "$SCRATCH/make.log" 2>&1 || { echo "RECOMPUTE FAIL: make"; tail -n 5 "$SCRATCH/make.log"; exit 1; }
mkdir -p build
nice -n 10 gcc -O2 -g -Iinclude -Ishim -o build/test_shim_prop_size tests/test_shim_prop_size.c || exit 1
./build/test_shim_prop_size > "$SCRATCH/shim.out" 2>&1; shim_rc=$?
nice -n 10 nvcc -cudart=none -arch=sm_120 -Xlinker -rpath -Xlinker "$PWD" tests/test_device_prop_overlay.cu -o build/test_device_prop_overlay -L. -lfakegpu || exit 1
./build/test_device_prop_overlay > "$SCRATCH/overlay.out" 2>&1; overlay_rc=$?
CUDA_VISIBLE_DEVICES= nice -n 10 python3 - "$SCRATCH" "$HERE/metrics.json" <<'EOF'
import json, re, sys
scratch, committed = sys.argv[1], sys.argv[2]
shim = open(f"{scratch}/shim.out").read()
over = open(f"{scratch}/overlay.out").read()
def grab(pat, text, name):
    m = re.search(pat, text)
    assert m, f"missing {name} in test output"
    return int(m.group(1))
got = {
    "shim_sizeof": grab(r"sizeof\(struct cudaDeviceProp\) = (\d+)", shim, "shim sizeof"),
    "real_sizeof": grab(r"sizeof\(cudaDeviceProp\)\s+(\d+)", over, "real sizeof"),
    "l2_size": grab(r"sizeof\(l2CacheSize\)\s+(\d+)", over, "l2 sizeof"),
    "offsets": {
        "l2CacheSize": grab(r"offsetof\(l2CacheSize\)\s+(\d+)", over, "off596"),
        "maxThreadsPerMultiProcessor": grab(r"offsetof\(maxThreadsPerMultiProcessor\)\s+(\d+)", over, "off604"),
        "sharedMemPerMultiprocessor": grab(r"offsetof\(sharedMemPerMultiprocessor\)\s+(\d+)", over, "off624"),
    },
    "overlay_gate_passed": "gate PASSED" in over,
}
exp = json.load(open(committed))
bad = [k for k in ("shim_sizeof", "real_sizeof", "l2_size", "overlay_gate_passed")
       if got[k] != exp[k]]
bad += [k for k in ("l2CacheSize", "maxThreadsPerMultiProcessor", "sharedMemPerMultiprocessor")
        if got["offsets"][k] != exp["offsets"][k]]
if bad:
    print(f"RECOMPUTE FAIL: drift in {bad}: {got}");
    sys.exit(1)
print(f"RECOMPUTE PASS (stable keys match): {got}")
EOF
rc=$?
echo "shim_rc=$shim_rc overlay_rc=$overlay_rc driver_rc=$rc"
[ "$shim_rc" -eq 0 ] && [ "$overlay_rc" -eq 0 ] && [ "$rc" -eq 0 ] || exit 1
grep -h "WARN" "$SCRATCH"/shim.out "$SCRATCH"/overlay.out | sort -u | head -n 5
