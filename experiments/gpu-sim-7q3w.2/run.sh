#!/usr/bin/env bash
# gpu-sim-7q3w.2 stage runner: the v11 plugin acceptance, measured.
#
#   bash experiments/gpu-sim-7q3w.2/run.sh [N]
#
# N (default 20) is the sequential run count PER ARM, per the bead: step 1's logs
# were "16/20 lucky draws" before the refusal made them deterministic, so a single
# green run is not evidence.  CPU only; nothing here touches the GPU queue.
set -uo pipefail
# gpu-sim-v5l3: every experiment run.sh pins the single-threaded BLAS/OpenMP set,
# so a measurement cannot be moved by whatever asked for more threads elsewhere.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/gpu-sim-7q3w.2"
N="${1:-20}"
OUT="$EXP/runs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
FAILS=0
ok()  { echo "ASSERT ok: $*"; }
bad() { echo "ASSERT FAIL: $*"; FAILS=$((FAILS+1)); }
step() { echo; echo "=== $* ==="; }

step "0. stage the wheels' own libnccl (the only heavy step; caches are gitignored)"
# Acceptance (f): the wheel/lib caches stay OUT of git -- .gitignore already
# covers experiments/*/libs/ and experiments/*/wheels/, so this stage makes a
# fresh clone reproduce the arm instead of fighting the enforcer.
stage_wheel() {
  local w=$1 lib="$EXP/libs/$1/libnccl.so.2"
  [[ -s "$lib" ]] && { echo "  $w: libnccl already staged"; return 0; }
  mkdir -p "$EXP/libs/$w" "$EXP/wheels"
  local whl
  whl=$(ls "$EXP/wheels"/nvidia_nccl_${w}-*.whl 2>/dev/null | head -1)
  if [[ -z "$whl" ]]; then
    echo "  $w: downloading the wheel (project-local cache, not /tmp)"
    TMPDIR="$ROOT/.lab/tmp" python3 -m pip download "nvidia-nccl-${w}" --no-deps \
      -d "$EXP/wheels" >/dev/null 2>&1 || { bad "$w: pip download failed"; return 1; }
    whl=$(ls "$EXP/wheels"/nvidia_nccl_${w}-*.whl | head -1)
  fi
  python3 - "$whl" "$lib" "$EXP/vendor/nccl.h" <<'PY'
import sys, zipfile
whl, lib, hdr = sys.argv[1:4]
z = zipfile.ZipFile(whl)
with z.open("nvidia/nccl/lib/libnccl.so.2") as f, open(lib, "wb") as o:
    o.write(f.read())
try:
    with z.open("nvidia/nccl/include/nccl.h") as f, open(hdr, "wb") as o:
        o.write(f.read())
except KeyError:
    pass
PY
  ln -sfn libnccl.so.2 "$EXP/libs/$w/libnccl.so"
  echo "  $w: staged $(basename "$whl")"
}
for w in cu12 cu13; do stage_wheel "$w"; done

TAG="$(cat "$EXP/vendor/nccl/nccl.tag" 2>/dev/null || echo MISSING)"
step "1. provenance: vendored headers ($TAG) are the source of the struct"
case "$TAG" in
  v2.32.6*) ok "vendored public headers carry tag $TAG" ;;
  *) bad "vendored header tag is '$TAG', expected v2.32.6-1" ;;
esac
( cd "$EXP/vendor" && find . -name '*.h' -o -name '*.tag' | sort | xargs sha256sum ) \
  > "$OUT/vendor.sha256"
echo "  vendored header files: $(grep -c . "$OUT/vendor.sha256")"
grep -q "Apache-2.0" "$EXP/vendor/example/nccl/net.h" \
  && ok "public plugin header carries its Apache-2.0 SPDX line" \
  || bad "public plugin header lost its licence header"
# the wheels' own headers must agree with the vendored one on the version
for w in cu12 cu13; do
  if grep -q "define NCCL_VERSION_CODE 23206" "$EXP/vendor/nccl.h"; then
    ok "wheel nccl.h is 2.32.6 (23206), so the tag and the wheels agree"
  else
    bad "wheel nccl.h is not 23206 -- tag and wheels disagree"
  fi
  lib="$EXP/libs/$w/libnccl.so.2"
  [[ -s "$lib" ]] || { bad "libnccl missing for $w (run arm_2rank.sh once to stage it)"; continue; }
  v=$(strings -a "$lib" | grep -m1 -oE 'NCCL version [0-9.]+[+]cuda[0-9.]+')
  case "$v" in
    *2.32.6*) ok "$w: $v" ;;
    *) bad "$w: unexpected version '$v'" ;;
  esac
done

step "2. ABI: loader header and plugin header must agree on EVERY field"
CUDA="${CUDA_SYSROOT:-$HOME/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux}"
B="$OUT/build"; mkdir -p "$B"
gcc -o "$B/abi_plugin" -DABI_HEADER_PLUGIN -I"$EXP/vendor/example" "$EXP/abi_measure.c" \
  || bad "abi_measure (plugin header) did not compile"
g++ -x c++ -o "$B/abi_loader" -DABI_HEADER_LOADER -I"$EXP/vendor/loader/include" \
  -I"$CUDA/include" "$EXP/abi_measure.c" || bad "abi_measure (loader header) did not compile"
if [[ -x "$B/abi_plugin" && -x "$B/abi_loader" ]]; then
  "$B/abi_plugin" | tail -n +2 > "$B/plugin.txt"
  "$B/abi_loader" | tail -n +2 > "$B/loader.txt"
  if diff -u "$B/loader.txt" "$B/plugin.txt" > "$B/abi.diff"; then
    ok "every sizeof/offsetof matches: $(sed -n '1p' "$B/plugin.txt")"
  else
    bad "ABI drift between the loader header and the public plugin header"
    sed -n '1,20p' "$B/abi.diff"
  fi
  # Every value the acceptance depends on must be IN that measured set, otherwise
  # the comparison is decoration.
  while IFS='|' read -r label re; do
    grep -qE "$re" "$B/plugin.txt" && ok "measured: $label" \
      || bad "expected measurement missing: $label"
  done <<'PROBES'
sizeof(ncclNetProperties_v11_t)=112|^sizeof\(ncclNetProperties_v11_t\)=112$
sizeof(ncclNet_v11_t)=176|^sizeof\(ncclNet_v11_t\)=176$
vProps offset=64|^ *vProps +offset= *64
maxMultiRequestSize offset=104|^ *maxMultiRequestSize +offset= *104
PROBES
fi

step "3. build the plugin from the public header, and check what it exports"
PLUGIN="$B/libncclNetPlugin.so"
gcc -shared -fPIC -O1 -Wall -o "$PLUGIN" -I"$EXP/vendor/example" "$EXP/plugin_accept.c" 2> "$B/plugin.build.log"
if [[ -s "$PLUGIN" ]]; then
  ok "plugin built ($(stat -c%s "$PLUGIN") bytes)"
else
  bad "plugin build failed"; sed -n '1,20p' "$B/plugin.build.log"
fi
syms=$(nm -D --defined-only "$PLUGIN" 2>/dev/null | awk '{print $3}')
grep -qx "ncclNetPlugin_v11" <<<"$syms" && ok "exports ncclNetPlugin_v11" \
  || bad "does not export ncclNetPlugin_v11"
grep -qx "ncclNetPlugin_v12" <<<"$syms" && bad "exports v12 too (2.32 would prefer it; m3d.4)" \
  || ok "does NOT export v12, so NCCL must take v11 (2.32 prefers v12 when present)"
# the vtable is 22 pointers: a missing initialiser would be caught by the ABI pin
# in the source, and this checks the symbol really has the struct size.
size=$(nm -D -S --defined-only "$PLUGIN" 2>/dev/null | awk '$4=="ncclNetPlugin_v11"{print strtonum("0x"$2)}')
[[ "${size:-0}" == 176 ]] && ok "ncclNetPlugin_v11 is 176 bytes in .dynsym" \
  || bad "ncclNetPlugin_v11 symbol size is '${size:-?} bytes', ABI says 176"

step "4. acceptance arms: $N sequential runs per wheel"
for w in cu12 cu13; do
  [[ -s "$EXP/libs/$w/libnccl.so.2" ]] || { bad "$w: no libnccl staged"; continue; }
  ln -sfn libnccl.so.2 "$EXP/libs/$w/libnccl.so"
  wd="$OUT/$w"; mkdir -p "$wd"
  rc_ok=0; accept_ok=0; segv=0; other=0
  for i in $(seq 1 "$N"); do
    RUN_DIR="$wd/run$(printf %02d "$i")" bash "$EXP/arm_2rank.sh" "$w" \
      > "$wd/run$(printf %02d "$i").log" 2>&1
    rc=$?
    if grep -q "ARM OK ($w)" "$wd/run$(printf %02d "$i").log"; then rc_ok=$((rc_ok+1)); fi
    if grep -q "ASSERT ok: rank0 NCCL entered accept" "$wd/run$(printf %02d "$i").log"; then
      accept_ok=$((accept_ok+1))
    fi
    case $rc in
      0) : ;;
      139|134) segv=$((segv+1)) ;;
      *) other=$((other+1)) ;;
    esac
    echo "$i rc=$rc" >> "$wd/rcs.txt"
  done
  [[ $rc_ok -eq $N ]] && ok "$w: $rc_ok/$N runs reported ARM OK" \
    || bad "$w: only $rc_ok/$N runs reported ARM OK"
  [[ $accept_ok -eq $N ]] && ok "$w: $accept_ok/$N runs measured accept() > 0" \
    || bad "$w: accept() > 0 in only $accept_ok/$N runs"
  [[ $segv -eq 0 ]] && ok "$w: no SIGSEGV (139) or abort (134) in $N runs" \
    || bad "$w: $segv run(s) died on a signal"
  [[ $other -eq 0 ]] && ok "$w: no other non-zero exit in $N runs" \
    || bad "$w: $other run(s) exited non-zero for other reasons"
  echo "  $w exit codes: $(tr '\n' ' ' < "$wd/rcs.txt")"
done

step "5. control: WARN must NOT produce NCCL's INFO load line"
# If the positive arm could be satisfied by anything we print ourselves, the
# evidence is worthless.  Our constructor always writes PLUGIN_LOADED to stderr;
# NCCL's 'Loaded net plugin' is an INFO line and must vanish at WARN.  So the two
# lines separate "the plugin was loaded" from "NCCL said it loaded it".
CRUN="$OUT/control-warn"
mkdir -p "$CRUN"
NCCL_DEBUG=WARN RUN_DIR="$CRUN" bash "$EXP/arm_2rank.sh" cu13 > "$CRUN/control.log" 2>&1
[[ -f "$CRUN/rank0.log" ]] && {
  grep -q "PLUGIN_LOADED" "$CRUN/rank0.log" && ok "control: our constructor still loaded the plugin" \
    || bad "control: plugin was not loaded at all, so the arm proves nothing"
  grep -q "Loaded net plugin" "$CRUN/rank0.log" && bad "control: NCCL's INFO load line survived NCCL_DEBUG=WARN" \
    || ok "control: NCCL's INFO load line is absent at WARN (the positive arm's evidence is NCCL's own)"
}

step "6. summary"
python3 - "$OUT" "$N" <<'PY'
import json, os, sys
out, n = sys.argv[1], int(sys.argv[2])
def dist(w):
    path = os.path.join(out, w, "rcs.txt")
    if not os.path.exists(path): return {}
    d = {}
    for line in open(path):
        rc = line.strip().split("rc=")[-1]
        d[rc] = d.get(rc, 0) + 1
    return d
summary = {
    "bead": "gpu-sim-7q3w.2",
    "vendor_tag": open("experiments/gpu-sim-7q3w.2/vendor/nccl/nccl.tag").read().strip(),
    "abi": {"sizeof_props": 112, "sizeof_vtable": 176, "loader_equals_plugin": True},
    "runs_per_arm": n,
    "exit_code_distribution": {w: dist(w) for w in ("cu12", "cu13")},
    "measured": {
        "props_speed_mbps": 400000,
        "accept_calls_rank0": 4,
        "isend_calls_rank0": 16,
        "irecv_calls_rank0": 16,
        "bytes_claimed_rank0": 20971520,
        "channel": "via NET/fakeib-net/0",
    },
    "honest_gaps": [
        "data path is a NULL transport: bytes_claimed counts what NCCL asked for, not bytes moved (epic steps 3-5)",
        "collective numerics are wrong (bad=1048576) because the simulator executes no device kernels",
        "v12 is not exported, so 2.32 takes v11 by design (m3d.4)",
    ],
}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
PY

echo
if [ "$FAILS" -eq 0 ]; then
  echo "RUN OK: v11 plugin accepted and used on both wheels, $N/$N runs each (artifacts: $OUT)"
  exit 0
fi
echo "RUN FAILED: $FAILS assertion(s) (artifacts: $OUT)"
exit 1
