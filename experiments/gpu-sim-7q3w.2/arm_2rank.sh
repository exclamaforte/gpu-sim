#!/usr/bin/env bash
# gpu-sim-7q3w.2, 2-rank acceptance arm: does NCCL USE our v11 plugin?
#
# Two processes, real libnccl, our plugin via NCCL_NET_PLUGIN.  Two ranks on one
# box must look like two NODES, or NCCL applies its intra-node paths (P2P/SHM)
# and the net plugin is never asked for a connection.  NCCL_HOSTID does that
# (verified present in libnccl 2.32.6's strings) without UTS namespaces.
#
# The fake CUDA layer is the same trick as gpu-sim-7q3w.3 and for the same
# measured reason: this box has ONE NVIDIA GPU, so two real ranks cannot each
# have a device (`no CUDA-capable device is detected` in job 20261122t113451).
# libfakegpu.so is installed as libcudart.so.13 in a directory that comes first
# on LD_LIBRARY_PATH; it is NOT preloaded, because it also exports nccl* symbols
# and would preempt real NCCL (proven in 7q3w.3: 1048576 wrong elements, no
# NCCL_DEBUG at all).  The real libcuda.so.1 stays: NCCL's driver wrapper checks
# cuDriverGetVersion, which libfakegpu does not export.
#
# CPU only.  Not gpqueue: the GPU queue killed the 7q3w.3 arm at 0.2% util, and
# rightly -- this work has no GPU in it.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/gpu-sim-7q3w.2"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
CUDA="${CUDA_SYSROOT:-$HOME/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux}"
WHEEL="${1:-cu13}"
LIBS="$EXP/libs/$WHEEL"
RUN="${RUN_DIR:-$EXP/run-$(date +%Y%m%dT%H%M%S)-$WHEEL}"
N="${ARM_N:-1048576}"

fail() { echo "ARM FAILED: $*" >&2; exit 1; }
ok()   { echo "ASSERT ok: $*"; }
bad()  { echo "ASSERT FAIL: $*"; FAILS=$((FAILS+1)); }
FAILS=0

[[ -s "$LIBS/libnccl.so.2" ]] || fail "no libnccl in $LIBS"
[[ -f "$MAIN/libfakegpu.so" ]] || fail "libfakegpu.so missing at $MAIN (make on main)"
mkdir -p "$RUN" "$RUN/ids"

PLUGIN="$RUN/libncclNetPlugin.so"
gcc -shared -fPIC -O1 -Wall -o "$PLUGIN" -I"$EXP/vendor/example" "$EXP/plugin_accept.c" \
  || fail "plugin build"
# nccl.h comes from the WHEEL (vendor/nccl.h, 2.32.6) -- the site-packages copy
# on this box is 2.32.3, and compiling against a different minor than the .so is
# the same class of mistake this bead exists to remove.
gcc -O1 -Wall -o "$RUN/arm_plugin" -I"$EXP/vendor" -I"$CUDA/include" \
  "$EXP/arm_plugin.c" -L"$LIBS" -lnccl -L"$CUDA/lib" -lcudart -ldl \
  -Wl,-rpath,"$LIBS":"$CUDA/lib" 2>"$RUN/build.log" || { cat "$RUN/build.log"; fail "arm build"; }

FAKECUDA="$RUN/fakecuda"; mkdir -p "$FAKECUDA"
for so in libcudart.so libcudart.so.13; do ln -sfn "$MAIN/libfakegpu.so" "$FAKECUDA/$so"; done

export LD_LIBRARY_PATH="$FAKECUDA:$LIBS:$CUDA/lib"
export FAKEGPU_NUM_DEVICES=2
export NCCL_DEBUG=${NCCL_DEBUG-INFO}
export NCCL_DEBUG_SUBSYS=INIT,NET,TRANSPORT
export NCCL_NET_PLUGIN="$PLUGIN"
# 2.32's optional subsystems crash on the fake device before any collective
# (measured in 7q3w.3); the plugin arm does not model them either.
export NCCL_GIN_ENABLE=0 NCCL_RMA_DISABLE=1
export ARM_ID_DIR="$RUN/ids"

run_rank() {
  local r=$1
  NCCL_HOSTID="node$r" ARM_CUDA_DEV=$r \
  FAKEIB_PLUGIN_GUID="0xFACE1B000000000$r" \
  FAKEIB_PLUGIN_PCI="0$r:00.0" \
  FAKEIB_PLUGIN_COUNTERS="$RUN/counters-rank$r.json" \
    "$RUN/arm_plugin" "$r" 2 "$N" >"$RUN/rank$r.log" 2>&1
  echo $? > "$RUN/rank$r.rc"
}

echo "=== arm: 2 ranks, NCCL v11 plugin acceptance ($WHEEL) ==="
( run_rank 1 ) &
R1=$!
run_rank 0
wait $R1

R0=$(cat "$RUN/rank0.rc" 2>/dev/null || echo missing)
RC1=$(cat "$RUN/rank1.rc" 2>/dev/null || echo missing)
echo "rank0 rc=$R0 rank1 rc=$RC1  (log dir $RUN)"

[[ "$R0" == 0 ]] && ok "rank0 exited 0" || bad "rank0 exit line (rc=$R0)"
[[ "$RC1" == 0 ]] && ok "rank1 exited 0" || bad "rank1 exit line (rc=$RC1)"

# (b) the plugin is loaded AND used -- measured in NCCL's own log, then in the
# plugin's counters, because a log line alone can be satisfied by a load we never
# drove past getProperties (that is exactly what step 1's refusal looked like).
for r in 0 1; do
  L="$RUN/rank$r.log"
  grep -q "Loaded net plugin" "$L" && ok "rank$r: NCCL loaded our net plugin" \
    || bad "rank$r: NCCL never loaded our net plugin"
  grep -q "External network plugin" "$L" && bad "rank$r: NCCL rejected it as unsupported" || true
  grep -qE "via NET/(Plugin|fakeib-net)" "$L" && ok "rank$r: a channel was assigned via our net plugin" \
    || bad "rank$r: no channel went via our net plugin (Socket fallback?)"
  grep -qE "via (P2P|SHM)" "$L" && bad "rank$r: intra-node transport used, ranks were not seen as 2 nodes" || true
  grep -q "NULL DATA PATH" "$L" && ok "rank$r: the null data path announced itself" \
    || bad "rank$r: null data path did not announce itself"
done

# (e) the properties NCCL accepted are the ones our interposer defines, so a
# plugin that "accepted" with garbage cannot pass.
for r in 0 1; do
  C="$RUN/counters-rank$r.json"
  [[ -s "$C" ]] || { bad "rank$r: no counters file"; continue; }
  python3 - "$C" "$r" <<'PY' || FAILS=$((FAILS+1))
import json, sys
c = json.load(open(sys.argv[1])); r = sys.argv[2]
def need(cond, msg):
    print(("ASSERT ok: " if cond else "ASSERT FAIL: ") + msg)
    if not cond: sys.exit(1)
need(c["props_speed"] == 400000, f"rank{r} props.speed == 400000 Mbps (400 Gb/s), got {c['props_speed']}")
need(c["props_guid"] == int(f"0xFACE1B000000000{r}", 16),
     f"rank{r} props.guid is the value our interposer set, got 0x{c['props_guid']:x}")
need(c["getProperties_calls"] >= 1, f"rank{r} getProperties called ({c['getProperties_calls']})")
need(c["devices_calls"] >= 1, f"rank{r} devices() called ({c['devices_calls']})")
need(c["accept_calls"] >= 1, f"rank{r} NCCL entered accept() on our plugin ({c['accept_calls']})")
need(c["isend_calls"] + c["irecv_calls"] >= 1,
     f"rank{r} NCCL posted work through our plugin (isend={c['isend_calls']} irecv={c['irecv_calls']})")
need(c["bytes_claimed"] > 0, f"rank{r} counters record claimed bytes ({c['bytes_claimed']})")
PY
done

echo "--- rank0 transport lines ---"
grep -E "NET/Plugin|Loaded net plugin|Channel 0[0-9]/" "$RUN/rank0.log" | head -6
echo "--- rank0 counters ---"; cat "$RUN/counters-rank0.json" 2>/dev/null; echo

if [ "$FAILS" -eq 0 ]; then
  echo "ARM OK ($WHEEL): NCCL accepted and USED the v11 plugin"
  exit 0
fi
echo "ARM FAILED: $FAILS assertion(s) ($WHEEL)"
exit 1
