#!/bin/bash
set -u
FAMILY=${1:?usage: run_sweep_single.sh <family>}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/family_kernel.out"
TRACES="$HERE/traces"
mkdir -p "$TRACES"
STATUS="$HERE/sweep_single_status_${FAMILY}.tsv"
: > "$STATUS"
SIZES="1024 16384 262144 4194304 16777216"
TRIALS=100
BLOCK=256
BATCH=1
for n in $SIZES; do
  for inv in 1 2 3; do
    runid="${FAMILY}_${n}_single_inv${inv}"
    out="$TRACES/${runid}.txt"
    "$BIN" "$FAMILY" "$n" "$BLOCK" "$TRIALS" "$BATCH" > "$out" 2>&1
    rc=$?
    printf '%s\t%s\t%d\t%d\n' "$FAMILY" "$n" "$inv" "$rc" >> "$STATUS"
    echo "family=$FAMILY n=$n inv=$inv rc=$rc"
  done
done
echo "DONE $FAMILY"