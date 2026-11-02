#!/bin/bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/fk/family_kernel.out"
TRACES="$HERE/traces_fk_trial"
mkdir -p "$TRACES"
STATUS="$HERE/stage_fk_trial_status.tsv"
: > "$STATUS"
TRIALS=100
BATCH=40
START=$(date +%s)
while IFS=' ' read -r fam n block; do
    [ -z "$fam" ] && continue
    runid="${fam}_${n}_${block}"
    out="$TRACES/${runid}.txt"
    if [ -f "$out" ]; then echo "skip $runid"; continue; fi
    "$BIN" "$fam" "$n" "$block" "$TRIALS" "$BATCH" > "$out" 2>&1
    rc=$?
    printf '%s\t%s\t%s\t%d\n' "$fam" "$n" "$block" "$rc" >> "$STATUS"
    echo "fam=$fam n=$n block=$block rc=$rc"
done < "$HERE/rows_trial.txt"
echo "DONE"
