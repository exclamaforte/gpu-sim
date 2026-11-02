#!/bin/bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/fk/family_kernel.out"
TRACES="$HERE/traces_fk"
mkdir -p "$TRACES"
STATUS="$HERE/stage_fk_status.tsv"
: > "$STATUS"
TRIALS=100
BATCH=4000
START=$(date +%s)
BUDGET_SEC=1200
while IFS=' ' read -r fam n block; do
    [ -z "$fam" ] && continue
    NOW=$(date +%s)
    ELAPSED=$((NOW-START))
    if [ $ELAPSED -ge $BUDGET_SEC ]; then echo "BUDGET EXCEEDED"; break; fi
    runid="${fam}_${n}_${block}"
    out="$TRACES/${runid}.txt"
    if [ -f "$out" ]; then echo "skip $runid"; continue; fi
    "$BIN" "$fam" "$n" "$block" "$TRIALS" "$BATCH" > "$out" 2>&1
    rc=$?
    printf '%s\t%s\t%s\t%d\t%d\n' "$fam" "$n" "$block" "$rc" "$ELAPSED" >> "$STATUS"
    echo "fam=$fam n=$n block=$block rc=$rc elapsed=${ELAPSED}s"
done < "$HERE/rows.txt"
echo "DONE"
