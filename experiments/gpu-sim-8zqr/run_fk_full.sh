#!/bin/bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/fk/family_kernel.out"
TRACES="$HERE/traces_fk"
mkdir -p "$TRACES"
STATUS="$HERE/stage_fk_status.tsv"
: > "$STATUS"
TRIALS=100
BATCH=40
while IFS=' ' read -r fam n block; do
    [ -z "$fam" ] && continue
    runid="${fam}_${n}_${block}"
    out="$TRACES/${runid}.txt"
    if [ -f "$out" ]; then echo "skip $runid"; continue; fi
    "$BIN" "$fam" "$n" "$block" "$TRIALS" "$BATCH" > "$out" 2>&1
    rc=$?
    printf '%s\t%s\t%s\t%d\n' "$fam" "$n" "$block" "$rc" >> "$STATUS"
done < "$HERE/rows.txt"
echo "DONE"
