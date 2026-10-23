#!/bin/bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/family_kernel.out"
# fallback if not present
[ -x "$BIN" ] || BIN="$HERE/family_kernel.out"
TRACES="$HERE/traces"
mkdir -p "$TRACES"
STATUS="$HERE/stage1_status.tsv"
: > "$STATUS"
TRIALS=100
BATCH=40
START=$(date +%s)
BUDGET_SEC=7200
STOPPED=0
while IFS=' ' read -r fam n block; do
    [ -z "$fam" ] && continue
    NOW=$(date +%s)
    ELAPSED=$((NOW-START))
    if [ $ELAPSED -ge $BUDGET_SEC ]; then
        echo "BUDGET EXCEEDED at ${ELAPSED}s; stopping"
        STOPPED=1
        break
    fi
    # rate check after 30 min
    if [ $ELAPSED -ge 1800 ]; then
        USABLE=$(grep -c "usable" "$STATUS" 2>/dev/null || echo 0)
        # count rows processed
        NROWS=$(wc -l < "$STATUS")
        RATE=$(python3 -c "print(($NROWS)/($ELAPSED/3600.0))")
        # if rate < 100 usable/hour, stop
        # we approximate usable rate by processed rows; gate rejection handled later
        python3 -c "import sys; sys.exit(0 if float('$RATE')>=100 else 1)" || { echo "RATE ${RATE} < 100 rows/h; stopping"; STOPPED=1; break; }
    fi
    runid="${fam}_${n}_${block}"
    out="$TRACES/${runid}.txt"
    if [ -f "$out" ]; then
        echo "skip $runid (exists)"
        continue
    fi
    "$BIN" "$fam" "$n" "$block" "$TRIALS" "$BATCH" > "$out" 2>&1
    rc=$?
    printf '%s\t%s\t%s\t%d\t%d\n' "$fam" "$n" "$block" "$rc" "$ELAPSED" >> "$STATUS"
    echo "fam=$fam n=$n block=$block rc=$rc elapsed=${ELAPSED}s"
done < "$HERE/rows.txt"
echo "STOPPED=$STOPPED" >> "$STATUS"
echo "DONE stage1"