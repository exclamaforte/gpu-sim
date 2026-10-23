#!/bin/bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKB="$HERE/../gpu-sim-pkb"
STATUS="$HERE/run_all_status.tsv"
: > "$STATUS"
run_one() {
  local size=$1 proto=$2 runid=$3 instr=$4
  if [ "$instr" = "selfdesc" ]; then
    bash "$PKB/run_selfdesc.sh" "$size" "$runid" "$proto"
  else
    bash "$PKB/run_nonuniform.sh" "$size" "$runid" "$proto"
  fi
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%d\n' "$size" "$proto" "$runid" "$instr" "$rc" >> "$STATUS"
  echo ">>> runid=$runid rc=$rc"
}
for size in 65536 131072 262144 393216 524288 1048576; do
  for r in 1 2 3; do
    run_one "$size" Simple "sd_${size}_simple_r${r}" selfdesc
  done
done
for size in 524288 1048576; do
  for r in 1 2 3; do
    run_one "$size" Simple "nu_${size}_simple_r${r}" nonuniform
  done
done
echo "ALL DONE"
cat "$STATUS"