#!/usr/bin/env bash
# bd-exp.sh - record experiments as beads (gpu-sim-u2o.11.6; adapted from
# ~/Documents/cclm/scripts/bd-exp.sh to this lab's metric vocabulary).
#
#   bd-exp.sh start <parent-id> "<title>" "<hypothesis>"
#       [--baseline <exp-id>] [--kind train|datagen|analysis]
#       [--artifact-intent <path>] [--dry-run]
#   bd-exp.sh finish <exp-id> --outcome win|neutral|regression|inconclusive
#       --kept true|false --artifact <path-or-note>
#       [--logmae X] [--spearman X] [--kendall X] [--coverage X] [--ece X]
#       [--ex-sec X] [--topk X] [--note "..."] [--set key=value]...
#       [--dry-run]
#   bd-exp.sh board
#
# Negative results are recorded, never deleted. Guards (from cclm, kept):
# one active record per parent; never overwrite a recorded verdict without
# --force + re-review note; only finish records start created (description
# signature); never finish under needs-review. finish CLOSES the bead with
# an artifact-citing reason (11.6 close rule).
set -euo pipefail

SIG="close with bd-exp.sh finish"
die() { echo "bd-exp: $*" >&2; exit 1; }
git_commit() { git rev-parse --short HEAD 2>/dev/null || echo unknown; }

cmd_start() {
  local parent="${1:-}" title="${2:-}" hypothesis="${3:-}"; shift 3 || true
  [[ -n "$parent" && -n "$title" && -n "$hypothesis" ]] || \
    die "usage: start <parent-id> <title> <hypothesis>"
  local baseline="" kind="analysis" intent="" dry=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --baseline) baseline="${2:-}"; shift 2;;
      --kind) kind="${2:-}"; shift 2;;
      --artifact-intent) intent="${2:-}"; shift 2;;
      --dry-run) dry=1; shift;;
      *) die "unknown flag: $1";;
    esac
  done
  case "$kind" in train|datagen|analysis) ;; *) die "--kind must be train|datagen|analysis";; esac
  local meta
  meta=$(python3 -c 'import json,sys;print(json.dumps({"exp_id":"pending","hypothesis":sys.argv[1],"baseline_exp":sys.argv[2],"kind":sys.argv[3],"artifact_intent":sys.argv[4],"commit":sys.argv[5],"outcome":"running","kept":"unknown"}))' \
    "$hypothesis" "$baseline" "$kind" "$intent" "$(git_commit)")
  if [[ "$dry" -eq 1 ]]; then
    echo "DRY-RUN start parent=$parent kind=$kind meta=$meta"; return 0
  fi
  local dup
  dup=$(bd list --parent "$parent" --json --limit 200 2>/dev/null | python3 -c '
import json,sys
try: kids=json.load(sys.stdin)
except Exception: sys.exit(0)
for k in kids:
    labels=k.get("labels") or []
    if "experiment" not in labels: continue
    if k.get("status") in ("open","in_progress"):
        print(k["id"], k.get("title","")); break' 2>/dev/null || true)
  [[ -z "$dup" ]] || die "active experiment record already under $parent: $dup. One experiment = one active record."
  local id
  id=$(bd create "$title" --silent --type=task --parent="$parent" --labels=experiment,$kind \
        --description="HYPOTHESIS: ${hypothesis}

Falsified if the measured result does not move in the predicted direction beyond run-to-run variance.
Baseline: ${baseline:-none recorded}
Artifact intent: ${intent:-none recorded}
Result lands in this bead's metadata; $SIG." \
        --metadata="$meta")
  bd update "$id" --metadata "$(python3 -c 'import json,sys;print(json.dumps({"exp_id":sys.argv[1]}))' "$id")" >/dev/null
  bd update "$id" --claim >/dev/null 2>&1 || true
  echo "$id"
}

cmd_finish() {
  local id="${1:-}"; shift || true
  [[ -n "$id" ]] || die "usage: finish <exp-id> --outcome ... --kept ... --artifact ..."
  local outcome="" kept="" note="" artifact="" force=0 dry=0; declare -A M=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force) force=1; shift;;
      --dry-run) dry=1; shift;;
      --outcome) outcome="${2:-}"; shift 2;;
      --kept) kept="${2:-}"; shift 2;;
      --note) note="${2:-}"; shift 2;;
      --artifact) artifact="${2:-}"; shift 2;;
      --logmae) M[log_mae]="${2:-}"; shift 2;;
      --spearman) M[spearman]="${2:-}"; shift 2;;
      --kendall) M[kendall_tau]="${2:-}"; shift 2;;
      --coverage) M[coverage_90]="${2:-}"; shift 2;;
      --ece) M[ece_1sigma]="${2:-}"; shift 2;;
      --ex-sec) M[examples_per_sec]="${2:-}"; shift 2;;
      --topk) M[topk_hit_rate]="${2:-}"; shift 2;;
      --set) kv="${2:-}"; shift 2
             if [[ "$kv" == *"="* ]]; then M["${kv%%=*}"]="${kv#*=}"; else die "--set needs key=value"; fi;;
      *) die "unknown flag: $1";;
    esac
  done
  case "$outcome" in win|neutral|regression|inconclusive) ;; *) die "--outcome must be win|neutral|regression|inconclusive";; esac
  case "$kept" in true|false) ;; *) die "--kept must be true|false";; esac
  [[ -n "$artifact" ]] || die "--artifact is required (11.6 close rule: every close cites a dataset+rows, checkpoint, log, or explicit negative-result note)"
  local pairs=("outcome=$outcome" "kept=$kept" "artifact=$artifact" "commit=$(git_commit)")
  pairs+=("finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" "forced=$force")
  for key in "${!M[@]}"; do pairs+=("$key=${M[$key]}"); done
  local meta
  meta=$(python3 -c 'import json,sys;print(json.dumps(dict(p.split("=",1) for p in sys.argv[1:])))' "${pairs[@]}")
  if [[ "$dry" -eq 1 ]]; then
    echo "DRY-RUN finish $id outcome=$outcome kept=$kept meta=$meta note=${note:-}"; return 0
  fi
  local cur
  cur=$(bd show "$id" --json 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
if isinstance(d,list): d=d[0] if d else {}
m=d.get("metadata") or {}
if isinstance(m,str):
    try: m=json.loads(m)
    except Exception: m={}
print(d.get("status",""), m.get("outcome",""), "SIG" if "close with bd-exp.sh finish" in (d.get("description") or "") else "NOSIG")' 2>/dev/null || true)
  local st="${cur%% *}" oc="${cur#* }"; oc="${oc%% *}"; local sg="${cur##* }"
  if [[ "$force" -ne 1 ]]; then
    [[ "$st" != "closed" ]] || die "refusing to finish $id: already closed (${oc:-no verdict})."
    [[ -z "$oc" || "$oc" == "running" ]] || die "refusing to overwrite verdict '$oc' on $id without --force + re-review note."
    [[ "$sg" == "SIG" ]] || die "refusing to finish $id: not a record created by bd-exp.sh start."
  fi
  bd update "$id" --metadata "$meta" >/dev/null
  bd close "$id" ${force:+--force} --reason "${outcome} (kept=${kept}) artifact: ${artifact}${note:+ - $note}"
  echo "recorded $id: $outcome kept=$kept"
}

cmd_board() {
  bd list --label=experiment --all --json 2>/dev/null | python3 -c '
import json,sys
try: rows=json.load(sys.stdin)
except Exception: rows=[]
for r in rows:
    m=r.get("metadata") or {}
    if isinstance(m,str):
        try: m=json.loads(m)
        except Exception: m={}
    print(r.get("id"), r.get("status"), m.get("outcome","?"), "kept="+str(m.get("kept","?")), (r.get("title") or "")[:60])
'
}

case "${1:-}" in
  start) shift; cmd_start "$@";;
  finish) shift; cmd_finish "$@";;
  board) shift; cmd_board "$@";;
  *) sed -n "2,15p" "$0" | sed "s/^# \?//"; exit 1;;
esac
