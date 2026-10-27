#!/usr/bin/env bash
# Recompute the error-budget artefact and verify it matches the committed one.
set -euo pipefail
cd "$(dirname "$0")/../.."
scratch="$(mktemp -d .lab/tmp/hd5u.XXXXXX)"
trap 'rm -rf "$scratch"' EXIT
python3 scripts/error_budget.py --json > "$scratch/error_budget.json"
if ! diff -q experiments/gpu-sim-hd5u/error_budget.json "$scratch/error_budget.json" >/dev/null; then
  echo "MISMATCH: recomputed error_budget.json differs from committed" >&2
  diff experiments/gpu-sim-hd5u/error_budget.json "$scratch/error_budget.json" >&2 || true
  exit 1
fi
echo "error_budget.json reproduces byte-for-byte"
