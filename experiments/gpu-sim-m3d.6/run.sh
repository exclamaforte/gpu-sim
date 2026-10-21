#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# dry-run plan + a real 2-rank run
python3 scripts/gpusim run --nodes 1 --gpus-per-node 2 --dry-run -- echo hi
python3 scripts/gpusim run --nodes 1 --gpus-per-node 2 --timeout 10s -- python3 -c "import os; print(os.environ['GPUSIM_RANK'])" >/tmp/gpusim_run.out 2>&1
cat /tmp/gpusim_run.out
test -f runs/*/run.json || true
echo "DONE"
