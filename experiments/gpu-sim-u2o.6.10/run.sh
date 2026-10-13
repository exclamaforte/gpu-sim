#!/usr/bin/env bash
# gpu-sim-u2o.6.10 -- ONE supervised dose. ts only.
# Gate first: a failed dose wastes a shared queue slot and hangs a GPU three
# other projects use, so everything CPU-checkable is checked before submitting.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
python3 preflight_static.py || { echo "FATAL: gate failed; dose NOT spent"; exit 3; }
echo "=== ts job $(date -Is) ==="
python3 validate_once.py 2>&1 | tee output.log
rc=${PIPESTATUS[0]}
echo "=== exit=$rc ==="
exit "$rc"
