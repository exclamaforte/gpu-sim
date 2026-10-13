#!/usr/bin/env bash
# 0vk diagnostic: rerun the 4 failed 768M binaries, capture exit + stderr tail.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
for h in 1a14020380d85840 3deb300c35bee512 5252ae6618ab73ca 52b74687f0176489; do
  echo "=== $h ==="
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "no-smi"
  timeout 120 ./build/k_${h}.out > /tmp/diag_${h}.out 2> /tmp/diag_${h}.err
  echo "exit=$?"
  tail -n 3 /tmp/diag_${h}.out
  tail -n 3 /tmp/diag_${h}.err
done
echo "=== done ==="
