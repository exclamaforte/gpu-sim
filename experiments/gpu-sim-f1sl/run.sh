#!/bin/bash
# Reproduce the f1sl fix: the two worktree fixtures must FAIL (not skip) when
# `git worktree add` fails, and both test files must run with zero skips.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$REPO/.lab/tmp/f1sl-replay.log}"
mkdir -p "$(dirname "$OUT")"
cd "$REPO"

CUDA_VISIBLE_DEVICES= python3 -m pytest -q -rs \
    tests/test_verify_committed_run.py \
    tests/test_zsd3_janitor_detection.py | tee "$OUT"

python3 - "$OUT" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
if "skipped" in text.lower():
    raise SystemExit("F1SL FAIL: a skip survived (the bug is back):\n" + text[-2000:])
if "failed" in text.lower():
    raise SystemExit("F1SL FAIL: a test failed:\n" + text[-2000:])
passed = re.search(r"(\d+) passed", text)
if not passed or int(passed.group(1)) != 8:
    raise SystemExit(f"F1SL FAIL: expected 8 passed, got {passed.group(0) if passed else 'none'}")
print(f"F1SL OK: {passed.group(1)} passed, 0 skipped, 0 failed")
PY
