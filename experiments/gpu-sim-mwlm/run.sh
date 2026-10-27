#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/close_requires_artifact.py >/dev/null
python3 scripts/check_gate_citations.py | tail -1
