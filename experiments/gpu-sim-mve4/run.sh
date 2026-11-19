#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_k9e4_registry_superset.py tests/test_e7yw_trunk_resolution.py -q
