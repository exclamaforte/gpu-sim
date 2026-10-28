#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_subgroup_grank.py -q
