#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
python3 experiments/gpu-sim-8zqr/build.py --families gemm_naive --n-per-family 24 --min-size 262144 --build-dir experiments/gpu-sim-4r9k/build_gemm
python3 experiments/gpu-sim-8zqr/measure.py --build-dir experiments/gpu-sim-4r9k/build_gemm --out experiments/gpu-sim-4r9k/measured_gemm.jsonl
