#!/bin/bash
set -euo pipefail
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-4r9k
python3 experiments/gpu-sim-8zqr/build.py --families gemm_naive --n-per-family 24 --min-size 262144 --build-dir /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-4r9k/experiments/gpu-sim-4r9k/build_gemm
python3 experiments/gpu-sim-8zqr/measure.py --build-dir /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-4r9k/experiments/gpu-sim-4r9k/build_gemm --out /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-4r9k/experiments/gpu-sim-4r9k/measured_gemm.jsonl
