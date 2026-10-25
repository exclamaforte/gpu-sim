#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 experiments/gpu-sim-6bi/calibrate.py
