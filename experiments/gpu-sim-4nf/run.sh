#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 experiments/gpu-sim-4nf/snapshot.py
python3 experiments/gpu-sim-4nf/instrument.py
