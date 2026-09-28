# Pinned Toolchain & GPU Policy (v1)

Parent: gpu-sim-u2o.5.2

---

## 1. CUDA Toolkit / Driver / nvcc

For reproducible measurements on RTX 5090, pin:
- CUDA toolkit: `12.8` (or whatever the host has; record exact version in `measurement_spec.md` dataset row)
- Driver: `535.104.05` (record in `nvidia-smi -q` output attached to run)
- `nvcc --version` and `ptxas -v` outputs saved to `runs/<run_id>/toolchain.json`

---

## 2. GPU Clock / Power Policy

Before each measurement run:
- Lock GPU clock: `nvidia-smi -lgc 0` (or max stable clock)
- Set persistence mode: `nvidia-smi -pm 1`
- Set power limit to max: `nvidia-smi -pl <max_watts>`
- Record thermal state and clock frequency in dataset CSV
