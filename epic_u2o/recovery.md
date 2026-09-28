# GPU-Reset Recovery Procedure

Parent: gpu-sim-u2o.6.2

---

## Recovery Steps (on hang / driver failure)

1. Kill all CUDA processes (`kill -9` on PID group)
2. Reset GPU: `nvidia-smi --gpu-reset -i <gpu_id>` (if supported) or reload driver (`modprobe -r nvidia_uvm; modprobe nvidia_uvm`)
3. Verify recovery: `nvidia-smi -q` returns valid state; `cudaGetDeviceCount()` returns > 0
4. Log event in `measurement_log.json` with timestamp and hang duration
5. Skip measurement for that run (do NOT include in dataset CSV)
