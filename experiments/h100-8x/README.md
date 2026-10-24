# 8xH100 declared: a training job as a Perfetto trace

Open `trace.json` at https://ui.perfetto.dev (or `chrome://tracing`). Eight rank
rows, one per simulated device; every collective is a slice whose width is its
modelled duration. Click a slice to see bytes, dtype, count, algorithm and regime.

## Reproduce

```bash
make libfakenccl.so
python3 -m tools.torch_parallel.topology --ranks 8 --fabric nvlink \
        --device h100_sxm --out /tmp/topo_h100x8.bin
GPUSIM_TOPOLOGY=/tmp/topo_h100x8.bin GPUSIM_TRACE=/tmp/h8.jsonl \
FAKEGPU_NCCL_NO_FILL=1 LD_PRELOAD=$PWD/libfakenccl.so \
GPUSIM_ALLOW_EXTRAPOLATE=1 \
torchrun --nproc_per_node=8 --standalone tools/torch_parallel/train.py \
        --steps 6 --dp-mean --json summary.jsonl
python3 -m tools.torch_parallel.perfetto --trace /tmp/h8.jsonl \
        --out trace.json --label "8xH100 declared: DP8" --summary summary.jsonl
```

## What this run is

- world=8, DP8 (tp=1, pp=1, zero=1), 5.2M-parameter model, 6 steps, 2 warmup.
- Collectives: 1784 allreduce + 1776 allgather (3556 traced), 445 per rank.
- Trace span 1782 ms; loss 8.497 -> 8.028 (real optimisation: `--dp-mean` with the
  /dev/shm transport active).
- Fabric: DECLARED 8x H100 SXM over NVLink (132 SMs, 3.35 TB/s, 1 us link latency).

## What this run is NOT

Compute ran on ONE consumer GPU, time-sliced across eight processes. The eight rows
are the parallelism model's rows, not eight physical devices; the collective
durations come from the declared fabric, never from real H100s. The trace metadata
carries this caveat so it cannot be separated from the picture.

Absolute durations are therefore synthetic. The SHAPE is the real product: which
rank does what, in what order, at what message size, and how much of the step is
communication.
