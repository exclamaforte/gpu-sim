# 8xH100 declared job — full Perfetto trace (kernels + Python + collectives)

`trace-full.json` (53 MB, not committed — regenerate with the commands below) opens
at https://ui.perfetto.dev. Each of the 8 ranks gets THREE rows:

| row | source | what it is |
|---|---|---|
| `rank N / python` | torch.profiler `python_function` | 113k Python-level events |
| `rank N / kernels` | torch.profiler `kernel`, `cpu_op`, `cuda_runtime` | 6.3k real CUDA kernels + aten ops |
| `rank N / nccl (modelled)` | libfakenccl JSONL | 1,784 simulated collectives |

Plus `forward` / `backward` / `optimizer_step` scopes and torch's own
`nccl:all_reduce` annotations, so a collective can be read against the Python call
that issued it.

## Two things that had to be measured, not assumed

**Clock alignment.** torch.profiler stamps with its own base (observed 1.97e12 us)
while the interposer stamps CLOCK_REALTIME (epoch ns). Aligning after the fact means
guessing, and a wrong offset slides every collective off the kernel that issued it
while still looking plausible. train.py therefore emits an `fg_clock_anchor` marker
and stamps the wall clock AT THE SAME INSTANT the marker is recorded — not before
`prof.__enter__()`, because CUPTI init costs seconds (measured ~7 s) and an anchor
taken earlier put every collective 7 s off.

**Start times.** Even with the anchor, collectives sat a median 404 us from torch's
own `nccl:*` annotation — about 40% of a 948 us call. The merge now pairs each
modelled collective, IN ORDER, with torch's annotation of the same call (exact,
because the sequence is deterministic) and keeps only the modelled DURATION.
Median offset: 0.00 us.

## Honest limits

- **No per-frame Python stack traces.** `with_stack=True` was set, but this torch
  build's `export_chrome_trace` emits no `stack` field — verified by scanning the
  raw trace. The python row shows python_function events and the explicit phase
  scopes instead. Do not claim stack traces until a build that emits them is used.
- **The nccl row is a model.** Kernels and Python are real, measured on ONE physical
  GPU time-sliced across 8 processes. Collective widths come from a declared 8x
  H100 NVLink fabric and never touched hardware. That is why they are a separate
  row, not merged into the kernel track.

## Reproduce

```bash
make libfakenccl.so
python3 -m tools.torch_parallel.topology --ranks 8 --fabric nvlink \
        --device h100_sxm --out /tmp/topo_h100x8.bin
GPUSIM_TOPOLOGY=/tmp/topo_h100x8.bin GPUSIM_TRACE=/tmp/h8.jsonl \
FAKEGPU_NCCL_NO_FILL=1 GPUSIM_ALLOW_EXTRAPOLATE=1 LD_PRELOAD=$PWD/libfakenccl.so \
torchrun --nproc_per_node=8 --standalone tools/torch_parallel/train.py \
        --steps 3 --dp-mean --torch-profile /tmp/p --json summary.jsonl
python3 -m tools.torch_parallel.merge_profile \
        --torch-traces "/tmp/p.rank*.json" --trace /tmp/h8.jsonl \
        --summary summary.jsonl --out trace-full.json --label "8xH100 declared: DP8"
```
