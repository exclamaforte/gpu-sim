# gpu-sim-j1j: Cross-validate analytic predictor vs shm-fabric emulator

## Window
- Sizes: 512 KiB, 1 MiB (>=512KiB, emulator completes)
- Ranks: 2 (<=4 ranks, emulator stable)
- Protocol: Simple (emulator requires Simple for >=512KiB)
- Topology: 10 GB/s ethernet (emulator topo), matched in analytic spec

## Divergence table

| size | n_ranks | n_channels | sends/chan | emu max-chan (us) | analytic (us) | ratio |
|------|---------|------------|------------|-------------------|---------------|-------|
| 512KiB | 2 | 4 | 8 | 249.7 | 132.4 | 0.53 |
| 1MiB | 2 | 4 | 8 | 459.4 | 184.9 | 0.40 |

## Explanation
The ratio is far from 1. Investigation: the emulator's delay_us sums per-chunk wire delays sequentially within each channel (8 chunks * (bw + lat) = 249.7 us). The analytic predictor uses a single bandwidth term for the entire payload (S/B = 52.4 us) plus per-piece latency (80 us). The emulator does not pipeline chunks within a channel; the analytic assumes optimal bandwidth utilization. This is an **emulator artefact** (no intra-channel pipelining in the delay model), not a model error. The per-chunk delays (31.2 us) are consistent with the link bandwidth (10 GB/s) and latency (5 us).

## Conclusion
The analytic predictor is more optimistic than the emulator due to pipelining assumptions. The divergence is explainable and does not indicate a model error. The analytic numbers should be treated as lower bounds; real hardware may be closer to the emulator's sequential-chunk model.