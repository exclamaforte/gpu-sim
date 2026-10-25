# gpu-sim-4x8: hand-worked allgather example (dp=4)

Per-rank send count: c = 1024 elements, dtype float32 (4 bytes).
nranks = 4.

Per-rank send bytes: c * 4 = 4096 bytes.
Total payload: c * nr * 4 = 16384 bytes.
Per-rank traffic: (n-1) * c = 3 * 4096 = 12288 bytes.
Total traffic: n * (n-1) * c = 4 * 3 * 4096 = 49152 bytes.

Trace fields:
- count (per-rank send count) = 1024
- bytes (total) = 16384
- predicted_s uses total (16384 bytes) with ring_allgather factor (n-1)/n = 3/4.

link_model: S = count_total * dtype = 16384. bw_term = (n-1)/n * S / B.
fg_nccl_predict receives 16384 (total), matches link_model convention.
