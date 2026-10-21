# Canonical byte table (n=2, measured)

Convention: per phase per direction: piece count x piece size.
Total per direction = sum over phases. Total per rank = 2 x per direction? No:
for n=2 ring allreduce each rank sends per direction across 2 phases.

| S (bytes) | floats | isends/rank | distinct bytes | phases (digests) | per-phase pieces | per-phase bytes | total bytes/rank | byte factor |
|-----------|--------|-------------|----------------|------------------|------------------|-----------------|------------------|-------------|
| 4096      | 1024   | 2           | 4096           | 2                | 1                | 4096 (S)        | 8192             | 2.0 (2S)    |
| 65536     | 16384  | 8           | 16384          | 2                | 4                | 65536 (S)       | 131072           | 2.0 (2S)    |
| 262144    | 65536  | 8           | 65536          | 2                | 4                | 262144 (S)      | 524288           | 2.0 (2S)    |
| 524288    | 131072 | 16          | 32768          | 2                | 8                | 262144 (S/2)    | 524288           | 1.0 (S)     |
| 1048576   | 262144 | 16          | 65536          | 2                | 8                | 524288 (S/2)    | 1048576          | 1.0 (S)     |
| 2097152   | 524288 | 16          | 131072         | 2                | 8                | 1048576 (S/2)   | 2097152          | 1.0 (S)     |

Note: the byte factor changes from 2 to 1 between 256 KiB and 512 KiB. At 4 KiB the
factor is 2 (8sy holds). At 512 KiB+ the factor is 1 (only one phase's worth of
bytes per rank direction, or the two phases are fused). The model's 8sy derivation
(2S) is NOT universal; it holds at 4 KiB only.

Raw traces: /tmp/scratchpad/trace_*.log (regenerable).