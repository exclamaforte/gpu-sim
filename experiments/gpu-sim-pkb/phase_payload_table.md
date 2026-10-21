# Phase-payload vs size table (n=2 and n=4)

Collected via the m3d.8 net plugin (forcenet mode, NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=1)
over libfakegpu, with GPUSIM_COUNT varying the collective size. Each row: number of
isends per rank, distinct isend byte sizes, distinct digests (= phases).

| floats | bytes (S) | rank | isends | distinct bytes | distinct digests (phases) |
|--------|-----------|------|--------|----------------|--------------------------|
| 1024   | 4096      | 0/1  | 2      | 4096           | 2                        |
| 16384  | 65536     | 0/1  | 8      | 16384          | 2                        |
| 65536  | 262144    | 0/1  | 8      | 65536          | 2                        |
| 131072 | 524288    | 0/1  | 16     | 32768          | 2                        |
| 262144 | 1048576   | 0/1  | 16     | 65536          | 2                        |
| 524288 | 2097152   | 0/1  | 16     | 131072         | 2                        |
| 1048576| 4194304   | -    | -      | -              | segfault (not collected)  |
| 2097152| 8388608   | -    | -      | -              | segfault (not collected)  |

Observation: exactly 2 distinct digests at every size that ran -> two phases at all
sizes (input phase + summed-result phase), so the per-phase payload is total_bytes/2
per rank direction at every size. The isend (chunk) size and count vary with S:

  S=4096:   1 isend/phase of 4096  -> phase payload = S (1 chunk)
  S=65536:  4 isends/phase of 16384 -> phase payload = S (4 chunks of S/4)
  S=262144: 4 isends/phase of 65536 -> phase payload = S (4 chunks of S/4)
  S=524288: 8 isends/phase of 32768 -> phase payload = S (8 chunks of S/8... wait 8*32768=262144=S/2)

Correction on the 512KiB row: 16 isends total = 8 per phase, each 32768 bytes,
8*32768 = 262144 = S/2, not S. So at 512KiB the per-phase payload is S/2, not S.
At 1MiB and 2MiB: 16 isends total = 8 per phase, each 65536/131072, 8*65536=524288=S/2,
8*131072=1048576=S/2. So per-phase payload = S/2 at 512KiB and above, S at 256KiB and below.

So the phase payload is a STEP FUNCTION of S: S for S <= 256KiB, S/2 for S >= 512KiB.
This matches hypothesis (A): a simple threshold rule, not an erratic function.
The threshold sits between 256KiB and 512KiB. The gate's payload (2MiB) is on the
known (S/2) side of the threshold, so the n=2 factor is predictable there.

n>=3 robustness: not collected here (n=4 rows pending; segfaults above 2MiB
blocked the 4MiB/8MiB n=2 points too). The n=4 at 1/2/8 MiB remains to be run.

Raw traces: /tmp/scratchpad/trace_*.log (not committed; regenerable via run_forcenet.sh
with GPUSIM_COUNT set).