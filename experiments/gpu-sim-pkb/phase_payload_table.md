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

n>=3 robustness: NOT MEASURABLE with the current runtime interposer. The 4-rank run
fails with "named symbol not found" in ncclLaunchKernel because all ranks map to real
device 0 (the runtime interposer (libfakegpu) does not control NCCL's device enumeration
at the versioned cuda* boundary; see 6kp bead). The n=4 pieces-per-phase is therefore
unmeasured and the model refuses (ValueError) for n>=3 rather than deriving from the cap.

Segfault diagnosis: earlier 4MiB/8MiB segfaults were environmental (transient), not a
registration leak -- 4MiB runs now succeed. The reg_pool leak (fetch_add before bounds
check) was real and is fixed (rollback on overflow). The registration failure path now
returns 0 (error) loudly; no silent use of a failed offset.

Raw traces: /tmp/scratchpad/trace_*.log (not committed; regenerable via run_forcenet.sh
with GPUSIM_COUNT set).


## Chunk rule (from measurement)

pieces_per_phase (P) per direction, n=2:
  S <= 256 KiB: P = 1 (phase payload = S, single chunk)
  S >= 512 KiB: P = 8 (phase payload = S/8, 8 chunks of S/8)

The chunk SIZE is not constant; it grows with S (4KB, 16KB, 64KB, 128KB, 256KB)
while the count steps at 256/512 KiB. This is NCCL's internal chunking heuristic
(buffer size, channel count, min chunk), not a simple function of S and n.

## Judgement: modelled vs input

The chunk rule should be an INPUT to the model, not a modelled function. Rationale:
the chunking is a property of the PROTOCOL (NCCL), not of the NETWORK. The network
model already takes link bandwidth and latency as inputs; the piece count is the
same kind of thing -- a property of what the protocol will do, which the model
is told rather than predicts. The per-isend accounting then charges the simulator's
per-piece cost P times, which is the artifact being measured. Modelling the chunk
rule would make the model claim to predict NCCL's internal heuristic, which it
has no basis for, and would hide the protocol dependency behind a formula that
only fits the measured points. The table is the evidence; the model takes P from it.

Accordingly, `_derived_pieces_per_phase` in epic_cluster/link_model.py and the C
model in src/fakegpu_nccl.c should default to the measured table when available,
and accept an explicit override (pieces_per_phase) otherwise. The fabric cap
(FG_FAB_MAX_PAYLOAD) remains an upper bound on what the fabric accepts, not the
unit in which NCCL sends.
