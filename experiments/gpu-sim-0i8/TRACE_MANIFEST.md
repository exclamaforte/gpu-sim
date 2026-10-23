# Trace manifest for gpu-sim-0i8

Committed trace artifacts and the bead/claim each supports. Paths are relative to repo root.

## gpu-sim-0i8 (this bead): fabric value path post-fix certification
- experiments/gpu-sim-pkb/traces/run_sd_*_rank0.txt, run_sd_*_rank1.txt, trace_sd_*.log
  - self-describing payload exact on both ranks (n=3 per cell, 6 sizes, Simple pinned)
- experiments/gpu-sim-pkb/traces/run_nu_*_rank0.txt, run_nu_*_rank1.txt, trace_nu_*.log
  - non-uniform fill VALUE OK on both ranks (n=3 per cell, 524288 + 1048576, Simple pinned)
- experiments/gpu-sim-pkb/topo_single_eut.bin
  - single-node topology used by all runs above

## Pre-existing untracked traces committed for provenance (gpu-sim-pkb)
- run_652_* : gpu-sim-652 transport/identity (via NET/sim v12.isend)
- run_7vw_* : gpu-sim-7vw protocol selection (Simple pinned vs default)
- run_bam_* : gpu-sim-bam emulated-fabric-vs-real-CUDA boundary
- run_bl5-* : gpu-sim-bl5 byte-table measurement
- run_c1z_* : gpu-sim-c1z value-correctness withdrawal
- run_chanfix1_*, run_fix393a_*, run_fix524a_*, run_tag524a_*, run_testslotseq_* : fabric slot/seq/tag fixes
- negative_control_* : negative controls for fabric identity
- trace_652_*, trace_7vw not present; trace_selfdesc_small.log : small-size smoke
- (Any trace not supporting a claim is retained as historical evidence; none deleted.)
