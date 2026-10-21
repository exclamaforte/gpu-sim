# gpu-sim-5zv AC4: Route agreement (model vs plugin)

## Settings tested (model side)
1. No fault (loss_ppm=0, jitter=0): model loss_factor=1.0, exceed=0.0 -> completes
2. Loss (loss_ppm=100000): model loss_factor=1.111 -> retransmit predicted; plugin returns ELOSS (fabric trichotomy, m3d.9)
3. Jitter + tight timeout: model exceed>0 -> timeout predicted; plugin returns ETIMEDOUT (fabric trichotomy, m3d.9)

## Plugin side
The fabric trichotomy (ELOSS/ETIMEDOUT/EPEERDEAD) is enforced and demonstrated
in gpu-sim-m3d.9 (6 negative controls, all RED). The plugin returns ncclSystemError
for all non-OK (NCCL has no distinct enum); the distinction lives in the fabric
code + plugin reason+code line + trace (m3d.9 design.md NCCL error-code note).

## REFUSED
Full two-sided plugin run for all 3 settings on this box: requires multi-rank
real NCCL with NCCL_NET_PLUGIN, which needs the 6kp interposer for device identity
and real computation per rank. The 6kp interposer provides identity only, not
computation (8ii REFUSED on this ground). Settings 2 and 3 are REFUSED as
two-sided measurements; their categorical prediction is model-only.

## Verdict: PARTIAL (model side MET, plugin side REFUSED for settings 2-3)
