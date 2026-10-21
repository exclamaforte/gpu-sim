# gpu-sim-5zv AC4: Route agreement (model vs plugin)

## Settings tested
1. No fault (loss_ppm=0, jitter=0): MET (two-sided).
   Model: loss_factor=1.0, exceed=0.0 -> completes.
   Plugin: experiments/gpu-sim-6kp/net_plugin_over_interposed.log (fault-free run,
   "NET/Plugin: Loaded net plugin sim", "via NET/sim", no P2P/SHM/Socket).
   Categorical agreement: completes on both sides.

2. Loss (loss_ppm=100000): PREDICTED-FROM-FABRIC (model side only).
   Model: loss_factor=1.111 -> retransmit predicted.
   Plugin: would return ELOSS (fabric trichotomy, m3d.9) -- PREDICTED, not observed.

3. Jitter + tight timeout: PREDICTED-FROM-FABRIC (model side only).
   Model: exceed>0 -> timeout predicted.
   Plugin: would return ETIMEDOUT (fabric trichotomy, m3d.9) -- PREDICTED, not observed.

## REFUSED (two-sided plugin runs)
nvidia-smi lists exactly ONE device (NVIDIA GeForce RTX 5090, index 0; see
nvidia_smi_query.txt). Two ranks cannot both run real collectives on one device,
so no two-sided NCCL fault measurement exists on this box at any rank count.
8ii hit the same limit at n>=4; here it applies already at n=2.

## Verdict: PARTIAL (setting 1 MET two-sided; settings 2-3 REFUSED two-sided, PREDICTED-FROM-FABRIC)
