# 405B@80 optimizer-style rows (gpu-sim-scup v2, post-a5ro params)

Styles priced from the rig table (measure_opt_styles.py + fused-peak runs):
bf16_moments 6.0 steady / 10.0 peak; fp32_moments 12.0 steady / 20.0 default-peak / 16.0 fused-peak;
fp32_master flat legacy 14 (default, every published number). Floors use actual P=405.849e9.
Source: scripts/scaleout_answers.py --opt-state/--opt-peak, 405B, 10x8, batch=1.

| style | peak mode | CP convention | best config | step (s) | margin | mem/GPU |
|---|---|---|---|---|---|---|
| bf16_moments | default | ring all_to_all | tp=1 pp=40 cp=1 dp=2 zero=2 | 0.504 | 3.52% | 75.1 |
| bf16_moments | default | default all_gather | tp=1 pp=40 cp=1 dp=2 zero=2 | 0.504 | 3.52% | 75.1 |
| fp32_moments | none (steady, optimistic) | ring all_to_all | tp=1 pp=40 cp=1 dp=2 zero=3 | 0.504 | 3.52% | 75.1 |
| fp32_moments | none (steady, optimistic) | default all_gather | tp=1 pp=40 cp=1 dp=2 zero=3 | 0.504 | 3.52% | 75.1 |
| fp32_moments | fused | ring all_to_all | INFEASIBLE | - | - | - |
| fp32_moments | fused | default all_gather | INFEASIBLE | - | - | - |
| fp32_moments | default | ring all_to_all | INFEASIBLE | - | - | - |
| fp32_moments | default | default all_gather | INFEASIBLE | - | - | - |
