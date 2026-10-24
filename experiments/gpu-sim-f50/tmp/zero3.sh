#!/usr/bin/env bash
# f50 ZeRO-3 gate: FSDP loss curve == zero1 curve (FSDP owns its grads).
# Plus the mandatory default-gate replay (world4 DP unchanged).
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-f50
mkdir -p experiments/gpu-sim-f50/tmp
export TMPDIR=$PWD/experiments/gpu-sim-f50/tmp
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NCCL_NO_FILL=1
export FAKEGPU_NCCL_SIMULATE_ONLY=0
export GPUSIM_TOPOLOGY=$PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin
python3 -m tools.torch_parallel.topology --ranks 4 --fabric nvlink \
    --out $PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin > /dev/null
T=$PWD/experiments/gpu-sim-f50/tmp
rm -f /dev/shm/fgcoll_* $T/f50-z31.json $T/f50-z33.json
echo "--- world2 zero1 baseline (6 steps) ---"
FAKEGPU_NUM_DEVICES=2 \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    tools/torch_parallel/train.py --steps 6 --warmup 1 --optim sgd --lr 0.01 \
    --seed 7 --dp-mean --json $T/f50-z31.json
echo "--- world2 zero3 FSDP (6 steps) ---"
FAKEGPU_NUM_DEVICES=2 \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    tools/torch_parallel/train.py --steps 6 --warmup 1 --optim sgd --lr 0.01 \
    --seed 7 --zero 3 --dp-mean --json $T/f50-z33.json
python3 - <<'EOF'
import json
a=[json.loads(l) for l in open('/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-f50/experiments/gpu-sim-f50/tmp/f50-z31.json')]
b=[json.loads(l) for l in open('/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-f50/experiments/gpu-sim-f50/tmp/f50-z33.json')]
la=sum(r['losses'][i] for r in a for i in [0]) / len(a)
print('zero1 rows:', [(r['rank'], [round(v,4) for v in r['losses']]) for r in a])
print('zero3 rows:', [(r['rank'], [round(v,4) for v in r['losses']]) for r in b])
ma=[sum(r['losses'][i] for r in a)/len(a) for i in range(len(a[0]['losses']))]
mb=[sum(r['losses'][i] for r in b)/len(b) for i in range(len(b[0]['losses']))]
worst=max(abs(x-y) for x,y in zip(ma,mb))
assert len(ma)==len(mb), "length"
assert worst < 1e-4*max(1.0,abs(ma[0])), f"zero3 diverges: {worst:.2e}"
print(f"ZERO3 PASS: curves match, worst {worst:.2e}")
EOF
