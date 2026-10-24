#!/usr/bin/env bash
# gpu-sim-f50 validation: transport for all_gather / reduce_scatter / send-recv.
# Exits nonzero on any failure. Runs the committed-record default gate FIRST so
# a broken baseline fails fast, then the transport unit probe, then the four
# model gates (TP forward+20-step, PP forward, ZeRO-2 reassembly, ZeRO-3 curve).
#
# Rerun: ts -L "gpu-sim-f50: validate" --max-runtime 60m <this file>
# (split into per-gate jobs if the queue wants shorter slices).
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-f50
mkdir -p experiments/gpu-sim-f50/tmp
export TMPDIR=$PWD/experiments/gpu-sim-f50/tmp
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NCCL_NO_FILL=1
export PYTHONPATH=$PWD
T=$PWD/experiments/gpu-sim-f50/tmp
export GPUSIM_TOPOLOGY=$T/f50-topo.bin

echo "=== 0. default gate: SIMULATE_ONLY unset reproduces 149/86.1 ==="
python3 -m tools.torch_parallel.topology --ranks 4 --fabric nvlink \
    --out $T/f50-topo.bin > /dev/null
FAKEGPU_NUM_DEVICES=4 sh -c 'unset FAKEGPU_NCCL_SIMULATE_ONLY; rm -f /dev/shm/fgcoll_*; \
  python3 -m tools.torch_parallel.sweep --world-sizes 4 --steps 5 --fabric nvlink' \
  > $T/f50-gate.txt 2>&1
tail -2 $T/f50-gate.txt
python3 experiments/gpu-sim-n5h/gate_check.py $T/f50-gate.txt || exit 1

export FAKEGPU_NCCL_SIMULATE_ONLY=0
run2() { # $1=topo ranks $2=script $3=log
  python3 -m tools.torch_parallel.topology --ranks "$1" --fabric nvlink \
      --out $T/f50-topo.bin > /dev/null
  rm -f /dev/shm/fgcoll_*
  FAKEGPU_NUM_DEVICES=$1 GPUSIM_TRACE=$T/"$3" \
  python3 -m torch.distributed.run --nproc_per_node="$1" --standalone "$2" || exit 1
}

echo "=== 1. transport unit probe: AG f32/bf16, RS, P2P both directions ==="
python3 -m tools.torch_parallel.topology --ranks 2 --fabric nvlink \
    --out $T/f50-topo.bin > /dev/null
rm -f /dev/shm/fgcoll_*
FAKEGPU_NUM_DEVICES=2 GPUSIM_TRACE=$T/f50-probe.jsonl \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    experiments/gpu-sim-f50/ag_rs_p2p_probe.py --steps 3 || exit 1

echo "=== 2. TP gate: sharded fwd == reference, 20 SGD steps match + fall ==="
run2 2 experiments/gpu-sim-f50/tp_gate.py f50-tp.jsonl

echo "=== 3. PP gate: pp=2 forward == single, exact ==="
run2 2 experiments/gpu-sim-f50/pp_gate.py f50-pp.jsonl

echo "=== 4. ZeRO-2 gate: reassembled grads/dp == single-process grads ==="
FAKEGPU_NUM_DEVICES=1 GPUSIM_TRACE=$T/f50-z2-w1.jsonl \
python3 -m torch.distributed.run --nproc_per_node=1 --standalone \
    tools/torch_parallel/train.py --steps 3 --warmup 1 --optim sgd --lr 0.01 \
    --seed 7 --save-flat $T/f50-ref-flat.pt --json $T/f50-z2-w1.json || exit 1
FAKEGPU_NUM_DEVICES=2 GPUSIM_TRACE=$T/f50-z2-w2.jsonl \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    tools/torch_parallel/train.py --steps 3 --warmup 1 --optim sgd --lr 0.01 \
    --seed 7 --zero 2 --dp-mean --verify-flat $T/f50-ref-flat.pt \
    --json $T/f50-z2-w2.json || exit 1

echo "=== 5. ZeRO-3 gate: FSDP curve == zero1 curve ==="
FAKEGPU_NUM_DEVICES=2 GPUSIM_TRACE=$T/f50-z3a.jsonl \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    tools/torch_parallel/train.py --steps 6 --warmup 1 --optim sgd --lr 0.01 \
    --seed 7 --dp-mean --json $T/f50-z31.json || exit 1
FAKEGPU_NUM_DEVICES=2 GPUSIM_TRACE=$T/f50-z3b.jsonl \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    tools/torch_parallel/train.py --steps 6 --warmup 1 --optim sgd --lr 0.01 \
    --seed 7 --zero 3 --dp-mean --json $T/f50-z33.json || exit 1
python3 - "$T/f50-z31.json" "$T/f50-z33.json" <<'EOF' || exit 1
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ma = [sum(r['losses'][i] for r in a)/len(a) for i in range(len(a[0]['losses']))]
mb = [sum(r['losses'][i] for r in b)/len(b) for i in range(len(b[0]['losses']))]
worst = max(abs(x-y) for x, y in zip(ma, mb))
assert len(ma) == len(mb)
assert worst < 1e-4*max(1.0, abs(ma[0])), f"zero3 diverges: {worst:.2e}"
print(f"ZERO3 PASS: curves match, worst {worst:.2e}")
EOF
echo "ALL F50 GATES PASS"
