#!/usr/bin/env bash
# gpu-sim-n5h validation: data-correct all_reduce. GPU job body.
#  0. DEFAULT GATE: SIMULATE_ONLY unset -> world-4 replay reproduces the
#     committed 8y7.3 record (149 coll / 86.1 MB) byte-exact.
#  1. EXACT SUM: f32 exact + bf16 tol vs a HOST-side reference (both ranks).
#  2. PEER DEATH: rank 1 dies after init; rank 0 must dump (op,k,rank,
#     nranks,arrived=1) and SIGABRT -- never hang.
# Exits nonzero on any failure. Step 2 runs ranks as bare processes because
# torchrun tears down survivors when a peer exits.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-n5h
mkdir -p experiments/gpu-sim-n5h/tmp
export TMPDIR=$PWD/experiments/gpu-sim-n5h/tmp
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NCCL_NO_FILL=1
export GPUSIM_TOPOLOGY=$PWD/experiments/gpu-sim-n5h/tmp/n5h-topo.bin
python3 -m tools.torch_parallel.topology --ranks 4 --fabric nvlink \
    --out $PWD/experiments/gpu-sim-n5h/tmp/n5h-topo.bin > /dev/null

echo "--- 0. default gate: committed record reproduced ---"
export FAKEGPU_NUM_DEVICES=4
unset FAKEGPU_NCCL_SIMULATE_ONLY
python3 -m tools.torch_parallel.sweep --world-sizes 4 --steps 5 \
    --fabric nvlink > $PWD/experiments/gpu-sim-n5h/tmp/n5h-gate.txt 2>&1
tail -2 $PWD/experiments/gpu-sim-n5h/tmp/n5h-gate.txt
python3 experiments/gpu-sim-n5h/gate_check.py \
    $PWD/experiments/gpu-sim-n5h/tmp/n5h-gate.txt || exit 1

python3 -m tools.torch_parallel.topology --ranks 2 --fabric nvlink \
    --out $PWD/experiments/gpu-sim-n5h/tmp/n5h-topo.bin > /dev/null
export FAKEGPU_NUM_DEVICES=2 FAKEGPU_NCCL_SIMULATE_ONLY=0
echo "--- 1. known-tensor exact sum ---"
GPUSIM_TRACE=$PWD/experiments/gpu-sim-n5h/tmp/n5h-known.jsonl \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    experiments/gpu-sim-n5h/ar_probe.py
ar_rc=$?
echo "ar_probe rc=$ar_rc"
test $ar_rc -eq 0 || exit 1

echo "--- 2. peer-death barrier timeout ---"
export FAKEGPU_COLL_TIMEOUT_S=8
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29731 WORLD_SIZE=2
LOG=$PWD/experiments/gpu-sim-n5h/tmp/n5h-kill.log
rm -f "$LOG"
RANK=1 python3 experiments/gpu-sim-n5h/kill_test.py >>"$LOG" 2>&1 &
P1=$!
sleep 2
RANK=0 python3 experiments/gpu-sim-n5h/kill_test.py >>"$LOG" 2>&1 &
P0=$!
wait $P0
rc=$?
kill -9 $P1 2>/dev/null
echo "rank0 rc=$rc (want 134 = SIGABRT)"
grep -m1 "barrier timeout" "$LOG" || { echo "NO TIMEOUT DUMP"; exit 1; }
grep -m1 "arrived=1" "$LOG" || { echo "NO ARRIVAL COUNT"; exit 1; }
test $rc -eq 134 || { echo "WRONG EXIT (want SIGABRT)"; exit 1; }
echo "ALL VALIDATIONS PASS"
