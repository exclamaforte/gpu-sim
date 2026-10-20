#!/usr/bin/env python3
"""Validate a gpu-sim JSONL trace against the m3d.7 schema."""
import json, sys

REQUIRED = ["ts_ns","rank","nranks","op","count","dtype","bytes","algorithm","predicted_s","bottleneck_link","regime"]
OPS = {"allreduce","broadcast","allgather","reducescatter","alltoall","reduce","send","recv"}
ALGOS = {"ring_allreduce","ring_allgather","ring_reduce_scatter","ring_alltoall","tree_bcast"}
REGIMES = {"latency","bandwidth"}
DTYPES = {"int8","uint8","float16","int32","uint32","float32","int64","uint64","float64"}

def validate(path):
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for k in REQUIRED:
                if k not in obj:
                    raise ValueError(f"missing field {k}")
            if set(obj.keys()) != set(REQUIRED):
                raise ValueError(f"extra/missing keys: {set(obj.keys()) ^ set(REQUIRED)}")
            assert isinstance(obj["ts_ns"], int)
            assert isinstance(obj["rank"], int) and obj["rank"] >= 0
            assert isinstance(obj["nranks"], int) and obj["nranks"] >= 1
            assert obj["op"] in OPS
            assert isinstance(obj["count"], int) and obj["count"] >= 0
            assert obj["dtype"] in DTYPES
            assert isinstance(obj["bytes"], int) and obj["bytes"] >= 0
            assert obj["algorithm"] in ALGOS
            assert isinstance(obj["predicted_s"], (int,float)) and obj["predicted_s"] >= 0
            bl = obj["bottleneck_link"]
            assert bl is None or (isinstance(bl,int) and bl >= 0)
            assert obj["regime"] in REGIMES
            n += 1
    if n == 0:
        raise ValueError("empty trace")
    print(f"validate_trace: {n} records OK")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: validate_trace.py <trace.jsonl>", file=sys.stderr)
        sys.exit(2)
    try:
        validate(sys.argv[1])
    except Exception as e:
        print(f"INVALID: {e}", file=sys.stderr)
        sys.exit(1)
