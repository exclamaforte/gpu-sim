#!/usr/bin/env python3
"""Validate a gpu-sim JSONL trace against the m3d.7 schema."""
import json, sys

REQUIRED = ["ts_ns","rank","nranks","op","count","dtype","bytes","algorithm","predicted_s","bottleneck_link","regime"]
OPS = {"allreduce","broadcast","allgather","reducescatter","alltoall","reduce","send","recv",
       "forward_layer","backward_layer","optimizer","bubble","tp_allreduce","dp_allreduce"}
OPS_PREFIX_OK = True
ALGOS = {"ring_allreduce","ring_allgather","ring_reduce_scatter","ring_alltoall","tree_bcast","p2p"}
REGIMES = {"latency","bandwidth"}
DTYPES = {"int8","uint8","float16","bfloat16","int32","uint32","float32","int64","uint64","float64"}
COMPUTE_KINDS = {"compute","comm","idle","hidden"}
COMPUTE_REQUIRED = ["ts_ns","rank","nranks","op","kind","phase"]

def _is_header(obj):
    return obj.get("type") == "header" or "header" in obj

def validate(path):
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if _is_header(obj):
                n += 1
                continue
            # Collective shape: all m3d.7 required present (superset allowed)
            if all(k in obj for k in REQUIRED):
                assert isinstance(obj["ts_ns"], int)
                assert isinstance(obj["rank"], int) and obj["rank"] >= 0
                assert isinstance(obj["nranks"], int) and obj["nranks"] >= 1
                # op may be a collective op or a prefixed compute name; allow any str
                assert isinstance(obj["op"], str)
                assert isinstance(obj["count"], int) and obj["count"] >= 0
                assert obj["dtype"] in DTYPES
                assert isinstance(obj["bytes"], int) and obj["bytes"] >= 0
                assert isinstance(obj["algorithm"], str) and obj["algorithm"]
                ps = obj["predicted_s"]
                assert ps is None or (isinstance(ps, (int,float)) and ps >= 0)
                bl = obj["bottleneck_link"]
                assert bl is None or (isinstance(bl,int) and bl >= 0)
                assert isinstance(obj["regime"], str) and obj["regime"]
                n += 1
                continue
            # Compute/idle shape: additive extension
            if all(k in obj for k in COMPUTE_REQUIRED):
                assert isinstance(obj["ts_ns"], int)
                assert isinstance(obj["rank"], int) and obj["rank"] >= 0
                assert isinstance(obj["nranks"], int) and obj["nranks"] >= 1
                assert isinstance(obj["op"], str)
                assert obj["kind"] in COMPUTE_KINDS
                assert isinstance(obj["phase"], str)
                dur = obj.get("seconds", obj.get("predicted_s", 0))
                assert isinstance(dur, (int,float)) and dur >= 0
                n += 1
                continue
            raise ValueError(f"unrecognized record shape: keys={sorted(obj.keys())}")
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
