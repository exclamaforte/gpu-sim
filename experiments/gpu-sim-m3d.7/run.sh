#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
make clean >/dev/null
make check
# compile sweep helper
mkdir -p build
cc -O2 -g -Iinclude -Ishim -o build/sweep_collective tests/sweep_collective.c -L. -lfakegpu -Wl,-rpath,$(pwd)

python3 - <<'PY'
import json, subprocess, os, sys
sys.path.insert(0, "scripts")
from validate_trace import validate as validate_trace
from epic_cluster.spec import validate
from epic_cluster.topology import emit
from epic_cluster.link_model import predict

def make(n):
    nodes=[{"name":"n0","devices":[{"id":i,"model":"m"} for i in range(n)]}]
    models={"m":{"sm_count":80,"clock_mhz":1000,"mem_bytes":1<<30,"mem_bw_gbs":1000.0,"arch":"x","source":"test"}}
    links=[]
    for i in range(n):
        for j in range(i+1,n):
            links.append({"a":{"node":"n0","device":i},"b":{"node":"n0","device":j},"kind":"nvlink","bw_gbps":100.0,"latency_us":5.0,"mtu":4096,"loss_ppm":0,"jitter_us":0.0,"shared_group":None})
    ranks=[{"rank":i,"node":"n0","device":i} for i in range(n)]
    spec={"version":1,"models":models,"nodes":nodes,"links":links,"ranks":ranks}
    validate(spec)
    return spec

# C/Python agreement + monotonic on ring_allreduce
vals=[]
for n in (2,4,8):
    spec=make(n)
    open(f"/tmp/topo_{n}.bin","wb").write(emit(spec))
    p=predict(spec,"ring_allreduce",1024,"float32",n)
    env=dict(os.environ)
    env["LD_PRELOAD"]=os.path.abspath("libfakegpu.so")
    env["FAKEGPU_NUM_DEVICES"]=str(n)
    env["GPUSIM_TOPOLOGY"]=f"/tmp/topo_{n}.bin"
    env["GPUSIM_TRACE"]=f"/tmp/trace_{n}.jsonl"
    if os.path.exists(env["GPUSIM_TRACE"]): os.remove(env["GPUSIM_TRACE"])
    subprocess.run(["./build/ring_allreduce"], env=env, check=True)
    # schema validation
    subprocess.run([sys.executable, "scripts/validate_trace.py", env["GPUSIM_TRACE"]], check=True)
    recs=[json.loads(l) for l in open(env["GPUSIM_TRACE"])]
    c=recs[0]["predicted_s"]
    py=p["duration_s"]
    assert abs(c-py)<1e-15, (n,c,py)
    vals.append(c)
    print(n, c, py)
assert vals[0]<vals[1]<vals[2], vals
print("C/Python agreement + monotonic: PASS")

# Regime sweep: latency -> bandwidth crossover
n=4
spec=make(n)
open("/tmp/topo_sweep.bin","wb").write(emit(spec))
regimes_seen=set()
crossover=None
prev_regime=None
for count in [1, 16, 256, 1024, 4096, 65536, 1000000]:
    env=dict(os.environ)
    env["LD_PRELOAD"]=os.path.abspath("libfakegpu.so")
    env["FAKEGPU_NUM_DEVICES"]=str(n)
    env["GPUSIM_TOPOLOGY"]="/tmp/topo_sweep.bin"
    env["GPUSIM_TRACE"]=f"/tmp/trace_sweep_{count}.jsonl"
    if os.path.exists(env["GPUSIM_TRACE"]): os.remove(env["GPUSIM_TRACE"])
    subprocess.run(["./build/sweep_collective", str(count)], env=env, check=True)
    subprocess.run([sys.executable, "scripts/validate_trace.py", env["GPUSIM_TRACE"]], check=True)
    recs=[json.loads(l) for l in open(env["GPUSIM_TRACE"])]
    r=recs[0]["regime"]
    regimes_seen.add(r)
    p=predict(spec,"ring_allreduce",count,"float32",n)
    assert r == p["regime"], (count, r, p["regime"])
    print(f"count={count} regime={r} (model={p['regime']})")
    if prev_regime == "latency" and r == "bandwidth" and crossover is None:
        crossover = count
    prev_regime = r
assert "latency" in regimes_seen and "bandwidth" in regimes_seen, regimes_seen
print("regime sweep: PASS (latency + bandwidth both present)")

# negative control: corrupted topology fails
bad="/tmp/topo_bad.bin"
open("/tmp/topo_bad.bin","wb").write(b"GARBAGE")
env=dict(os.environ)
env["LD_PRELOAD"]=os.path.abspath("libfakegpu.so")
env["FAKEGPU_NUM_DEVICES"]="2"
env["GPUSIM_TOPOLOGY"]="/tmp/topo_bad.bin"
r=subprocess.run(["./build/ring_allreduce"], env=env)
assert r.returncode!=0, "corrupted topology did not fail"
print("negative control (corrupted topology): PASS (non-zero exit)")
PY