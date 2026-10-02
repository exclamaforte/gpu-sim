#!/usr/bin/env python3
"""gpu-sim-465.1 phase 0 go/no-go: does NCCL dlopen the plugin and accept it?

GO is the bead's own gate: NCCL loads an intentionally minimal plugin and
REPORTS it, with NCCL_DEBUG=INFO naming the plugin and the accepted ABI version.
Anything less is NO-GO.
"""
import os, sys, subprocess, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(HERE, "libncclNetPlugin.so")

env = dict(os.environ)
env["NCCL_NET_PLUGIN"] = PLUGIN
env["NCCL_DEBUG"] = "INFO"
env["NCCL_DEBUG_SUBSYS"] = "INIT,NET,ENV"
env["LD_LIBRARY_PATH"] = os.pathsep.join([
    "/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib",
    env.get("LD_LIBRARY_PATH", "")])

code = """
import torch, os, sys
torch.cuda.init()
d = torch.cuda.current_device()
n = torch.tensor([1.0]*8, device=f'cuda:{d}')
try:
    torch.cuda.set_device(d)
    out = torch.ops.nvidia.nccl.all_reduce if hasattr(torch.ops, 'nvidia') else None
except Exception as e:
    print('NCLOPS-UNAVAILABLE', e)
x = torch.ones(4, device=f'cuda:{d}')
x = x * 2.0
torch.cuda.synchronize()
print('TORCH_OK', x.sum().item())
"""
r = subprocess.run([sys.executable, "-c", code], env=env,
                   capture_output=True, text=True, timeout=300)
out = r.stdout + r.stderr
print(out[-4000:])
verdict = {
    "returncode": r.returncode,
    "plugin_mentioned": "libncclNetPlugin" in out or "ncclNetPlugin" in out,
    "net_subsys_lines": [l for l in out.splitlines()
                         if "NET" in l or "Plugin" in l or "plugin" in l][:40],
    "accepted_version_lines": [l for l in out.splitlines()
                               if "ncclNet_v" in l][:20],
}
print("VERDICT_JSON " + json.dumps(verdict))
