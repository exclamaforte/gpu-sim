#!/usr/bin/env python3
"""A1's first measured line: does torch see 8 B300s, and does real arithmetic run?

Runs the questions in a CHILD process (so the shim is loaded only there, and the
parent's RSS cannot contaminate the number), with LD_LIBRARY_PATH pointing at the
directory holding our libcuda.so.1. Prints one JSON object; exit code 0 iff every
assertion the bead's A1 names holds.

    python3 experiments/gpu-sim-7q3w.4/probe_vdev.py [--n 8] [--json]

What "aliasing" means here, and why this is not a reporting shim: the device count,
the names and the memory identity come from the shim (modelled), while the matmul
below really runs on the one real GPU through the real driver -- so a wrong answer
comes back wrong, and torch's own kernel launches are the ones executing.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "gpu-sim-7q3w.4"
SHIM = EXP / "shim"
BUILD = REPO / ".lab" / "tmp" / "vdev-build"

CHILD = r'''
import ctypes, json, os, resource, sys
import torch
out = {}
out["torch"] = torch.__version__
out["device_count"] = torch.cuda.device_count()
out["cudart_count"] = torch._C._cuda_getDeviceCount()
names, total, cc = [], [], []
for i in range(out["device_count"]):
    names.append(torch.cuda.get_device_name(i))
    p = torch.cuda.get_device_properties(i)
    total.append(int(p.total_memory))
    cc.append(f"{p.major}.{p.minor}")
out["names"] = names
out["total_memory_gib"] = [round(t / 2**30, 1) for t in total]
out["compute_capability"] = cc
out["is_available"] = torch.cuda.is_available()
# A1's other half: NVML's view of the same device, in the same invocation.
nv = ctypes.CDLL("libnvidia-ml.so.1")
nv.nvmlInit_v2.restype = ctypes.c_int
nv.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
nv.nvmlDeviceGetCount_v2.restype = ctypes.c_int
nv.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
nv.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
nv.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
nv.nvmlDeviceGetName.restype = ctypes.c_int
nv.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
nv.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
out["nvml_init_rc"] = int(nv.nvmlInit_v2())
cnt = ctypes.c_uint(0)
out["nvml_count_rc"] = int(nv.nvmlDeviceGetCount_v2(ctypes.byref(cnt)))
out["nvml_count"] = int(cnt.value)
dev = ctypes.c_void_p()
rc = nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev))
out["nvml_handle_rc"] = int(rc)
buf = ctypes.create_string_buffer(96)
rc = nv.nvmlDeviceGetName(dev, buf, 96)
out["nvml_name_rc"] = int(rc)
out["nvml_name"] = buf.value.decode() if rc == 0 else None
mem = (ctypes.c_ulonglong * 3)()
rc = nv.nvmlDeviceGetMemoryInfo(dev, mem)
out["nvml_mem_rc"] = int(rc)
out["nvml_total_gib"] = round(mem[0] / 2**30, 1) if rc == 0 else None
out["nvml_free_gib"] = round(mem[1] / 2**30, 1) if rc == 0 else None
# arithmetic: really executed by the real GPU behind the alias
torch.manual_seed(0)
a = torch.randn(512, 512, device="cuda:0")
b = a @ a
ref = a.cpu() @ a.cpu()
out["matmul_max_abs_err"] = float((b.cpu() - ref).abs().max())
out["fake_device"] = b.device.type + ":" + str(b.device.index)
out["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
with open("/proc/meminfo") as f:
    for line in f:
        if line.startswith("MemAvailable:"):
            out["mem_available_mib"] = int(line.split()[1]) // 1024
print("VDEV " + json.dumps(out))
'''


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    inc = Path.home() / ".local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
    # D1 is the only source of identity; every shim includes the generated header.
    subprocess.run([sys.executable, str(SHIM / "gen_spec.py"), str(BUILD)], check=True)
    subprocess.run([sys.executable, str(SHIM / "gen_forwarders.py"), str(BUILD)], check=True)
    subprocess.run([sys.executable, str(SHIM / "gen_nvml_forwarders.py"), str(BUILD)], check=True)
    # DT_SONAME deliberately NOT libcuda.so.1: our file is reached through a
    # symlink of that name, and if its soname were the same, glibc's dlopen of the
    # real driver would match the already-loaded map by soname and hand back OURSELF
    # (measured: infinite recursion inside cuInit, gdb bt 12 frames of cuInit).
    subprocess.run(["gcc", "-shared", "-fPIC", "-O1", "-I" + str(inc),
                    "-I" + str(BUILD), "-Wl,-soname,libcuda-vdev.so.1",
                    "-o", str(BUILD / "libcuda-vdev.so.1"),
                    str(SHIM / "vdev.c"), str(BUILD / "forwarders.c"),
                    str(BUILD / "forwarders.S"), "-ldl"], check=True)
    subprocess.run(["gcc", "-shared", "-fPIC", "-O1",
                    "-DNVML_NO_UNVERSIONED_FUNC_DEFS", "-I" + str(inc),
                    "-I" + str(BUILD), "-Wl,-soname,libnvidia-ml-vdev.so.1",
                    "-o", str(BUILD / "libnvidia-ml-vdev.so.1"),
                    str(SHIM / "nvml.c"), str(BUILD / "nvml_forwarders.c"),
                    str(BUILD / "nvml_forwarders.S"), "-ldl"], check=True)
    subprocess.run(["gcc", "-shared", "-fPIC", "-O1", "-I" + str(inc),
                    "-I" + str(BUILD), "-Wl,-soname,libcudart-vdev.so.1",
                    "-o", str(BUILD / "libcudart-vdev.so.1"),
                    str(SHIM / "cudart.c"), "-ldl"], check=True)
    for name, target in (("libcuda.so.1", "libcuda-vdev.so.1"),
                         ("libnvidia-ml.so.1", "libnvidia-ml-vdev.so.1")):
        link = BUILD / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    return BUILD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    build()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(BUILD) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["LD_PRELOAD"] = str(BUILD / "libcudart-vdev.so.1")
    env["VDEV_NVDEV"] = str(args.n)
    with tempfile.TemporaryDirectory(dir=str(REPO / ".lab" / "tmp")) as td:
        cp = subprocess.run([sys.executable, "-c", CHILD], env=env, cwd=td,
                            capture_output=True, text=True, timeout=300)
    line = [l for l in cp.stdout.splitlines() if l.startswith("VDEV ")]
    if not line:
        print(cp.stdout[-2000:], cp.stderr[-2000:])
        print(f"FAIL: the child printed no VDEV line (rc={cp.returncode})")
        return 1
    got = json.loads(line[0][5:])
    checks = {
        "device_count_is_N": got["device_count"] == args.n,
        "cudart_count_is_N": got["cudart_count"] == args.n,
        "nvml_count_is_N": got["nvml_count"] == args.n,
        "every_device_named_B300": all("B300" in n for n in got["names"]),
        "nvml_name_matches_cudart": got["nvml_name"] == got["names"][0],
        "b300_memory_identity": all(t > 200 for t in got["total_memory_gib"]),
        "nvml_total_matches": got["nvml_total_gib"] == got["total_memory_gib"][0],
        "capability_is_real_12_0": all(c == "12.0" for c in got["compute_capability"]),
        "cuda_available": got["is_available"] is True,
        "matmul_ran_on_the_alias": got["matmul_max_abs_err"] < 1e-3,
    }
    print(json.dumps({"measured": got, "checks": checks}, indent=2))
    if not all(checks.values()):
        print("FAIL: " + ", ".join(k for k, v in checks.items() if not v))
        return 1
    print(f"VDEV OK: device_count={got['device_count']} name={got['names'][0]} "
          f"total={got['total_memory_gib'][0]} GiB peak_rss={got['peak_rss_mib']} MiB "
          f"mem_available={got.get('mem_available_mib')} MiB "
          f"matmul_err={got['matmul_max_abs_err']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
