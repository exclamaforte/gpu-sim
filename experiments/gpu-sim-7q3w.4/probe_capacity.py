#!/usr/bin/env python3
"""gpu-sim-7q3w.4, first measurement: the two risks that kill the design late.

The bead says, in its own words, "measure early, not discover late":

  (a) 8 processes x torch RSS against the free-RAM budget;
  (c) the CUDA surface torch actually needs from a fake driver.

So this probe answers exactly those two, with numbers, before any model or
launcher is built. It imports nothing heavy itself -- every torch measurement runs
in a CHILD, so the parent's own RSS cannot contaminate the figure.

  python3 probe_capacity.py            # both measurements
  python3 probe_capacity.py --json     # machine-readable

1. CAPACITY. Current MemAvailable (not MemTotal: 20+ GiB are already held by other
   agents on this box), plus the measured RSS of a child that imports torch, and of
   a child that imports torch AND holds a ~12M-parameter float32 model with an
   optimizer after one CPU step. The 8-node design needs 8 such children; the bead
   budgets them at ~0.7 GiB each, which this measures rather than assumes.

2. DRIVER-API GAP. torch's CUDA library is dlopened against libcuda.so.1. Our
   libfakegpu.so was built for NCCL's device queries, not for torch's driver use,
   so the question is not "does it work" but "how many symbols are missing, and
   which one does torch hit first". Reported: undefined cu*/cuda* imports of
   libtorch_cuda.so, how many libfakegpu exports, the first missing few, and the
   literal failure torch produces when the shim is put in front of it.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _main_tree():
    """The MAIN checkout, not this worktree: libfakegpu.so is a build product and
    a worktree has none (the same lookup 7q3w.3's harness uses)."""
    try:
        common = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--path-format=absolute",
             "--git-common-dir"], capture_output=True, text=True,
            check=True).stdout.strip()
        return os.path.dirname(common)
    except Exception:
        return REPO


MAIN = _main_tree()
FAKEGPU = os.path.join(MAIN, "libfakegpu.so")
PY = sys.executable


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            m = re.match(r"(\w+):\s+(\d+) kB", line)
            if m:
                out[m.group(1)] = int(m.group(2)) // 1024  # MiB
    return out


def rss_mib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def run_child(code, env=None, timeout=300):
    """Run python code in a child; return (rc, rss_mib at exit, stderr tail)."""
    p = subprocess.Popen([PY, "-c", code], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=env)
    peak = 0
    while p.poll() is None:
        r = rss_mib(p.pid)
        if r:
            peak = max(peak, r)
        try:
            p.wait(timeout=0.05)
        except subprocess.TimeoutExpired:
            pass
    out, err = p.communicate(timeout=timeout)
    return (p.returncode, peak,
            (err or "").strip().splitlines()[-4:],
            (out or "").strip().splitlines()[-4:])


MODEL_CHILD = """
import torch, torch.nn as nn
torch.manual_seed(0)
# ~12M params: a 4-layer, d_model=256 transformer block stack, float32. This is
# the shape the bead asks for (10-50M), and CPU-only on purpose: the point here is
# memory, not speed.
m = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024,
                               batch_first=True), num_layers=4)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randn(8, 64, 256)
y = m(x).sum()
y.backward()
opt.step()
print("params", sum(p.numel() for p in m.parameters()))
"""


def measure_capacity():
    before = meminfo()
    rc1, rss_import, err1, out1 = run_child("import torch; print(torch.__version__)")
    rc2, rss_model, err2, out2 = run_child(MODEL_CHILD)
    after = meminfo()
    return {
        "mem_total_mib": before.get("MemTotal"),
        "mem_available_mib_before": before.get("MemAvailable"),
        "mem_available_mib_after": after.get("MemAvailable"),
        "child_import_torch": {"rc": rc1, "peak_rss_mib": rss_import,
                               "stdout": out1, "stderr": err1},
        "child_import_torch_12m_model_1step": {"rc": rc2, "peak_rss_mib": rss_model,
                                              "stdout": out2, "stderr": err2},
        "budget": {
            "design_nodes": 8,
            "needed_mib_import_only": (rss_import or 0) * 8,
            "needed_mib_with_model": (rss_model or 0) * 8,
            "fits_now": (rss_model or 0) * 8 <= before.get("MemAvailable", 0),
        },
    }


FORK_CHILD = """
import os, sys, torch, torch.nn as nn
torch.manual_seed(0)
m = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024,
                               batch_first=True), num_layers=4)
x = torch.randn(8, 64, 256)
pids = []
for _ in range(8):
    pid = os.fork()
    if pid == 0:
        y = m(x).sum(); y.backward()          # canary: each child must really run
        os._exit(0 if y.item() == y.item() else 1)
    pids.append(pid)
rcs = []
for pid in pids:
    _, st = os.waitpid(pid, 0)
    rcs.append(os.waitstatus_to_exitcode(st))
print("fork8_rss_kb", open("/proc/self/status").read().split("VmRSS:")[1].split()[0])
print("fork_rcs", rcs)
"""


def measure_fork_cow():
    """8 ranks as FORKED children of ONE warmed parent: pages are shared
    copy-on-write, so the marginal cost is not 8 x import.

    MEASURED DEAD END (2026-11-22): under this box's Python 3.14 + torch 2.13 the
    children never return. CPython says so before it starts -- 'This process is
    multi-threaded, use of fork() may lead to deadlocks in the child' -- and the
    warning is right: the parent has torch's thread pools up, the forked children
    deadlock before reaching backward(). Kept behind --with-fork so the finding is
    reproducible rather than folklore; the 8-process design must pay 8 imports.
    """
    rc, peak, err, out = run_child(FORK_CHILD)
    return {"rc": rc, "peak_rss_mib": peak, "stdout": out, "stderr": err}


def undefined_symbols(path, pattern):
    try:
        out = subprocess.run(["nm", "-D", "--undefined-only", path],
                             capture_output=True, text=True, timeout=300).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return sorted({line.split()[-1] for line in out.splitlines()
                   if re.match(pattern, line.split()[-1])})


def defined_symbols(path):
    try:
        out = subprocess.run(["nm", "-D", "--defined-only", path],
                             capture_output=True, text=True, timeout=300).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {line.split()[-1] for line in out.splitlines() if line.split()}


def measure_surface():
    libdir = None
    try:
        import torch  # noqa: F401  (located here, not in the child: path only)
        libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
    except Exception:
        libdir = ""
    torch_cuda = os.path.join(libdir, "libtorch_cuda.so") if libdir else ""
    result = {"libtorch_cuda": torch_cuda, "exists": os.path.exists(torch_cuda)}
    if not result["exists"]:
        result["error"] = "libtorch_cuda.so not found"
        return result
    need = undefined_symbols(torch_cuda, r"^(cu|cuda)[A-Z]\w+$")
    have = defined_symbols(FAKEGPU) if os.path.exists(FAKEGPU) else set()
    missing = [s for s in need if s not in have]
    result.update({
        "fakegpu": FAKEGPU,
        "fakegpu_exists": os.path.exists(FAKEGPU),
        "torch_cuda_undefined_cu_symbols": len(need),
        "fakegpu_exports": len(have),
        "missing": len(missing),
        "missing_first_20": missing[:20],
        "covered_pct": round(100.0 * (len(need) - len(missing)) / len(need), 1)
        if need else None,
    })
    # and the honest runtime answer: put the shim in front of torch and record
    # what torch actually says, rather than what we infer from symbol counts.
    tmp = tempfile.mkdtemp(prefix="7q3w4-cuda-", dir=os.path.join(REPO, ".lab", "tmp"))
    try:
        for name in ("libcuda.so.1", "libcuda.so"):
            shutil.copyfile(FAKEGPU, os.path.join(tmp, name))
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = tmp + ":" + env.get("LD_LIBRARY_PATH", "")
        env["FAKEGPU_NUM_DEVICES"] = "64"
        rc, rss, err, out = run_child(
            "import torch\n"
            "print('device_count', torch.cuda.device_count())\n"
            "print('is_available', torch.cuda.is_available())\n"
            "try:\n"
            "    print('name0', torch.cuda.get_device_name(0))\n"
            "except Exception as e:\n"
            "    print('name0 raised', type(e).__name__, e)\n", env=env, timeout=300)
        result["torch_with_fakegpu_as_libcuda"] = {
            "rc": rc, "peak_rss_mib": rss, "stdout_tail": out, "stderr_tail": err}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--with-fork", action="store_true",
                    help="also run the fork/COW arm (known to deadlock here)")
    args = ap.parse_args()
    data = {"capacity": measure_capacity(), "surface": measure_surface()}
    if args.with_fork:
        data["fork_cow"] = measure_fork_cow()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    c, s = data["capacity"], data["surface"]
    print("== capacity (risk a) ==")
    print(f"  MemTotal {c['mem_total_mib']} MiB, MemAvailable before "
          f"{c['mem_available_mib_before']} MiB")
    print(f"  child: import torch            rc={c['child_import_torch']['rc']} "
          f"peak RSS {c['child_import_torch']['peak_rss_mib']} MiB")
    print(f"  child: torch + 12M model, 1 step rc="
          f"{c['child_import_torch_12m_model_1step']['rc']} peak RSS "
          f"{c['child_import_torch_12m_model_1step']['peak_rss_mib']} MiB")
    b = c["budget"]
    print(f"  8 nodes need {b['needed_mib_with_model']} MiB with the model; "
          f"fits in {c['mem_available_mib_before']} MiB available: {b['fits_now']}")
    f = data.get("fork_cow")
    if f:
        print("== fork/COW (--with-fork) ==")
        print(f"  one warmed parent forks 8 children: rc={f['rc']} "
              f"peak RSS {f['peak_rss_mib']} MiB  {f['stdout']} {f['stderr']}")
    print("== driver-API surface (risk c) ==")
    print(f"  libtorch_cuda undefined cu*/cuda* symbols: "
          f"{s.get('torch_cuda_undefined_cu_symbols')}")
    print(f"  libfakegpu exports: {s.get('fakegpu_exports')}; covered "
          f"{s.get('covered_pct')}% -> MISSING {s.get('missing')}")
    print(f"  first missing: {', '.join(s.get('missing_first_20', [])[:10])}")
    t = s.get("torch_with_fakegpu_as_libcuda", {})
    print(f"  torch with the shim as libcuda.so.1: rc={t.get('rc')} "
          f"stdout={t.get('stdout_tail')} stderr={t.get('stderr_tail')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
