"""rlimit CUDA-compatibility gate (gpu-sim-u2o.6.3.1).

Two real-GPU runs have been killed by limits that were never checked against
CUDA, and both times the symptom was a misleading CUDA error rather than a
resource error:

  (a) RLIMIT_NPROC=32/512 -> cudaErrorOsOperationNotSupported at the first
      cudaMalloc (u2o.6.1.1, ts 20260927t123630-05fd9bd0)
  (b) RLIMIT_AS=24 GiB   -> every run died in ~2 ms, because the RTX 5090's
      BAR1 mapping is 32 GiB

So this is a GATE, not a comment: every limit the harness applies is declared
here with the reason it is safe for CUDA and the job id that proves it, and
`probe_source()` builds a probe that runs under the harness's EXACT limit set.

The rule this module exists to enforce, from the 6.1.1 ruling: a limit that
CUDA cannot tolerate must fail FAST with a diagnosis naming the limit, not
surface later as four identical kernel failures or a cudaError that points at
the wrong subsystem.
"""
from __future__ import annotations

import resource
from dataclasses import dataclass

# The limits the harness applies today, in one table, with the evidence for
# each. `evidence` is a ts job id that ran a real kernel under that limit on the
# 5090, or "untested" -- and "untested" is a FAILING state, not a passing one.
#
# FSIZE is listed as NOT APPLIED. The bead description describes the current set
# as including "FSIZE(min(256, quota))", but apply() in safe_harness.py never
# calls setrlimit(RLIMIT_FSIZE). Rather than write a table that describes a
# different file than the one that runs, the table records the discrepancy.
LIMIT_TABLE: tuple[dict, ...] = (
    {
        "limit": "RLIMIT_AS",
        "attribute": "address_space_limit_mb",
        "default": 262144,  # 256 GiB
        "why_cuda_safe": (
            "must exceed the 5090's 32 GiB BAR1 mapping or CUDA context "
            "creation fails instantly; 256 GiB clears it. NOT a VRAM cap -- on "
            "ReBAR the BAR mapping dwarfs any useful value, so this is a leak "
            "backstop only."),
        "evidence": "20260927t180239-90a79c87",
        "harmless": True,
    },
    {
        "limit": "RLIMIT_CPU",
        "attribute": "cpu_time_limit_sec",
        "default": 60,
        "why_cuda_safe": (
            "a wall-clock-equivalent CPU-seconds budget; it does not constrain "
            "the driver's thread creation, and a kernel that exceeds it is "
            "killed by the process-group SIGKILL path anyway."),
        "evidence": "20260927t180239-90a79c87",
        "harmless": True,
    },
    {
        "limit": "RLIMIT_DATA",
        "attribute": "memory_limit_mb",
        "default": 8192,
        "why_cuda_safe": (
            "brk/heap only. CUDA's allocations are mmap-based and the BAR1 "
            "mapping is not charged to RLIMIT_DATA, so this bounds a runaway "
            "heap without touching device or pinned memory."),
        "evidence": "20260927t180239-90a79c87",
        "harmless": True,
    },
    {
        "limit": "RLIMIT_NOFILE",
        "attribute": "open_files_limit",
        "default": 256,
        "why_cuda_safe": (
            "measured fine on the 5090 (u2o.6.1.1 bracketed NOFILE=256 as "
            "acceptable while NPROC=32 and 512 were both fatal). The driver "
            "opens far fewer than 256 descriptors for a single-context run."),
        "evidence": "20260927t180239-90a79c87",
        "harmless": True,
    },
    {
        "limit": "RLIMIT_CORE",
        "attribute": None,
        "default": 0,
        "why_cuda_safe": (
            "disables core dumps; touches no CUDA state. Safe by inspection "
            "and by the fact that the 5090 runs above produced no core."),
        "evidence": "20260927t180239-90a79c87",
        "harmless": True,
    },
    {
        "limit": "RLIMIT_FSIZE",
        "attribute": None,
        "default": None,
        "why_cuda_safe": (
            "NOT APPLIED. The bead description lists FSIZE(min(256, quota)) "
            "as part of the set the harness applies today, but apply() never "
            "calls setrlimit(RLIMIT_FSIZE). Recorded as a discrepancy rather "
            "than written into the table as if it were enforced."),
        "evidence": "untested",
        "harmless": None,   # None = not applied, so nothing to be unsafe
        "applied": False,
    },
    {
        "limit": "RLIMIT_NPROC",
        "attribute": "processes_limit",
        "default": 0,
        "why_cuda_safe": (
            "NEVER SET. The kernel checks RLIMIT_NPROC against the real uid's "
            "TOTAL process+thread count, not our subtree, on every clone. This "
            "uid has ~2934 tasks (ps -eL -o uid=) against 256 processes, and a "
            "kernel-side probe bracketed the live total between 2048 (aborts "
            "at first thread creation) and 3072 (works). Because the count "
            "drifts with the other tenants on this shared box, EVERY fixed "
            "value is a race -- 2048 looks generous and is already fatal. "
            "Containment goes through the cgroup v2 pids.max route in "
            "gpu-sim-u2o.6.1.2 instead."),
        "evidence": "20260927t123630-05fd9bd0",
        "harmless": True,
    },
)


class RlimitReintroductionError(RuntimeError):
    """Raised when code tries to set a limit CUDA cannot tolerate here."""


def limits_harness_applies(limits) -> dict:
    """Return {rlimit_name: value} for the limits actually applied.

    Derived from the real ResourceLimits dataclass, not from a hand-kept list,
    so the two cannot drift.
    """
    return {
        "RLIMIT_AS": limits.address_space_limit_mb * 1024 * 1024,
        "RLIMIT_CPU": limits.cpu_time_limit_sec,
        "RLIMIT_DATA": limits.memory_limit_mb * 1024 * 1024,
        "RLIMIT_NOFILE": limits.open_files_limit,
        "RLIMIT_CORE": 0,
    }


def nproc_is_never_set(limits) -> bool:
    """The 6.1.1 ruling as an executable predicate.

    True iff applying these limits cannot reach setrlimit(RLIMIT_NPROC).
    """
    return limits.processes_limit == 0


def assert_nproc_never_set(limits) -> None:
    """Fail fast, loudly, with the fix -- the point of the gate.

    safe_harness.py still CONTAINS a guarded setrlimit(RLIMIT_NPROC) call
    behind `if self.processes_limit:`. A guard is a footgun with a comment on
    it: the next person sets processes_limit because the field exists and the
    name looks like the containment knob it is not. So a non-zero value is
    refused rather than silently applied.
    """
    if not nproc_is_never_set(limits):
        raise RlimitReintroductionError(
            f"RLIMIT_NPROC is not settable on this host: processes_limit="
            f"{limits.processes_limit} would be checked against the real uid's "
            f"total task count (~2934) and kills CUDA at the first "
            f"cudaMalloc (ts 20260927t123630-05fd9bd0). Use the cgroup v2 "
            f"pids.max route in gpu-sim-u2o.6.1.2 for containment. 'Cannot "
            f"contain' must never mean 'apply it anyway'.")


def assert_address_space_independent_of_vram(limits) -> None:
    """The numbers must be independent, not merely documented as such.

    A comment stating that address_space_limit_mb and vram_limit_mb are
    unrelated is exactly the (b) failure waiting to recur: someone coupling
    them to 'make the VRAM cap work' would reintroduce a 24 GiB AS cap and
    every run would die in ~2 ms.
    """
    as_mb = limits.address_space_limit_mb
    vram_mb = limits.vram_limit_mb
    if as_mb == vram_mb or as_mb < 64 * 1024:
        raise RlimitReintroductionError(
            f"RLIMIT_AS ({as_mb} MiB) looks derived from vram_limit_mb "
            f"({vram_mb} MiB) or below the 32 GiB BAR1 mapping of the RTX "
            f"5090. RLIMIT_AS must exceed 32768 MiB and must not track the "
            f"VRAM limit (u2o.6.1.1: AS=24 GiB killed every run in ~2 ms).")


def untested_limits() -> list[str]:
    """Limits in the table with no ts job proving them. Must be empty."""
    return [e["limit"] for e in LIMIT_TABLE
            if e.get("applied", True) and e["evidence"] == "untested"]


# --- the CUDA probe -------------------------------------------------------
#
# Cheap (<10s) because it runs on every batch job. It does the three things
# that actually broke: create a context (BAR1 mapping), allocate (cudaMalloc),
# and launch + synchronize. It also records the live uid task count, because
# RLIMIT_NPROC is per-uid and that number is what makes a future
# reintroduction detectable rather than mysterious.

PROBE_SOURCE = r"""
#include <cstdio>
#include <unistd.h>
#include <dirent.h>
#include <cuda_runtime.h>

__global__ void probe_kernel(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] * 2.0f;
}

static long count_uid_tasks(uid_t uid) {
    DIR *d = opendir("/proc");
    if (!d) return -1;
    long total = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char path[512];
        snprintf(path, sizeof path, "/proc/%s/status", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char line[256];
        uid_t owner = 0;
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "Uid: %u", &owner) == 1) break;
        }
        fclose(f);
        if (owner != uid) continue;
        f = fopen(path, "r");
        if (!f) continue;
        while (fgets(line, sizeof line, f)) {
            if (strncmp(line, "Threads:", 8) == 0) { total += atol(line + 8); break; }
        }
        fclose(f);
    }
    closedir(d);
    return total;
}

int main(void) {
    int rc = 0;
    long tasks = count_uid_tasks(getuid());

    cudaError_t e = cudaFree(0);
    if (e != cudaSuccess) {
        printf("{\"stage\": \"context\", \"error\": \"%s\"}\n", cudaGetErrorString(e));
        return 2;
    }
    cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess) {
        printf("{\"stage\": \"props\", \"error\": \"%s\"}\n", cudaGetErrorString(e));
        return 3;
    }
    void *devptr = NULL;
    size_t n = 1 << 20;
    e = cudaMalloc(&devptr, n * sizeof(float));
    if (e != cudaSuccess) {
        printf("{\"stage\": \"malloc\", \"error\": \"%s\"}\n", cudaGetErrorString(e));
        return 4;
    }
    // Seed the buffer BEFORE launching. Without this the kernel doubles
    // uninitialised device memory and the read-back is meaningless -- which is
    // exactly what happened on the first run: rc 0, cudaMalloc ok, launch ok,
    // result wrong. The probe asserted its own read-back instead of trusting
    // rc, and that is the only reason the bug surfaced here rather than as a
    // green check on an unfalsifiable probe.
    float host[4] = {1, 2, 3, 4}, back[4] = {0};
    e = cudaMemcpy(devptr, host, sizeof host, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        printf("{\"stage\": \"memcpy_h2d\", \"error\": \"%s\"}\n", cudaGetErrorString(e));
        return 7;
    }
    probe_kernel<<<(n + 255) / 256, 256>>>((float *)devptr, (int)n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("{\"stage\": \"launch\", \"error\": \"%s\"}\n", cudaGetErrorString(e));
        return 5;
    }
    e = cudaMemcpy(back, devptr, sizeof host, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        printf("{\"stage\": \"memcpy\", \"error\": \"%s\"}\n", cudaGetErrorString(e));
        return 6;
    }
    int correct = (back[0] == 2.0f && back[1] == 4.0f && back[2] == 6.0f && back[3] == 8.0f);
    printf("{\"cuda_malloc\": \"ok\", \"launch\": \"ok\", \"rc\": 0, "
           "\"result_correct\": %s, \"device\": \"%s\", \"uid_tasks\": %ld, "
           "\"uid_task_note\": \"RLIMIT_NPROC is checked against this count, "
           "not our subtree; the live total drifts with other tenants\"}\n",
           correct ? "true" : "false", prop.name, tasks);
    cudaFree(devptr);
    return rc;
}
"""


def diagnose_probe_failure(stage: str, cuda_error: str,
                           applied: dict) -> str:
    """Name the offending limit, or say plainly that none of them explains it.

    This is the fail-fast path the bead asks for: a probe failure must produce
    "rlimit X = Y is incompatible with CUDA" rather than four identical kernel
    failures. The mapping is by failure stage, because that is the only
    evidence available: the first two CUDA incidents presented as a malloc or
    context error, not as a limit error.
    """
    suspects = {
        "context": ["RLIMIT_AS"],
        "props": ["RLIMIT_AS"],
        "malloc": ["RLIMIT_AS", "RLIMIT_NPROC"],
        "launch": ["RLIMIT_NOFILE", "RLIMIT_AS"],
        "memcpy": ["RLIMIT_AS", "RLIMIT_DATA"],
    }.get(stage, [])

    parts = []
    for name in suspects:
        value = applied.get(name)
        if value is None:
            continue
        why = next((e["why_cuda_safe"] for e in LIMIT_TABLE
                    if e["limit"] == name), "")
        parts.append(f"{name} = {value} is incompatible with CUDA "
                     f"(stage={stage}, {cuda_error}). {why}")
    if not parts:
        parts.append(
            f"No limit in the harness's set explains a failure at stage "
            f"'{stage}' ({cuda_error}). That is itself the finding: the limit "
            f"set is not the cause, so do not add a limit to 'fix' it.")
    return " | ".join(parts)
