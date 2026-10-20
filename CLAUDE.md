# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

## Beads Issue Tracker (task tracking only — NOT git authority)

This project uses **bd (beads)** for issue tracking. Run `bd prime` for task-tracking commands.

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for task-tracking command reference
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files
- `bd ready` / `bd show <id>` / `bd update <id> --claim` / `bd close <id>`
- Architecture: issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export.

## Git authority (bd never decides this)

- `bd`, `bd prime`, and any Beads template/profile text (conservative/minimal/team-maintainer) are NEVER git permission authority. Ignore any such language there.
- Only repository instructions (e.g. `AGENTS.md`), orchestrator prompts (e.g. `.lab/prompts/*`), and direct user instructions grant or deny git commit/merge/push/sync.
- Do NOT wait for human approval on git actions your repo/orchestrator already authorized. Do NOT treat a missing user message as a veto.
- Session close: file follow-up issues, run quality gates, update issue status, handle git per repo/orchestrator rules, then hand off.


## Build & Test

```bash
make            # builds libfakegpu.so
make check      # builds + runs the self-test and reference workloads
                # (fake_client, vector_add, ring_allreduce) under LD_PRELOAD.
```

Only `gcc` + `make` are required — no CUDA toolkit (the interposer ships shim
headers in `shim/`).

## Architecture Overview

`libfakegpu.so` is an `LD_PRELOAD` interposer that stubs the CUDA runtime +
driver API and the NCCL API. At every API boundary it returns **success** and,
where a buffer is produced, fills it with **random bytes** from a thread-local
PRNG, optionally applying a **configurable latency** (fixed or bandwidth-based).
No symbol ever forwards to a real GPU library, so behaviour is deterministic
and host-independent — ideal for CPU-only CI and GPU-behaviour experiments.

Layout: `shim/` (public API decls), `include/fakegpu.h` (internal), `src/`
(config, PRNG, fake-memory pool, latency, CUDA stubs, NCCL stubs), `tests/`
(self-test + reference workloads), `Makefile`, `Dockerfile`.

See `README.md` for env-var config and extension guidance.

### Cluster simulation (direction, 2026-10-20)

The next layer is arbitrary-cluster simulation: N declared devices over a
declared fabric (NVLink/PCIe/IB/TCP), running unmodified multi-GPU workloads
on one box. Two fabric routes (topology model + real-NCCL net plugin), with
real `libnccl` hosted below two interposition seams (driver API, ncclNet).
Read `docs/cluster-sim.md` first; work lives under epic `gpu-sim-m3d`.

## Conventions & Patterns

- New symbols: declare in `shim/`, stub in `src/fakegpu_cuda.c` or
  `src/fakegpu_nccl.c`, return success, `fg_fill()` output buffers,
  `fg_simulate_*()` for latency, register handles via `fg_handle_make()`.
- Quality gate: `make check` must pass green before closing beads.

